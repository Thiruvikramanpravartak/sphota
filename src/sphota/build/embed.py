# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Embedding with sentence-transformers + an on-disk cache (chunked and resumable)."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from sphota.build.common import DATA_DIR, HF_CACHE, log

# Per-model input conventions. e5 models need a "query: " prefix for symmetric tasks.
PREFIX = {
    "intfloat/multilingual-e5-small": "query: ",
    "intfloat/multilingual-e5-base": "query: ",
    "intfloat/e5-small-v2": "query: ",
}
SHORT = {
    "BAAI/bge-m3": "bge-m3",
    "sentence-transformers/LaBSE": "labse",
    "intfloat/multilingual-e5-small": "me5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "pmMiniLM",
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6",
    "BAAI/bge-small-en-v1.5": "bge-small-en",
    "thenlper/gte-small": "gte-small",
    "Qwen/Qwen3-Embedding-0.6B": "qwen3-0.6b",
}

_MODELS: dict[str, object] = {}


def short(model: str) -> str:
    return SHORT.get(model, model.replace("/", "__"))


def load(model: str):
    if model not in _MODELS:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(int(os.environ.get("SPHOTA_THREADS", "4")))
        _MODELS[model] = SentenceTransformer(model, cache_folder=str(HF_CACHE), device="cpu")
    return _MODELS[model]


def encode(model: str, texts: list[str], batch_size: int = 64) -> np.ndarray:
    m = load(model)
    pre = PREFIX.get(model, "")
    # sort by length for fast padding, then restore order
    order = np.argsort([len(t) for t in texts])
    inp = [pre + texts[i] for i in order]
    emb = m.encode(
        inp,
        batch_size=batch_size,
        normalize_embeddings=True,  # type: ignore[attr-defined]
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    out = np.empty_like(emb)
    out[order] = emb
    return out.astype(np.float32)


def cached(model: str, name: str, texts: list[str], scale: str, chunk: int = 2048) -> np.ndarray:
    """Encode ``texts`` with ``model``; cache in ``data/emb/<scale>/<model>/<name>.npy``.

    Encoding is chunked; partial chunks survive interruption so long jobs resume.
    """
    h = hashlib.sha1("\x00".join(texts).encode()).hexdigest()[:10]
    d = DATA_DIR / "emb" / scale / short(model)
    d.mkdir(parents=True, exist_ok=True)
    final = d / f"{name}-{h}.npy"
    if final.exists():
        return np.load(final)
    parts = []
    t0 = time.time()
    for s in range(0, len(texts), chunk):
        p = d / f".{name}-{h}.part{s // chunk:05d}.npy"
        if p.exists():
            parts.append(np.load(p))
            continue
        e = encode(model, texts[s : s + chunk])
        np.save(p, e)
        parts.append(e)
        done = s + len(e)
        rate = done / max(time.time() - t0, 1e-9)
        log(f"{short(model)}/{name}: {done}/{len(texts)} ({rate:.0f} sent/s)")
    emb = np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)
    np.save(final, emb)
    for p in d.glob(f".{name}-{h}.part*.npy"):
        p.unlink()
    return emb


def speed(model: str, texts: list[str]) -> float:
    """Sentences per second on this machine (CPU)."""
    encode(model, texts[:16])  # warm-up
    t0 = time.time()
    encode(model, texts)
    return len(texts) / (time.time() - t0)


def path_of(model: str, name: str, scale: str) -> Path | None:
    d = DATA_DIR / "emb" / scale / short(model)
    hits = sorted(d.glob(f"{name}-*.npy"))
    return hits[0] if hits else None
