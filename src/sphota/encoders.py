# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Text encoders. Default: the v1 anchor through ONNX Runtime (no torch).

* :class:`OnnxEncoder`: HF ``tokenizers`` + ``onnxruntime``; mean pooling + L2 norm.
* :class:`TorchEncoder`: any ``sentence-transformers`` model (optional extra ``[torch]``).

Models are downloaded on first use into the Hugging Face cache and work offline afterwards.
``SPHOTA_OFFLINE=1`` forbids downloads; ``SPHOTA_BUNDLE`` points to an offline bundle made by
``sphota bundle``.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import numpy as np


class EncoderError(RuntimeError):
    pass


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def offline() -> bool:
    return os.environ.get("SPHOTA_OFFLINE", "").lower() in {"1", "true", "yes"}


def _bundle_dir() -> Path | None:
    b = os.environ.get("SPHOTA_BUNDLE")
    return Path(b) if b else None


def hf_file(repo: str, filename: str, revision: str | None = None) -> Path:
    """Resolve a file from an offline bundle or the HF Hub cache (downloading if allowed)."""
    b = _bundle_dir()
    if b is not None:
        p = b / "models" / repo.replace("/", "__") / filename
        if p.exists():
            return p
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - depends on extras
        raise EncoderError(
            "encoding text needs the encoder extra: pip install 'sphota[encode]'"
        ) from e
    if offline():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return Path(hf_hub_download(repo, filename, revision=revision, local_files_only=offline()))
    except Exception as e:
        if offline():
            raise EncoderError(
                f"{repo}/{filename} is not cached and SPHOTA_OFFLINE=1. Run `sphota bundle` on a "
                "connected machine and set SPHOTA_BUNDLE, or unset SPHOTA_OFFLINE."
            ) from e
        raise EncoderError(
            f"could not download {repo}/{filename}: {e}. Check your connection or proxy; "
            "once downloaded the model is cached and works offline."
        ) from e


class _LRU:
    def __init__(self, size: int = 4096):
        self.size = size
        self.d: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, k: str) -> np.ndarray | None:
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v

    def put(self, k: str, v: np.ndarray) -> None:
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.size:
            self.d.popitem(last=False)


class _CachedEncoder:
    name: str
    dim: int

    def __init__(self, cache_size: int = 4096):
        self._cache = _LRU(cache_size)

    def _encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out: list[np.ndarray | None] = [self._cache.get(t) for t in texts]
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            uniq = list(dict.fromkeys(texts[i] for i in todo))
            vecs = self._encode(uniq)
            m = dict(zip(uniq, vecs))
            for t, v in m.items():
                self._cache.put(t, v)
            for i in todo:
                out[i] = m[texts[i]]
        return np.stack(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)


class OnnxEncoder(_CachedEncoder):
    """ONNX Runtime sentence encoder with mean pooling (e5 / MiniLM style models)."""

    def __init__(
        self,
        repo: str = "intfloat/multilingual-e5-small",
        onnx_file: str = "onnx/model.onnx",
        tokenizer_file: str = "tokenizer.json",
        prefix: str = "query: ",
        max_length: int = 512,
        batch_size: int = 32,
        threads: int | None = None,
        name: str | None = None,
        pooling: str = "mean",
    ):
        super().__init__()
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise EncoderError(
                "encoding text needs onnxruntime + tokenizers: pip install 'sphota[encode]'"
            ) from e
        self.name = name or repo
        self.prefix = prefix
        self.batch_size = batch_size
        if pooling not in {"mean", "cls"}:
            raise EncoderError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
        self.pooling = pooling
        tok = Tokenizer.from_file(str(hf_file(repo, tokenizer_file)))
        tok.enable_truncation(max_length=max_length)
        pad = "[PAD]" if tok.token_to_id("[PAD]") is not None else "<pad>"
        tok.enable_padding(pad_id=tok.token_to_id(pad) or 0, pad_token=pad)
        self.tok = tok
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(hf_file(repo, onnx_file)), so, providers=["CPUExecutionProvider"]
        )
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.dim = int(self.sess.get_outputs()[0].shape[-1] or 384)

    def _encode(self, texts: list[str]) -> np.ndarray:
        order = np.argsort([len(t) for t in texts])
        res = np.zeros((len(texts), self.dim), np.float32)
        for s in range(0, len(texts), self.batch_size):
            ids = order[s : s + self.batch_size]
            enc = self.tok.encode_batch([self.prefix + texts[i] for i in ids])
            input_ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": input_ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(input_ids)
            hidden = self.sess.run(None, feed)[0]  # (b, t, h)
            if self.pooling == "cls":
                pooled = hidden[:, 0, :].astype(np.float32)
            else:
                m = mask[..., None].astype(np.float32)
                pooled = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            res[ids] = pooled
        return res


class TorchEncoder(_CachedEncoder):
    """Any sentence-transformers model (needs ``pip install 'sphota[torch]'``)."""

    def __init__(self, model: str, prefix: str = "", batch_size: int = 64, device: str = "cpu"):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EncoderError(
                f"encoding with {model} needs sentence-transformers: pip install 'sphota[torch]'"
            ) from e
        if offline():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.name = model
        self.prefix = prefix
        self.batch_size = batch_size
        self.m = SentenceTransformer(model, device=device)
        self.dim = int(self.m.get_sentence_embedding_dimension() or 0)

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = self.m.encode(
            [self.prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(v, dtype=np.float32)


_DEFAULT: dict[str, Encoder] = {}


def default_encoder(variant: str | None = None) -> Encoder:
    """The v1 reference encoder (anchor model, fp32 ONNX). ``variant='int8'`` is 4x smaller."""
    variant = variant or os.environ.get("SPHOTA_ENCODER", "fp32")
    if variant not in _DEFAULT:
        if variant == "fp32":
            _DEFAULT[variant] = OnnxEncoder(name="intfloat/multilingual-e5-small")
        elif variant == "int8":
            _DEFAULT[variant] = OnnxEncoder(
                onnx_file="onnx/model_qint8_avx512_vnni.onnx",
                name="intfloat/multilingual-e5-small@int8",
            )
        elif variant == "torch":
            _DEFAULT[variant] = TorchEncoder("intfloat/multilingual-e5-small", prefix="query: ")
        else:
            raise EncoderError(f"unknown SPHOTA_ENCODER={variant!r}; use fp32, int8 or torch")
    return _DEFAULT[variant]
