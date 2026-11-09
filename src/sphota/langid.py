# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Language identification in the anchor space (NumPy only), used by per-file refinement.

A multinomial logistic regression on L2-normalized raw anchor vectors (multilingual-e5-small),
trained on MASSIVE train utterances (51 locales) and FLORES-200 dev (``sphota.build.langid``,
D-038). It needs no extra model: the anchor vector of each message is already computed.

:func:`language_keys` gives each message a grouping key for refinement: its writing system when
only one common language uses it (Tamil, Greek, Georgian, ...), or writing system + predicted
language when several do (Latin, Cyrillic, Arabic, Devanagari, ...).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from sphota.refine import script_of

DIR = Path(__file__).parent / "data" / "langid"
SHARED_SCRIPTS = {"LATIN", "CYRILLIC", "ARABIC", "DEVANAGARI", "BENGALI", "CJK"}


@functools.lru_cache(maxsize=1)
def _model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sphota.release import sha256_file

    meta = json.loads((DIR / "langid.json").read_text(encoding="utf-8"))
    p = DIR / "langid.npz"
    if sha256_file(p) != meta["sha256"]:
        raise RuntimeError(f"checksum mismatch for {p}; reinstall sphota (refusing to load it)")
    with np.load(p, allow_pickle=False) as z:
        return z["W"].astype(np.float32), z["b"].astype(np.float32), z["classes"]


def predict(X: np.ndarray) -> np.ndarray:
    """Predicted MASSIVE locale (e.g. ``"en"``, ``"hi"``, ``"zh-CN"``) for raw anchor vectors."""
    W, b, classes = _model()
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return classes[np.argmax(X @ W + b, axis=1)]


def language_keys(texts: list[str], X: np.ndarray) -> np.ndarray:
    """Per-message grouping key: script, plus predicted language for shared scripts."""
    scripts = [script_of(t) for t in texts]
    need = [i for i, s in enumerate(scripts) if s in SHARED_SCRIPTS]
    pred = predict(np.asarray(X)[need]) if need else []
    keys = np.array(scripts, dtype=object)
    for i, p in zip(need, pred):
        keys[i] = f"{scripts[i]}:{p}"
    return keys.astype(str)
