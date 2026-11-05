# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Adapters: map another model's embeddings into the Sphota v1 canonical space.

Adapter file format (spec: docs/spec/07-adapters-registry.md): a ``.npz`` with

* ``A``    float32 (d_model, D): linear map applied to L2-normalized model vectors,
* ``b``    float32 (D,): bias,
* ``meta`` 0-d unicode array holding a JSON object (model id, prefix, space version, ...).

Optional (``kind == "mlp"``): ``A2`` (D, D), ``b2`` (D,) for one residual layer
``y = h + relu(h) @ A2 + b2`` where ``h = x @ A + b``. Loaded with ``allow_pickle=False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sphota.release import ReleaseError, load_npz, load_release, sha256_file


class AdapterError(RuntimeError):
    pass


@dataclass
class Adapter:
    model: str
    A: np.ndarray
    b: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    A2: np.ndarray | None = None
    b2: np.ndarray | None = None

    @property
    def prefix(self) -> str:
        return str(self.meta.get("prefix", ""))

    @property
    def in_dim(self) -> int:
        return int(self.A.shape[0])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if X.shape[1] != self.in_dim:
            raise AdapterError(
                f"adapter for {self.model} expects {self.in_dim}-d vectors, got {X.shape[1]}-d. "
                "Did you pass vectors from a different model?"
            )
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        H = X @ self.A + self.b
        if self.A2 is not None and self.b2 is not None:
            H = H + np.maximum(H, 0) @ self.A2 + self.b2
        return H.astype(np.float32)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "A": self.A.astype(np.float32),
            "b": self.b.astype(np.float32),
            "meta": np.array(json.dumps({"model": self.model, **self.meta}, sort_keys=True)),
        }
        if self.A2 is not None and self.b2 is not None:
            arrays["A2"] = self.A2.astype(np.float32)
            arrays["b2"] = self.b2.astype(np.float32)
        with open(path, "wb") as f:
            savez: Any = np.savez  # numpy's stubs reject **kwargs of arrays
            savez(f, **arrays)
        return path


def load_adapter(path: str | Path, sha256: str | None = None) -> Adapter:
    path = Path(path)
    if path.suffix != ".npz":
        raise AdapterError(f"adapters must be .npz files (got {path.name}); pickles are refused")
    if sha256 is not None and sha256_file(path) != sha256:
        raise AdapterError(f"checksum mismatch for adapter {path}; refusing to load it")
    z = load_npz(path)
    meta = json.loads(str(z["meta"]))
    return Adapter(
        model=meta.pop("model"), A=z["A"], b=z["b"], meta=meta, A2=z.get("A2"), b2=z.get("b2")
    )


def registry(release_root: Path | None = None) -> dict[str, Any]:
    rel = load_release(str(release_root) if release_root else None)
    p = rel.root / "adapters" / "registry.json"
    if not p.exists():
        return {"adapters": {}}
    return dict(json.loads(p.read_text(encoding="utf-8")))


def list_adapters() -> list[str]:
    return sorted(registry()["adapters"])


def registry_entry(model: str) -> dict[str, Any] | None:
    """Registry entry for an HF id or short name (None if unknown)."""
    reg = registry()["adapters"]
    if model in reg:
        return dict(reg[model])
    for v in reg.values():
        if v.get("short") == model:
            return dict(v)
    return None


@lru_cache(maxsize=32)
def get_adapter(model: str) -> Adapter:
    """Registry adapter for ``model`` (HF id, short name, or path to an ``.npz``)."""
    if model.endswith(".npz") and Path(model).exists():
        return load_adapter(model)
    reg = registry()["adapters"]
    key = model
    if key not in reg:
        by_short = {v.get("short"): k for k, v in reg.items()}
        key = by_short.get(model, model)
    if key not in reg:
        raise AdapterError(
            f"no certified adapter for {model!r}. Available: {', '.join(sorted(reg)) or 'none'}. "
            f"Train one with: sphota train-adapter --model {model}"
        )
    entry = reg[key]
    rel = load_release()
    try:
        return load_adapter(rel.root / "adapters" / entry["file"], sha256=entry["sha256"])
    except ReleaseError as e:  # pragma: no cover
        raise AdapterError(str(e)) from e
