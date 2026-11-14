# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Adapters: map another model's embeddings into the Sphota v1 canonical space.

Adapter file format (spec: docs/spec/07-adapters-registry.md): a ``.npz`` with

* ``A``    float32 (d_model, D): linear map applied to L2-normalized model vectors,
* ``b``    float32 (D,): bias,
* ``meta`` 0-d unicode array holding a JSON object (model id, prefix, space version, ...).

Optional (``kind == "mlp"``): ``A2`` (D, D), ``b2`` (D,) for one residual layer
``y = h + relu(h) @ A2 + b2`` where ``h = x @ A + b``.

Optional (``kind == "mlp2"``, adapter set v1.1, D-028): ``W1`` (d_model, H), ``c1`` (H,),
``W2`` (H, D), ``c2`` (D,) for a residual network on the input:
``y = x @ A + b + gelu(x @ W1 + c1) @ W2 + c2`` (tanh-approximated GELU, numpy only).
Arrays may be stored as float16; they are used as float32. Loaded with ``allow_pickle=False``.

Adapter sets: ``v1.2`` (default since D-053: the v1.1 recipe plus short-command training pairs;
same-cell agreement on held-out commands +23 points, sentences unchanged, D-052), ``v1.1``
(nonlinear, sentences only, D-028/D-030) and ``v1`` (the original linear registry frozen inside the
release). Choose with ``adapters="v1.1"`` or ``SPHOTA_ADAPTERS=v1.1``.
"""

from __future__ import annotations

import json
import os
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
    W1: np.ndarray | None = None
    c1: np.ndarray | None = None
    W2: np.ndarray | None = None
    c2: np.ndarray | None = None

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
        if self.W1 is not None and self.c1 is not None and self.W2 is not None:
            Z = X @ self.W1 + self.c1
            Z = 0.5 * Z * (1.0 + np.tanh(0.7978845608028654 * (Z + 0.044715 * Z**3)))
            H = H + Z @ self.W2 + (self.c2 if self.c2 is not None else 0.0)
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
        for k in ("W1", "c1", "W2", "c2"):
            v = getattr(self, k)
            if v is not None:
                arrays[k] = v  # stored as given (the v1.1 set uses float16)
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
    f32 = {k: z[k].astype(np.float32) for k in ("W1", "c1", "W2", "c2") if k in z}
    return Adapter(
        model=meta.pop("model"),
        A=z["A"].astype(np.float32),
        b=z["b"].astype(np.float32),
        meta=meta,
        A2=z.get("A2"),
        b2=z.get("b2"),
        **f32,
    )


ADAPTER_SETS = ("v1", "v1.1", "v1.2")
# D-053: v1.2 (v1.1 + short-command training pairs) passed its pre-registered rule (D-052); still
# before the first public release, so no stored codes change meaning
DEFAULT_SET = "v1.2"


def _set_name(adapters: str | None) -> str:
    name = adapters or os.environ.get("SPHOTA_ADAPTERS") or DEFAULT_SET
    if name not in ADAPTER_SETS:
        raise AdapterError(f"unknown adapter set {name!r}; use one of {', '.join(ADAPTER_SETS)}")
    return name


def _set_dir(name: str) -> Path:
    if name == "v1":
        return load_release().root / "adapters"
    return Path(__file__).parent / "data" / "adapters" / name


def registry(release_root: Path | None = None, adapters: str | None = None) -> dict[str, Any]:
    """Adapter registry of a set (``v1``: frozen, inside the release; ``v1.1`` / ``v1.2``:
    nonlinear)."""
    name = _set_name(adapters)
    if release_root is not None and name == "v1":
        p = load_release(str(release_root)).root / "adapters" / "registry.json"
    else:
        p = _set_dir(name) / "registry.json"
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


def get_adapter(model: str, adapters: str | None = None) -> Adapter:
    """Registry adapter for ``model`` (HF id, short name, or path to an ``.npz``).

    ``adapters`` picks the set: ``"v1.2"`` (default, nonlinear, trained with short commands too),
    ``"v1.1"`` (nonlinear, sentences only) or ``"v1"`` (frozen, linear);
    ``SPHOTA_ADAPTERS`` sets the default for a process.
    """
    return _get_adapter(model, _set_name(adapters))


@lru_cache(maxsize=32)
def _get_adapter(model: str, set_name: str) -> Adapter:
    if model.endswith(".npz") and Path(model).exists():
        return load_adapter(model)
    reg = registry(adapters=set_name)["adapters"]
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
    try:
        return load_adapter(_set_dir(set_name) / entry["file"], sha256=entry["sha256"])
    except ReleaseError as e:  # pragma: no cover
        raise AdapterError(str(e)) from e
