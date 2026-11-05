# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Loading the frozen release artifacts (numpy only), with checksum verification.

A release directory contains ``manifest.json`` plus ``.npz`` / ``.json`` files listed in it
with their SHA-256. Only ``.npz`` (loaded with ``allow_pickle=False``) and ``.json`` are ever
read, so a release or adapter file cannot execute code.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_RELEASE_DIR = Path(__file__).parent / "data" / "v1"


class ReleaseError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@dataclass
class Release:
    """The frozen v1 coordinate space: projection, codebooks, calibration, labels, tables."""

    root: Path
    manifest: dict[str, Any]
    mu: np.ndarray
    W: np.ndarray
    codebooks: list[np.ndarray]
    temperatures: np.ndarray
    noise_scale: float
    labels: dict[str, dict[str, str]] = field(default_factory=dict)
    l1_neighbors: np.ndarray | None = None
    l1_neighbor_dist: np.ndarray | None = None
    _cnorms: list[np.ndarray] = field(default_factory=list)

    @property
    def version(self) -> str:
        return str(self.manifest["version"])

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(int(c.shape[0]) for c in self.codebooks)

    @property
    def dim(self) -> int:
        return int(self.W.shape[1])

    @property
    def anchor(self) -> dict[str, Any]:
        return dict(self.manifest["anchor"])

    def cnorm(self, level: int) -> np.ndarray:
        if not self._cnorms:
            self._cnorms = [(c.astype(np.float64) ** 2).sum(1) for c in self.codebooks]
        return self._cnorms[level]


def _verify(root: Path, manifest: dict[str, Any]) -> None:
    for name, meta in manifest.get("files", {}).items():
        p = root / name
        if not p.exists():
            raise ReleaseError(
                f"release file missing: {p}. Reinstall sphota or re-run `sphota bundle`."
            )
        got = sha256_file(p)
        if got != meta["sha256"]:
            raise ReleaseError(
                f"checksum mismatch for {p} (expected {meta['sha256'][:12]}…, got {got[:12]}…). "
                "The file was modified or corrupted; reinstall sphota."
            )


def release_dir() -> Path:
    env = os.environ.get("SPHOTA_RELEASE_DIR")
    return Path(env) if env else PACKAGE_RELEASE_DIR


@lru_cache(maxsize=4)
def load_release(root: str | None = None, verify: bool = True) -> Release:
    r = Path(root) if root else release_dir()
    mpath = r / "manifest.json"
    if not mpath.exists():
        raise ReleaseError(
            f"no Sphota release found at {r}. If you installed from source, run "
            "`python -m sphota.build.release --scale small` or set SPHOTA_RELEASE_DIR."
        )
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if verify:
        _verify(r, manifest)
    space = load_npz(r / "space.npz")
    cb = load_npz(r / "codebooks.npz")
    calib = json.loads((r / "calibration.json").read_text(encoding="utf-8"))
    labels_path = r / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    nb_path = r / "neighbors.npz"
    nb = load_npz(nb_path) if nb_path.exists() else {}
    n_levels = len([k for k in cb if k.startswith("level")])
    return Release(
        root=r,
        manifest=manifest,
        mu=space["mu"].astype(np.float32),
        W=space["W"].astype(np.float32),
        codebooks=[cb[f"level{i + 1}"].astype(np.float32) for i in range(n_levels)],
        temperatures=np.asarray(calib["temperatures"], dtype=np.float64),
        noise_scale=float(calib.get("noise_scale", 1.0)),
        labels=labels,
        l1_neighbors=nb.get("l1_neighbors"),
        l1_neighbor_dist=nb.get("l1_neighbor_dist"),
    )
