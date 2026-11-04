# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Air-gapped bundles: ``sphota bundle`` copies the release and the encoder model files into one
directory with a checksummed manifest. Use with ``SPHOTA_OFFLINE=1 SPHOTA_BUNDLE=<dir>``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from sphota.release import load_release, sha256_file

ENCODER_FILES = {
    "intfloat/multilingual-e5-small": ["onnx/model.onnx", "tokenizer.json"],
}
INT8_FILE = "onnx/model_qint8_avx512_vnni.onnx"


def make_bundle(out: str | Path, int8: bool = False) -> Path:
    from sphota.encoders import hf_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rel = load_release()
    dst_rel = out / "release" / rel.version
    if dst_rel.exists():
        shutil.rmtree(dst_rel)
    shutil.copytree(rel.root, dst_rel, ignore=shutil.ignore_patterns("__pycache__"))
    files: dict[str, str] = {}
    for repo, names in ENCODER_FILES.items():
        for name in [*names, *([INT8_FILE] if int8 else [])]:
            src = hf_file(repo, name)
            dst = out / "models" / repo.replace("/", "__") / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            files[str(dst.relative_to(out))] = sha256_file(dst)
    for p in sorted(dst_rel.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(out))] = sha256_file(p)
    (out / "bundle.json").write_text(
        json.dumps(
            {
                "space": rel.version,
                "files": files,
                "usage": f"SPHOTA_OFFLINE=1 SPHOTA_BUNDLE={out} SPHOTA_RELEASE_DIR={dst_rel}",
            },
            indent=2,
        )
    )
    return out


def verify_bundle(path: str | Path) -> list[str]:
    """Return a list of problems (empty if the bundle is intact)."""
    path = Path(path)
    meta = json.loads((path / "bundle.json").read_text())
    bad = []
    for name, digest in meta["files"].items():
        p = path / name
        if not p.exists():
            bad.append(f"missing {name}")
        elif sha256_file(p) != digest:
            bad.append(f"checksum mismatch {name}")
    return bad
