#!/usr/bin/env python3
# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Refuse a wheel that contains anything but the git-tracked package files.

    python scripts/check_wheel.py dist/sphota-*.whl

Guards the release against a tampered build (the LiteLLM 2026 compromises shipped a ``.pth``
startup hook that runs on every Python start). Checks: no ``.pth`` files, no files outside
``sphota/`` and the dist-info, every package file is tracked by git and byte-identical to the
checked-out source, and nothing tracked under ``src/sphota`` is missing.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tracked() -> dict[str, str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "src/sphota"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    files = {}
    for p in filter(None, out.split("\0")):
        if "__pycache__" in p:
            continue
        files[p.removeprefix("src/")] = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    return files


def check(wheel: Path) -> list[str]:
    want = tracked()
    problems = []
    seen = set()
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            if name.endswith(".pth"):
                problems.append(f"startup hook (.pth) in wheel: {name}")
            if ".dist-info/" in name:
                if not name.split("/")[0].startswith("sphota-"):
                    problems.append(f"unexpected dist-info: {name}")
                continue
            if not name.startswith("sphota/"):
                problems.append(f"file outside the package: {name}")
                continue
            digest = hashlib.sha256(z.read(name)).hexdigest()
            if name not in want:
                problems.append(f"not tracked by git: {name}")
            elif want[name] != digest:
                problems.append(f"differs from the git source: {name}")
            seen.add(name)
    problems += [f"missing from wheel: {n}" for n in sorted(set(want) - seen)]
    return problems


if __name__ == "__main__":
    wheels = [Path(a) for a in sys.argv[1:]]
    if not wheels:
        sys.exit("usage: check_wheel.py dist/sphota-*.whl")
    bad = [f"{w.name}: {p}" for w in wheels for p in check(w)]
    print("\n".join(bad) if bad else f"ok: {', '.join(w.name for w in wheels)}")
    sys.exit(1 if bad else 0)
