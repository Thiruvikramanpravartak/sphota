#!/usr/bin/env python3
# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Write tests/fixtures/refine_parity.json: synthetic inputs and the Python refinement's outputs.

The same fixture is checked by tests/test_refine.py (Python) and atlas/tests/refine-parity.mjs
(TypeScript), so the library and the Meaning Atlas cannot drift apart (D-042, D-048).
"""

import json
from pathlib import Path

import numpy as np

from sphota.refine import refine

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(7)
    dim, n_lang, per = 32, 4, 60
    topics = rng.normal(size=(12, dim)) * 2.0
    accents = rng.normal(size=(n_lang, dim)) * 1.2
    rows, keys = [], []
    for li in range(n_lang):
        t = rng.integers(0, 12, per)
        rows.append(topics[t] + accents[li] + rng.normal(size=(per, dim)) * 0.6)
        keys += [f"LATIN:l{li}"] * per
    V = np.concatenate(rows)
    V = (V / np.linalg.norm(V, axis=1, keepdims=True)).astype(np.float32).astype(np.float64)
    l1 = rng.integers(0, 50, len(V))
    out = {"V": V.tolist(), "keys": keys, "l1": l1.tolist(), "expected": {}}
    for detail, rule in [("fine", "2sqrt"), ("medium", "sqrt"), ("coarse", "0.5sqrt")]:
        r = refine(V, "kmeans", scripts=np.array(keys), language="centre", k_rule=rule, l1=l1)
        out["expected"][detail] = {
            "labels": r.labels.tolist(),
            "parent": r.parent,
            "representative": r.representative,
        }
    p = ROOT / "tests" / "fixtures" / "refine_parity.json"
    p.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
