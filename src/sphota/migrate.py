# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Version migration tables (spec Part 6).

``build_table`` encodes reference texts (or anchor vectors) with two release directories and
records, for each source prefix, the distribution of target prefixes. ``apply_table`` maps
stored codes through a table. v1 is frozen, so today this tooling is exercised by tests that
build a table between s1 and a perturbed copy of s1 (see tests/test_migrate.py).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sphota.codes import from_indices
from sphota.release import load_release
from sphota.space import project, quantize


def table_from_vectors(
    src_dir: str | Path, dst_dir: str | Path, X: np.ndarray, level: int = 2
) -> dict:
    """Migration table from anchor vectors ``X`` (n, d) encoded in both spaces."""
    a = load_release(str(src_dir))
    b = load_release(str(dst_dir))
    ia = quantize(a, project(a, X), levels=level).indices
    ib = quantize(b, project(b, X), levels=level).indices
    out = {}
    for lv in range(1, level + 1):
        counts: dict[str, Counter] = defaultdict(Counter)
        for ra, rb in zip(ia[:, :lv], ib[:, :lv]):
            counts[str(from_indices(ra))][str(from_indices(rb))] += 1
        out[str(lv)] = {
            k: [[t, round(n / sum(c.values()), 4)] for t, n in c.most_common(5)]
            for k, c in counts.items()
        }
    top1 = [v[0][1] for v in out[str(level)].values()]
    return {
        "from": a.version,
        "to": b.version,
        "levels": out,
        "n_prefixes": len(out[str(level)]),
        "mean_top1": float(np.mean(top1)) if top1 else 0.0,
    }


def build_table(
    src_dir: str | Path,
    dst_dir: str | Path,
    texts_path: str | Path | None = None,
    out: str | Path = "migration.json",
    level: int = 2,
    X: np.ndarray | None = None,
) -> dict:
    if X is None:
        from sphota.encoders import default_encoder

        if texts_path is None:
            raise ValueError("pass --texts <file with one text per line> (or vectors via the API)")
        texts = [
            t.strip()
            for t in Path(texts_path).read_text(encoding="utf-8").splitlines()
            if t.strip()
        ]
        X = default_encoder().encode(texts)
    res = table_from_vectors(src_dir, dst_dir, X, level=level)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def migrate_code(code: str, table: dict) -> tuple[str, float]:
    """Map one code: use the deepest prefix present in the table."""
    from sphota.codes import parse

    c = parse(code)
    for lv in sorted((int(k) for k in table["levels"]), reverse=True):
        if lv > c.depth:
            continue
        hit = table["levels"][str(lv)].get(str(c.prefix(lv)))
        if hit:
            return hit[0][0], float(hit[0][1])
    return str(c.prefix(1)), 0.0


def apply_table(codes_file: str | Path, table_path: str | Path, out: str | Path) -> int:
    table = json.loads(Path(table_path).read_text(encoding="utf-8"))
    n = 0
    with (
        open(codes_file, newline="", encoding="utf-8") as f,
        open(out, "w", newline="", encoding="utf-8") as g,
    ):
        r = csv.DictReader(f)
        fields = [*list(r.fieldnames or []), "code_" + table["to"], "migration_p"]
        w = csv.DictWriter(g, fieldnames=fields)
        w.writeheader()
        for row in r:
            new, p = migrate_code(row["code"], table)
            row["code_" + table["to"]] = new
            row["migration_p"] = f"{p:.4f}"
            w.writerow(row)
            n += 1
    return n
