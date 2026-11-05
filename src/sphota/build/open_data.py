# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Milestone M9: tag a multilingual Wikipedia sample with Sphota s1 codes.

    python -m sphota.build.open_data --scale small      # 68k sentences, 12 languages (cached embeddings)
    python -m sphota.build.open_data --scale full       # scalable: same pipeline, GPU recommended

Output: ``data/<scale>/open/sphota-wiki-s1-<scale>.parquet`` (+ ``stats.json``). Not committed
(CC-BY-SA text); publish with ``scripts/upload_dataset.py`` (not run automatically).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from sphota.build import embed
from sphota.build.common import DATA_DIR, log
from sphota.codes import from_indices
from sphota.release import load_release
from sphota.space import project, quantize


def build(scale: str) -> dict:
    rel = load_release()
    wiki = pd.read_parquet(DATA_DIR / scale / "wiki.parquet")
    X = embed.cached("intfloat/multilingual-e5-small", "wiki", wiki["text"].tolist(), scale)
    q = quantize(rel, project(rel, X))
    conf = q.confidence
    labels = rel.labels
    df = pd.DataFrame(
        {
            "text": wiki["text"].values,
            "lang": wiki["lang"].values,
            "wiki_title": wiki["title"].values,
            "sphota": [str(from_indices(r)) for r in q.indices],
            "sphota_l1": [str(from_indices(r[:1])) for r in q.indices],
            "sphota_l2": [str(from_indices(r[:2])) for r in q.indices],
            "conf_l1": conf[:, 0].round(4),
            "conf_l2": conf[:, 1].round(4),
            "conf_l3": conf[:, 2].round(4),
            "conf_l4": conf[:, 3].round(4),
            "label_en": [labels.get(str(r[0]), {}).get("en", "") for r in q.indices],
            "space": rel.version,
        }
    )
    out = DATA_DIR / scale / "open"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sphota-wiki-s1-{scale}.parquet"
    df.to_parquet(path, index=False)
    stats = {
        "rows": len(df),
        "languages": df["lang"].value_counts().to_dict(),
        "distinct_l1": int(df["sphota_l1"].nunique()),
        "distinct_l2": int(df["sphota_l2"].nunique()),
        "distinct_full": int(df["sphota"].nunique()),
        "median_conf_l1": float(np.median(conf[:, 0])),
        "bytes": path.stat().st_size,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"wrote {path}: {stats}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    build(ap.parse_args().scale)


if __name__ == "__main__":
    main()
