# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Milestone R bake-off: anchor candidates x quantizers, tiny scale, CPU.

    python -m sphota.build.bakeoff --scale tiny

Writes ``bench/results/bakeoff-<scale>.json``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from sphota.build import embed
from sphota.build.common import DATA_DIR, LANGS, ROOT, SCALES, log
from sphota.build.fit import (
    QUANTIZERS,
    fit_space,
    nmi,
    prefix_agree,
    retrieval_top1,
    utilization,
)

ANCHORS = [
    "BAAI/bge-m3",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def load_texts(scale: str):
    d = DATA_DIR / scale
    langs = LANGS[: SCALES[scale]["n_langs"]]
    dev = pd.read_parquet(d / "flores_dev.parquet")[langs]
    test = pd.read_parquet(d / "flores_devtest.parquet")[langs]
    wiki = pd.read_parquet(d / "wiki.parquet")
    tat = pd.read_parquet(d / "tatoeba.parquet")
    sib = pd.read_parquet(d / "sib200_test.parquet")
    return langs, dev, test, wiki, tat, sib


def embed_all(model: str, scale: str):
    langs, dev, test, wiki, tat, _ = load_texts(scale)
    E = {}
    for lang in langs:
        E[("dev", lang)] = embed.cached(model, f"flores_dev_{lang}", dev[lang].tolist(), scale)
        E[("test", lang)] = embed.cached(
            model, f"flores_devtest_{lang}", test[lang].tolist(), scale
        )
    E["wiki"] = embed.cached(model, "wiki", wiki["text"].tolist(), scale)
    E["tat_x"] = embed.cached(model, "tatoeba_x", tat["text"].tolist(), scale)
    E["tat_en"] = embed.cached(model, "tatoeba_en", tat["en"].tolist(), scale)
    return E


def run(scale: str, anchors: list[str], quantizers: list[str], D: int) -> dict:
    langs, dev, test, wiki, tat, sib = load_texts(scale)
    K = SCALES[scale]["k"]
    sib_test = sib[sib["split"] == "test"].set_index("index_id")["category"]
    results: dict = {"scale": scale, "langs": langs, "K": K, "D": D, "anchors": {}}
    for model in anchors:
        log(f"== {model}")
        t0 = time.time()
        E = embed_all(model, scale)
        sps = embed.speed(model, test["en"].tolist()[:256])
        groups = [np.stack([E[("dev", lang)][i] for lang in langs]) for i in range(len(dev))]
        groups += [np.stack([a, b]) for a, b in zip(E["tat_x"], E["tat_en"])]
        corpus = np.concatenate([E["wiki"]] + [E[("dev", lang)] for lang in langs])
        r: dict = {"sent_per_s_cpu": sps, "dim": int(E["wiki"].shape[1])}
        # raw cross-lingual retrieval (en -> x), anchor space
        r["raw_retrieval_top1"] = float(
            np.mean([retrieval_top1(E[("test", "en")], E[("test", lang)]) for lang in langs[1:]])
        )
        for align in ["pca", "aligned"]:
            space = fit_space(corpus, None if align == "pca" else groups, D=D)
            Y = {lang: space(E[("test", lang)]) for lang in langs}
            r[f"{align}_retrieval_top1"] = float(
                np.mean([retrieval_top1(Y["en"], Y[lang]) for lang in langs[1:]])
            )
            Xfit = space(corpus)
            for qn in quantizers:
                tq = time.time()
                q = QUANTIZERS[qn](K).fit(Xfit)
                fit_s = time.time() - tq
                codes = {lang: q.encode(Y[lang]) for lang in langs}
                agree = np.mean([prefix_agree(codes["en"], codes[lang]) for lang in langs[1:]], 0)
                util = utilization(q.encode(space(E["wiki"])), K[0])
                ids = [i for i in range(len(test)) if i in sib_test.index]
                topic_nmi = nmi(codes["en"][ids, 0], sib_test.loc[ids].to_numpy())
                r[f"{align}/{qn}"] = {
                    "xl_agree": [round(float(a), 4) for a in agree],
                    **util,
                    "topic_nmi_l1": topic_nmi,
                    "fit_seconds": fit_s,
                }
                log(f"{align}/{qn}: agree={np.round(agree, 3)} util={util} nmi={topic_nmi:.3f}")
        r["total_seconds"] = time.time() - t0
        results["anchors"][model] = r
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="tiny", choices=list(SCALES))
    ap.add_argument("--anchors", nargs="*", default=ANCHORS)
    ap.add_argument("--quantizers", nargs="*", default=["rkm", "rqema", "tree"])
    ap.add_argument("--dim", type=int, default=128)
    a = ap.parse_args()
    res = run(a.scale, a.anchors, a.quantizers, a.dim)
    out = ROOT / "bench" / "results" / f"bakeoff-{a.scale}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
