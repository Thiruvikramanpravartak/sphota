# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Multi-domain evaluation on public, license-clean data (D-031, M14).

    python -m sphota.build.domains --scale small

Does Sphota group real text by meaning outside Wikipedia/FLORES? Six domains:

=====================  ==========================================  =============  ============
domain                 dataset (Hugging Face)                      license        languages
=====================  ==========================================  =============  ============
conversational         mteb/amazon_massive_intent (MASSIVE)        CC-BY-4.0      14 (parallel)
customer support       PolyAI/banking77                            CC-BY-4.0      en
technical Q&A          mteb/stackexchange-clustering               CC-BY-SA       en
science                mteb/arxiv-clustering-s2s                   CC0-1.0        en
product reviews        mteb/IndicReviewsClusteringP2P              CC0-1.0        13 Indic (parallel)
code                   mteb/CodeSearchNetRetrieval                 MIT            6 prog. langs
=====================  ==========================================  =============  ============

Grouping metrics (labelled data): V-measure (the MTEB clustering metric) and purity of Sphota's
fixed groupings (themes = first code block, 4096 cells; regions = 32), next to k-means fitted on
each dataset's own anchor embeddings at the same granularity (an upper reference: it sees the
data, Sphota's groups are frozen and shared) and TF-IDF + k-means (classical). Cross-lingual
metrics (parallel data): share of translations in the same theme / region as the pivot language.
Cross-model (MASSIVE): multilingual MiniLM through the v1 and v1.1 adapters vs the anchor. Code:
docstring and code in the same theme / region, and docstring→code retrieval recall@1/@10 among
the sampled pairs with raw anchor vectors vs Sphota canonical vectors.

Writes ``bench/results/domains-<scale>.json``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time

import numpy as np

from sphota.build import embed
from sphota.build.common import ROOT, log

ANCHOR = "intfloat/multilingual-e5-small"
MASSIVE_LANGS = [
    "en",
    "hi",
    "ta",
    "te",
    "kn",
    "ml",
    "bn",
    "ur",
    "es",
    "fr",
    "de",
    "zh-CN",
    "ar",
    "ru",
]
INDIC_LANGS = ["hi", "bn", "mr", "te", "ta", "ur", "gu", "kn", "or", "ml", "pa", "as", "bd"]
CODE_LANGS = ["python", "java", "javascript", "go", "php", "ruby"]
SIZES = {
    "tiny": {"massive": 300, "flat": 600, "indic": 200, "code": 150},
    "small": {"massive": 2974, "flat": 5000, "indic": 1000, "code": 1000},
}
SIZES["medium"] = SIZES["full"] = SIZES["small"]


def _hf(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def _sample(n_total: int, n: int, seed: int = 0) -> np.ndarray:
    if n >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


# ------------------------------------------------------------------ loaders


def load_massive(n: int) -> dict[str, tuple[list[str], list[str]]]:
    out = {}
    for lg in MASSIVE_LANGS:
        with gzip.open(_hf("mteb/amazon_massive_intent", f"test/{lg}.json.gz"), "rt") as f:
            rows = sorted((json.loads(line) for line in f), key=lambda r: int(r["id"]))
        idx = _sample(len(rows), n)
        out[lg] = ([rows[i]["text"] for i in idx], [rows[i]["label"] for i in idx])
    return out


def load_banking(n: int) -> tuple[list[str], list[str]]:
    import pandas as pd

    df = pd.read_parquet(_hf("PolyAI/banking77", "data/test-00000-of-00001.parquet"))
    idx = _sample(len(df), n)
    return df.text.iloc[idx].tolist(), [str(x) for x in df.label.iloc[idx]]


def _first_split_jsonl(repo: str, fname: str, n: int) -> tuple[list[str], list[str]]:
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(_hf(repo, fname), "rt", encoding="utf-8") as f:
        j = json.loads(f.readline())
    idx = _sample(len(j["sentences"]), n)
    return [j["sentences"][i] for i in idx], [str(j["labels"][i]) for i in idx]


def load_indic(n: int) -> dict[str, tuple[list[str], list[str]]]:
    import pandas as pd

    out = {}
    for lg in INDIC_LANGS:
        df = pd.read_parquet(
            _hf("mteb/IndicReviewsClusteringP2P", f"{lg}/test-00000-of-00001.parquet")
        )
        S = [s for i in range(len(df)) for s in df.sentences[i]]
        L = [str(x) for i in range(len(df)) for x in df.labels[i]]
        idx = _sample(len(S), n)
        out[lg] = ([S[i] for i in idx], [L[i] for i in idx])
    return out


def load_code(n: int) -> dict[str, tuple[list[str], list[str]]]:
    """(docstrings, code) pairs per programming language."""
    import pandas as pd

    out = {}
    for lg in CODE_LANGS:
        q = pd.read_parquet(
            _hf("mteb/CodeSearchNetRetrieval", f"{lg}-queries/test-00000-of-00001.parquet")
        )
        c = pd.read_parquet(
            _hf("mteb/CodeSearchNetRetrieval", f"{lg}-corpus/test-00000-of-00001.parquet")
        )
        r = pd.read_parquet(
            _hf("mteb/CodeSearchNetRetrieval", f"{lg}-qrels/test-00000-of-00001.parquet")
        )
        qtext = dict(zip(q["id"].astype(str), q["text"]))
        ctext = dict(zip(c["id"].astype(str), c["text"]))
        pairs = [
            (qtext[a], ctext[b])
            for a, b in zip(r["query-id"].astype(str), r["corpus-id"].astype(str))
            if a in qtext and b in ctext
        ]
        idx = _sample(len(pairs), n)
        out[lg] = ([pairs[i][0] for i in idx], [pairs[i][1][:4000] for i in idx])
    return out


# ------------------------------------------------------------------ metrics


def nearest(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    cn = (C**2).sum(1)
    return np.concatenate(
        [np.argmin(cn[None] - 2 * Y[s : s + 4096] @ C.T, 1) for s in range(0, len(Y), 4096)]
    )


def purity(pred: np.ndarray, labels: list[str]) -> float:
    from collections import Counter

    by: dict[int, Counter] = {}
    for p, lab in zip(pred, labels):
        by.setdefault(int(p), Counter())[lab] += 1
    return sum(c.most_common(1)[0][1] for c in by.values()) / len(labels)


def grouping(
    texts: list[str], labels: list[str], X: np.ndarray, Y: np.ndarray, grids: dict
) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import v_measure_score

    def m(pred):
        return {
            "v_measure": round(float(v_measure_score(labels, pred)), 4),
            "purity": round(purity(pred, labels), 4),
            "groups": len(set(pred.tolist())),
        }

    k_true = len(set(labels))
    th, rg = nearest(Y, grids["themes"]), nearest(Y, grids["regions"])
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    out = {"n": len(texts), "labels": k_true, "sphota_themes": m(th), "sphota_regions": m(rg)}
    for name, k in [
        ("kmeans_k_true", k_true),
        ("kmeans_same_as_themes", len(set(th.tolist()))),
        ("kmeans_32", 32),
    ]:
        k = max(2, min(k, len(texts) - 1))
        out[name] = m(KMeans(k, n_init=3, random_state=0).fit_predict(Xn))
    T = TfidfVectorizer(max_features=20000, sublinear_tf=True).fit_transform(texts)
    out["tfidf_kmeans_k_true"] = m(KMeans(max(2, k_true), n_init=3, random_state=0).fit_predict(T))
    return out


# ------------------------------------------------------------------ run


def run(scale: str) -> dict:
    from sphota.adapters import get_adapter
    from sphota.regions import _load as regions
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    grids = {"themes": rel.codebooks[0].astype(np.float64), "regions": regions()[0]}
    sz = SIZES[scale]

    def enc(name: str, texts: list[str], model: str = ANCHOR) -> np.ndarray:
        return embed.cached(model, f"domain-{name}", texts, scale)

    res: dict = {"scale": scale, "space": rel.version, "anchor": ANCHOR, "domains": {}}

    # conversational (parallel, 14 languages) + cross-lingual + cross-model
    t0 = time.time()
    mas = load_massive(sz["massive"])
    Xm = {lg: enc(f"massive-{lg}-{len(t)}", t) for lg, (t, _) in mas.items()}
    Ym = {lg: project(rel, x) for lg, x in Xm.items()}
    th = {lg: nearest(y, grids["themes"]) for lg, y in Ym.items()}
    rg = {lg: nearest(y, grids["regions"]) for lg, y in Ym.items()}
    d: dict = {
        "dataset": "mteb/amazon_massive_intent",
        "license": "CC-BY-4.0",
        "grouping": {lg: grouping(*mas[lg], Xm[lg], Ym[lg], grids) for lg in ["en", "hi", "ta"]},
        "cross_lingual_vs_en": {
            lg: {
                "same_theme": round(float((th[lg] == th["en"]).mean()), 4),
                "same_region": round(float((rg[lg] == rg["en"]).mean()), 4),
            }
            for lg in MASSIVE_LANGS
            if lg != "en"
        },
    }
    pm = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    xmodel = {}
    for lg in ["en", "hi"]:
        Xo = enc(f"massive-{lg}-{len(mas[lg][0])}", mas[lg][0], model=pm)
        for s in ["v1", "v1.1"]:
            Yo = get_adapter(pm, s)(Xo)
            xmodel[f"{lg}_{s}"] = {
                "same_theme": round(float((nearest(Yo, grids["themes"]) == th[lg]).mean()), 4),
                "same_region": round(float((nearest(Yo, grids["regions"]) == rg[lg]).mean()), 4),
            }
    d["cross_model_pmMiniLM"] = xmodel
    d["seconds"] = round(time.time() - t0)
    res["domains"]["conversational"] = d
    log(f"conversational done ({d['seconds']}s)")

    # flat English domains
    flat = {
        "customer_support": ("PolyAI/banking77", "CC-BY-4.0", lambda n: load_banking(n)),
        "technical_qa": (
            "mteb/stackexchange-clustering",
            "CC-BY-SA (Stack Exchange)",
            lambda n: _first_split_jsonl("mteb/stackexchange-clustering", "test.jsonl", n),
        ),
        "science": (
            "mteb/arxiv-clustering-s2s",
            "CC0-1.0",
            lambda n: _first_split_jsonl("mteb/arxiv-clustering-s2s", "test.jsonl.gz", n),
        ),
    }
    for name, (repo, lic, loader) in flat.items():
        t0 = time.time()
        texts, labels = loader(sz["flat"])
        X = enc(f"{name}-{len(texts)}", texts)
        res["domains"][name] = {
            "dataset": repo,
            "license": lic,
            "grouping": {"en": grouping(texts, labels, X, project(rel, X), grids)},
            "seconds": round(time.time() - t0),
        }
        log(f"{name} done")

    # Indian-language reviews (parallel, 13 languages)
    t0 = time.time()
    ind = load_indic(sz["indic"])
    Xi = {lg: enc(f"indic-{lg}-{len(t)}", t) for lg, (t, _) in ind.items()}
    Yi = {lg: project(rel, x) for lg, x in Xi.items()}
    thi = {lg: nearest(y, grids["themes"]) for lg, y in Yi.items()}
    rgi = {lg: nearest(y, grids["regions"]) for lg, y in Yi.items()}
    res["domains"]["reviews_indic"] = {
        "dataset": "mteb/IndicReviewsClusteringP2P",
        "license": "CC0-1.0",
        "grouping": {lg: grouping(*ind[lg], Xi[lg], Yi[lg], grids) for lg in ["hi", "ta", "bn"]},
        "cross_lingual_vs_hi": {
            lg: {
                "same_theme": round(float((thi[lg] == thi["hi"]).mean()), 4),
                "same_region": round(float((rgi[lg] == rgi["hi"]).mean()), 4),
            }
            for lg in INDIC_LANGS
            if lg != "hi"
        },
        "seconds": round(time.time() - t0),
    }
    log("reviews done")

    # code: docstring vs code
    t0 = time.time()
    code = load_code(sz["code"])
    cres = {}
    for lg, (docs, srcs) in code.items():
        Xd, Xc = enc(f"code-{lg}-doc-{len(docs)}", docs), enc(f"code-{lg}-src-{len(srcs)}", srcs)
        Yd, Yc = project(rel, Xd), project(rel, Xc)

        def recall(Q, D):
            Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
            D = D / np.linalg.norm(D, axis=1, keepdims=True)
            rank = (-(Q @ D.T)).argsort(1)
            gold = np.arange(len(Q))[:, None]
            return {
                "r@1": round(float((rank[:, :1] == gold).any(1).mean()), 4),
                "r@10": round(float((rank[:, :10] == gold).any(1).mean()), 4),
            }

        cres[lg] = {
            "n": len(docs),
            "same_theme": round(
                float((nearest(Yd, grids["themes"]) == nearest(Yc, grids["themes"])).mean()), 4
            ),
            "same_region": round(
                float((nearest(Yd, grids["regions"]) == nearest(Yc, grids["regions"])).mean()), 4
            ),
            "retrieval_raw_anchor": recall(Xd, Xc),
            "retrieval_canonical": recall(Yd, Yc),
        }
    res["domains"]["code"] = {
        "dataset": "mteb/CodeSearchNetRetrieval",
        "license": "MIT",
        "per_language": cres,
        "seconds": round(time.time() - t0),
    }
    log("code done")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=list(SIZES), default="small")
    a = ap.parse_args()
    out = run(a.scale)
    p = ROOT / "bench" / "results" / f"domains-{a.scale}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {p}")
