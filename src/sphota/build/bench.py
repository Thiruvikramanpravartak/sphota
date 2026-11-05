# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Official Sphota benchmark (milestone M5).

    python -m sphota.build.bench --scale small

Test data (never used to fit the space, codebooks, calibration or adapters):
* FLORES-200 devtest, 1012 meanings x 12 languages (cross-lingual agreement, retrieval)
* SIB-200 topic labels for the English devtest sentences (topic retrieval)
* STS-B pairs scored >= 4.5/5 (paraphrase stability, English)

Writes ``bench/results/benchmark-<scale>.json``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from sphota.build import embed
from sphota.build.common import DATA_DIR, ROOT, log
from sphota.build.release import embeddings, load

ANCHOR = "intfloat/multilingual-e5-small"


def agree(a: np.ndarray, b: np.ndarray) -> list[float]:
    eq = a == b
    return [round(float(np.all(eq[:, : i + 1], 1).mean()), 4) for i in range(a.shape[1])]


def consistent_truncated(ca, cb, depth_a, depth_b) -> tuple[float, float]:
    """For truncated codes: fraction where the shorter is a prefix of the longer, and mean depth."""
    ok = []
    for x, y, da, db in zip(ca, cb, depth_a, depth_b):
        d = min(da, db)
        ok.append(bool(np.all(x[:d] == y[:d])))
    return float(np.mean(ok)), float(np.mean(np.concatenate([depth_a, depth_b])))


def truncation_curve(qa, qb, taus=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    out = []
    for tau in taus:
        da = np.maximum(1, (qa.confidence >= tau).sum(1))
        db = np.maximum(1, (qb.confidence >= tau).sum(1))
        cons, depth = consistent_truncated(qa.indices, qb.indices, da, db)
        out.append(
            {"min_confidence": tau, "consistent": round(cons, 4), "mean_depth": round(depth, 3)}
        )
    return out


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def code_rank_scores(rel, qi: np.ndarray, di: np.ndarray, qcand=None) -> np.ndarray:
    """Similarity matrix between query codes and doc codes using shared-prefix depth plus
    hierarchy-aware geometry at L1 (vectorized approximation of sphota.similarity)."""
    from sphota.distance import LEVEL_WEIGHTS

    w = np.array(LEVEL_WEIGHTS) / sum(LEVEL_WEIGHTS)
    C1 = rel.codebooks[0]
    C1n = C1 / np.linalg.norm(C1, axis=1, keepdims=True)
    s = w[0] * np.maximum(0, C1n[qi[:, 0]] @ C1n[di[:, 0]].T)
    same = np.ones((len(qi), len(di)), bool)
    for lvl in range(qi.shape[1]):
        same &= qi[:, lvl][:, None] == di[:, lvl][None, :]
        if lvl == 0:
            s = np.where(same, w[0], s)
        else:
            s = s + w[lvl] * same
    return s


def retrieval_metrics(
    S: np.ndarray, rel_mask: np.ndarray, k: int = 10, tie_noise: np.ndarray | None = None
) -> dict:
    if tie_noise is not None:  # random tie-breaking for discrete scores
        S = S + tie_noise
    order = np.argsort(-S, 1)[:, :k]
    hit = np.take_along_axis(rel_mask, order, 1)
    return {
        "top1": round(float(hit[:, 0].mean()), 4),
        f"recall@{k}": round(float(hit.any(1).mean()), 4),
        f"p@{k}": round(float(hit.mean()), 4),
    }


def run(scale: str) -> dict:
    from sphota.release import load_release
    from sphota.space import project, quantize

    rel = load_release()
    langs, _dev, test, wiki, _tat = load(scale)
    E = embeddings(scale)
    d = DATA_DIR / scale
    res: dict = {"scale": scale, "space": rel.version, "languages": langs, "n_meanings": len(test)}
    Y = {lg: project(rel, E[("test", lg)]) for lg in langs}
    Q = {lg: quantize(rel, Y[lg]) for lg in langs}
    QS = {lg: quantize(rel, Y[lg], stable=True) for lg in langs}

    # ---- cross-lingual agreement (en <-> x), default and stable
    per_lang = {lg: agree(Q["en"].indices, Q[lg].indices) for lg in langs[1:]}
    res["cross_lingual"] = {
        "mean": [round(float(x), 4) for x in np.mean(list(per_lang.values()), 0)],
        "per_language": per_lang,
        "stable_mode_mean": [
            round(float(x), 4)
            for x in np.mean([agree(QS["en"].indices, QS[lg].indices) for lg in langs[1:]], 0)
        ],
        "all_pairs_mean": [
            round(float(x), 4)
            for x in np.mean(
                [
                    agree(Q[a].indices, Q[b].indices)
                    for i, a in enumerate(langs)
                    for b in langs[i + 1 :]
                ],
                0,
            )
        ],
    }
    # PCA-only baseline for the alignment effect (same codebook sizes, refit) is in design-sweep
    log(
        f"cross-lingual: {res['cross_lingual']['mean']} stable {res['cross_lingual']['stable_mode_mean']}"
    )

    # ---- truncation curve and calibration on unseen data
    curves = [truncation_curve(Q["en"], Q[lg]) for lg in langs[1:]]
    res["truncation_cross_lingual"] = [
        {
            "min_confidence": c["min_confidence"],
            "consistent": round(float(np.mean([cv[i]["consistent"] for cv in curves])), 4),
            "mean_depth": round(float(np.mean([cv[i]["mean_depth"] for cv in curves])), 3),
        }
        for i, c in enumerate(curves[0])
    ]
    rel_tab = []
    for lvl in range(4):
        conf = np.concatenate(
            [Q["en"].confidence[:, lvl] for _ in langs[1:]]
            + [Q[lg].confidence[:, lvl] for lg in langs[1:]]
        )
        ag = np.concatenate(
            [
                np.all(Q["en"].indices[:, : lvl + 1] == Q[lg].indices[:, : lvl + 1], 1)
                for lg in langs[1:]
            ]
            * 2
        )
        bins = np.linspace(0, 1, 6)
        rows, ece = [], 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf >= lo) & ((conf < hi) | (hi == 1))
            if m.sum():
                rows.append(
                    {
                        "bin": f"{lo:.1f}-{hi:.1f}",
                        "n": int(m.sum()),
                        "conf": round(float(conf[m].mean()), 3),
                        "agree": round(float(ag[m].mean()), 3),
                    }
                )
                ece += m.mean() * abs(conf[m].mean() - ag[m].mean())
        rel_tab.append({"level": lvl + 1, "ece": round(float(ece), 4), "bins": rows})
    res["calibration_unseen"] = rel_tab
    # selective accuracy: only codes whose level-l confidence >= tau (both sides)
    sel = []
    for lvl in (0, 1):
        for tau in (0.0, 0.5, 0.7, 0.8, 0.9):
            cov, acc = [], []
            for lg in langs[1:]:
                m = (Q["en"].confidence[:, lvl] >= tau) & (Q[lg].confidence[:, lvl] >= tau)
                ag = np.all(Q["en"].indices[:, : lvl + 1] == Q[lg].indices[:, : lvl + 1], 1)
                cov.append(m.mean())
                acc.append(ag[m].mean() if m.any() else np.nan)
            sel.append({"level": lvl + 1, "min_confidence": tau, "coverage": round(float(np.mean(cov)), 4),
                        "agreement": round(float(np.nanmean(acc)), 4)})
    res["selective_cross_lingual"] = sel
    log(f"selective: {sel}")
    log("truncation: " + str(res["truncation_cross_lingual"]))

    # ---- paraphrase stability (STS-B >= 4.5, English)
    para = pd.read_parquet(d / "stsb_paraphrases.parquet")
    Pa = project(rel, embed.cached(ANCHOR, "stsb_a", para["a"].tolist(), scale))
    Pb = project(rel, embed.cached(ANCHOR, "stsb_b", para["b"].tolist(), scale))
    qa, qb = quantize(rel, Pa), quantize(rel, Pb)
    qas, qbs = quantize(rel, Pa, stable=True), quantize(rel, Pb, stable=True)
    res["paraphrase"] = {
        "n_pairs": len(para),
        "agree": agree(qa.indices, qb.indices),
        "stable_mode": agree(qas.indices, qbs.indices),
        "truncation": truncation_curve(qa, qb),
    }
    log(f"paraphrase: {res['paraphrase']['agree']} stable {res['paraphrase']['stable_mode']}")

    # ---- reindex-free migration: documents indexed as codes by model A, queried by model B
    from sphota.adapters import get_adapter, registry

    S_ref = code_rank_scores(rel, qa.indices, qb.indices)
    eyep = np.eye(len(Pa), dtype=bool)
    tie = np.random.default_rng(1).random(S_ref.shape) * 1e-6
    mig = {
        "anchor→anchor (codes)": retrieval_metrics(S_ref, eyep, tie_noise=tie),
        "anchor→anchor (raw vectors)": retrieval_metrics(l2n(Pa) @ l2n(Pb).T, eyep),
    }
    for model in registry()["adapters"]:
        ad = get_adapter(model)
        Xb_raw = embed.cached(model, "stsb_b", para["b"].tolist(), scale)
        Xa_raw = embed.cached(model, "stsb_a", para["a"].tolist(), scale)
        Bm = ad(Xb_raw)
        short = registry()["adapters"][model]["short"]
        di = quantize(rel, Bm).indices
        S = code_rank_scores(rel, qa.indices, di)
        alt = qa.cand[0][:, :3]
        S2 = S.copy()
        for j in range(1, 3):
            qi = qa.indices.copy()
            qi[:, 0] = alt[:, j]
            S2 = np.maximum(S2, 0.9 * code_rank_scores(rel, qi, di))
        mig[f"index {short} → query anchor (canonical vectors)"] = retrieval_metrics(
            l2n(Pa) @ l2n(Bm).T, eyep
        )
        mig[f"index {short} → query anchor (codes)"] = retrieval_metrics(S, eyep, tie_noise=tie)
        mig[f"index {short} → query anchor (codes + alternates)"] = retrieval_metrics(
            S2, eyep, tie_noise=tie
        )
        mig[f"{short} → {short} (raw vectors, re-index baseline)"] = retrieval_metrics(
            l2n(Xa_raw) @ l2n(Xb_raw).T, eyep
        )
    res["migration_paraphrase_retrieval"] = {"n_docs": len(Pb), **mig}
    log(f"migration: {mig}")

    # ---- utilization (held-out: FLORES devtest all languages + STS sentences; and training corpus)
    held = np.concatenate([Q[lg].indices for lg in langs] + [qa.indices, qb.indices])
    corpus = quantize(rel, project(rel, E["wiki"])).indices

    def util(codes):
        c = np.bincount(codes[:, 0], minlength=rel.sizes[0])
        p = c[c > 0] / c.sum()
        out = {
            "l1_used": int((c > 0).sum()),
            "l1_used_frac": round(float((c > 0).mean()), 4),
            "l1_perplexity": round(float(np.exp(-(p * np.log(p)).sum())), 1),
            "max_cell_share": round(float(c.max() / c.sum()), 5),
        }
        for lvl in range(1, 4):
            cl = np.bincount(codes[:, lvl], minlength=rel.sizes[lvl])
            out[f"l{lvl + 1}_used_frac"] = round(float((cl > 0).mean()), 4)
        return out

    res["utilization"] = {
        "training_corpus": {"n": len(corpus), **util(corpus)},
        "held_out": {"n": len(held), **util(held)},
    }
    log(f"utilization: {res['utilization']}")

    # ---- retrieval: (a) cross-lingual translation retrieval en -> x ; (b) topic retrieval
    rng = np.random.default_rng(0)
    n = len(test)
    eye = np.eye(n, dtype=bool)
    tr = {"raw_anchor": [], "canonical": [], "codes": [], "codes_multiprobe": []}
    for lg in langs[1:]:
        tr["raw_anchor"].append(
            retrieval_metrics(l2n(E[("test", "en")]) @ l2n(E[("test", lg)]).T, eye)
        )
        tr["canonical"].append(retrieval_metrics(l2n(Y["en"]) @ l2n(Y[lg]).T, eye))
        S = code_rank_scores(rel, Q["en"].indices, Q[lg].indices)
        noise = rng.random(S.shape) * 1e-6
        tr["codes"].append(retrieval_metrics(S, eye, tie_noise=noise))
        # multi-probe: query also matches docs under its L1 alternates (top-3 L1 candidates)
        alt = Q["en"].cand[0][:, :3]
        S2 = S.copy()
        for j in range(1, 3):
            qi = Q["en"].indices.copy()
            qi[:, 0] = alt[:, j]
            S2 = np.maximum(S2, 0.9 * code_rank_scores(rel, qi, Q[lg].indices))
        tr["codes_multiprobe"].append(retrieval_metrics(S2, eye, tie_noise=noise))
    res["retrieval_translation"] = {
        k: {m: round(float(np.mean([r[m] for r in v])), 4) for m in v[0]} for k, v in tr.items()
    }
    log(f"translation retrieval: {res['retrieval_translation']}")
    sib = pd.read_parquet(d / "sib200_test.parquet")
    sib = sib[sib["split"] == "test"].set_index("index_id")["category"]
    ids = np.array([i for i in range(n) if i in sib.index])
    cats = sib.loc[ids].to_numpy()
    same = cats[:, None] == cats[None, :]
    np.fill_diagonal(same, False)
    mask_self = ~np.eye(len(ids), dtype=bool)
    Xe = l2n(E[("test", "en")][ids])
    Ye = l2n(Y["en"][ids])
    qi = Q["en"].indices[ids]
    topic = {}
    for name, S in [
        ("raw_anchor", Xe @ Xe.T),
        ("canonical", Ye @ Ye.T),
        ("codes", code_rank_scores(rel, qi, qi) + rng.random((len(ids), len(ids))) * 1e-6),
    ]:
        S = np.where(mask_self, S, -np.inf)
        topic[name] = retrieval_metrics(S, same)
    # cross-lingual topic retrieval: English queries, Hindi documents
    qh = Q["hi"].indices[ids]
    Sx = code_rank_scores(rel, qi, qh) + rng.random((len(ids), len(ids))) * 1e-6
    topic["codes_en_to_hi"] = retrieval_metrics(np.where(mask_self, Sx, -np.inf), same)
    Xh = l2n(E[("test", "hi")][ids])
    topic["raw_anchor_en_to_hi"] = retrieval_metrics(np.where(mask_self, Xe @ Xh.T, -np.inf), same)
    res["retrieval_topic_sib200"] = {"n": len(ids), **topic}
    log(f"topic retrieval: {topic}")

    # ---- speed (this machine)
    from sphota.encoders import default_encoder

    sample = test["en"].tolist()[:256]
    enc = default_encoder("fp32")
    enc.encode(sample[:8])
    enc._cache.d.clear()
    t = time.time()
    Xs = enc.encode(sample)
    fp32 = len(sample) / (time.time() - t)
    enc8 = default_encoder("int8")
    enc8.encode(sample[:8])
    t = time.time()
    X8 = enc8.encode(sample)
    int8 = len(sample) / (time.time() - t)
    Yb = project(rel, np.repeat(Xs, 40, 0))
    t = time.time()
    quantize(rel, Yb)
    qspeed = len(Yb) / (time.time() - t)
    res["speed"] = {
        "encode_fp32_sent_per_s": round(fp32, 1),
        "encode_int8_sent_per_s": round(int8, 1),
        "quantize_vectors_per_s": round(qspeed),
        "cpu": "4 vCPU (CI container)",
    }
    res["int8_variant_agreement"] = agree(
        quantize(rel, project(rel, Xs)).indices, quantize(rel, project(rel, X8)).indices
    )
    log(f"speed {res['speed']} int8 agreement {res['int8_variant_agreement']}")

    # ---- cross-model agreement (from adapters results)
    p = ROOT / "bench" / "results" / f"adapters-{scale}.json"
    if p.exists():
        ad = json.loads(p.read_text())
        res["cross_model"] = {
            m: {
                k: v[k]
                for k in [
                    "agreement_same_text_en",
                    "agreement_same_text",
                    "agreement_cross_lingual",
                    "fidelity_en",
                    "grade",
                ]
            }
            for m, v in ad.items()
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    a = ap.parse_args()
    res = run(a.scale)
    out = ROOT / "bench" / "results" / f"benchmark-{a.scale}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
