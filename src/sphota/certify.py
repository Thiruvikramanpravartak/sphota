# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Adapter training and conformance certification (spec Part 8).

``train_adapter``: encode the same texts with the anchor and with the candidate model, then fit
a ridge regression from the candidate's (normalized) vectors to the anchor's canonical
vectors. Cheap (seconds on CPU once texts are encoded) and surprisingly strong.

``certify``: on the official test set (FLORES-200 devtest, 12 languages) measure

* ``agreement_same_text[l]``: P(model code prefix = anchor code prefix) at each level,
  separately for English and for all languages,
* ``agreement_cross_lingual[l]``: P(model code of English = model code of translation),
* ``fidelity``: mean cosine between adapted and anchor canonical vectors,

and grade against :data:`THRESHOLDS`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from sphota.adapters import Adapter, get_adapter, load_adapter
from sphota.release import load_release
from sphota.space import project, quantize

# Conformance thresholds for space s1 (spec Part 8, section 3). Agreement is measured
# against the reference encoder on the same text.
THRESHOLDS: dict[str, dict[str, float]] = {
    # neighbourhood-level interoperability on English text (D-013); chance is ~0.002
    "certified": {"l1_neighborhood_en": 0.55, "l1_top3_en": 0.10, "fidelity_en": 0.50},
    # additionally across 12 languages
    "certified_multilingual": {"l1_top3_all": 0.30, "fidelity_all": 0.45},
    # anchor variants (other runtimes / precisions of the anchor itself)
    "variant": {"all_L1": 0.95, "all_L4": 0.80},
}

EVAL_LANGS = ["en", "hi", "ta", "es", "fr", "de", "zh", "ar", "ru", "ja", "pt", "bn"]


def _ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    A = np.linalg.solve(G, Xc.T @ Yc)
    return A.astype(np.float32), (my - mx @ A).astype(np.float32)


def fit_adapter(
    model: str,
    X_model: np.ndarray,
    Y_canon: np.ndarray,
    alpha: float = 1.0,
    prefix: str = "",
    meta: dict[str, Any] | None = None,
) -> Adapter:
    """Ridge map from L2-normalized model vectors to canonical vectors."""
    X = X_model / np.maximum(np.linalg.norm(X_model, axis=1, keepdims=True), 1e-12)
    # choose alpha relative to the data scale
    A, b = _ridge(X.astype(np.float64), Y_canon.astype(np.float64), alpha * len(X) / 1000)
    return Adapter(
        model=model,
        A=A,
        b=b,
        meta={
            "space": "s1",
            "kind": "linear",
            "prefix": prefix,
            "train": {"pairs": len(X), "method": "ridge", "alpha": alpha},
            "created": dt.date.today().isoformat(),
            **(meta or {}),
        },
    )


def _default_texts(n: int) -> list[str]:
    """Training texts: FLORES-200 dev (12 languages) + Wikipedia sample, via the HF Hub."""
    try:
        from sphota.build.data import flores, wikipedia
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "default training texts need: pip install 'sphota[build]' "
            "(or pass --texts your_texts.txt)"
        ) from e
    texts = []
    df = flores("dev", EVAL_LANGS)
    for lang in EVAL_LANGS:
        texts += df[lang].tolist()
    per = max(0, (n - len(texts)) // 4)
    for lang in ["en", "hi", "es", "zh"]:
        texts += [r["text"] for r in wikipedia(lang, per)]
    return texts[:n]


def train_adapter(
    model: str,
    out: str | Path | None = None,
    texts_path: str | Path | None = None,
    prefix: str = "",
    n: int = 20000,
    alpha: float = 1.0,
) -> Path:
    from sphota.encoders import TorchEncoder, default_encoder

    if texts_path:
        texts = [
            t.strip()
            for t in Path(texts_path).read_text(encoding="utf-8").splitlines()
            if t.strip()
        ][:n]
    else:
        texts = _default_texts(n)
    if len(texts) < 500:
        raise ValueError(f"need at least 500 training texts, got {len(texts)}")
    rel = load_release()
    Y = project(rel, default_encoder().encode(texts))
    X = TorchEncoder(model, prefix=prefix).encode(texts)
    ad = fit_adapter(model, X, Y, alpha=alpha, prefix=prefix)
    out = Path(out) if out else Path(f"{model.split('/')[-1]}.sphota-adapter.npz")
    return ad.save(out)


def load_eval(quick: bool = False) -> dict[str, list[str]]:
    from sphota.build.data import flores

    df = flores("devtest", EVAL_LANGS)
    if quick:
        df = df.iloc[:300]
    return {lang: df[lang].tolist() for lang in EVAL_LANGS}


def evaluate(Y_model: dict[str, np.ndarray], Y_anchor: dict[str, np.ndarray]) -> dict[str, Any]:
    """Certificate metrics from canonical vectors per language (model-adapted and anchor)."""
    rel = load_release()
    qfull = {k: quantize(rel, v, levels=1) for k, v in Y_model.items()}
    qm = {k: quantize(rel, v).indices for k, v in Y_model.items()}
    qa = {k: quantize(rel, v).indices for k, v in Y_anchor.items()}

    def agree(a: np.ndarray, b: np.ndarray) -> list[float]:
        eq = a == b
        return [float(np.all(eq[:, : i + 1], 1).mean()) for i in range(a.shape[1])]

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(
            np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
        )

    langs = list(Y_model)
    same_all = np.mean([agree(qm[k], qa[k]) for k in langs], 0)
    xl = np.mean([agree(qm["en"], qm[k]) for k in langs if k != "en"], 0)
    xl_anchor = np.mean([agree(qa["en"], qa[k]) for k in langs if k != "en"], 0)
    nb = rel.l1_neighbors

    def top3(lang: str) -> float:
        return float((qfull[lang].cand[0][:, :3] == qa[lang][:, :1]).any(1).mean())

    def neighborhood(lang: str) -> float:
        a, m = qa[lang][:, 0], qm[lang][:, 0]
        near = (m == a) | ((nb[a][:, :8] == m[:, None]).any(1) if nb is not None else False)
        return float(np.mean(near))

    return {
        "l1_top3_en": round(top3("en"), 4),
        "l1_top3_all": round(float(np.mean([top3(k) for k in Y_model])), 4),
        "l1_neighborhood_en": round(neighborhood("en"), 4),
        "agreement_same_text_en": [round(x, 4) for x in agree(qm["en"], qa["en"])],
        "agreement_same_text": [round(float(x), 4) for x in same_all],
        "agreement_cross_lingual": [round(float(x), 4) for x in xl],
        "anchor_cross_lingual": [round(float(x), 4) for x in xl_anchor],
        "fidelity_en": round(cos(Y_model["en"], Y_anchor["en"]), 4),
        "fidelity_all": round(float(np.mean([cos(Y_model[k], Y_anchor[k]) for k in langs])), 4),
        "n_per_lang": len(Y_model["en"]),
        "languages": langs,
    }


def grade(m: dict[str, Any]) -> dict[str, Any]:
    t1, t2 = THRESHOLDS["certified"], THRESHOLDS["certified_multilingual"]
    checks = {k: bool(m[k] >= v) for k, v in t1.items()}
    multi = {k: bool(m[k] >= v) for k, v in t2.items()}
    ok = all(checks.values())
    is_multi = ok and all(multi.values())
    return {
        "checks": {**checks, **{f"multilingual_{k}": v for k, v in multi.items()}},
        "pass": ok,
        "multilingual": is_multi,
        "grade": "certified-multilingual" if is_multi else "certified" if ok else "not-certified",
    }


def certify(
    model: str, adapter: str | Path | Adapter | None = None, prefix: str = "", quick: bool = False
) -> dict[str, Any]:
    from sphota.encoders import TorchEncoder, default_encoder

    if isinstance(adapter, Adapter):
        ad = adapter
    elif adapter is not None:
        ad = load_adapter(adapter)
    else:
        try:
            ad = get_adapter(model)
        except Exception:
            ad = load_adapter(train_adapter(model, prefix=prefix))
    data = load_eval(quick)
    rel = load_release()
    enc = TorchEncoder(model, prefix=ad.prefix or prefix)
    anc = default_encoder()
    Ym = {k: ad(enc.encode(v)) for k, v in data.items()}
    Ya = {k: project(rel, anc.encode(v)) for k, v in data.items()}
    m = evaluate(Ym, Ya)
    cert = {
        "model": model,
        "space": rel.version,
        "date": dt.date.today().isoformat(),
        "test_set": "FLORES-200 devtest" + (" (quick, 300/lang)" if quick else ""),
        "thresholds": THRESHOLDS,
        **m,
        **grade(m),
    }
    cert["badge"] = badge(cert)
    return cert


def badge(cert: dict[str, Any]) -> str:
    color = {"certified-multilingual": "brightgreen", "certified": "green"}.get(
        cert["grade"], "red"
    )
    nbh = cert["l1_neighborhood_en"]
    text = f"{cert['grade']}%20nbhd%20{nbh:.2f}".replace("-", "--")
    return f"![Sphota {cert['space']}](https://img.shields.io/badge/sphota%20{cert['space']}-{text}-{color})"


def render_certificate(cert: dict[str, Any]) -> str:
    def row(name: str, vals: list[float]) -> str:
        return f"  {name:<28}" + "  ".join(f"L{i + 1} {v:.2f}" for i, v in enumerate(vals))

    mark = {True: "PASS", False: "FAIL"}
    lines = [
        "┌─ Sphota conformance certificate ─────────────────────────────",
        f"  model      {cert['model']}",
        f"  space      {cert['space']}     date {cert['date']}",
        f"  test set   {cert['test_set']}, {len(cert['languages'])} languages, n={cert['n_per_lang']}/lang",
        row("agreement w/ anchor (en)", cert["agreement_same_text_en"]),
        row("agreement w/ anchor (all)", cert["agreement_same_text"]),
        row("cross-lingual (model)", cert["agreement_cross_lingual"]),
        row("cross-lingual (anchor ref)", cert["anchor_cross_lingual"]),
        f"  L1 in top-3 candidates     en {cert['l1_top3_en']:.2f}   all {cert['l1_top3_all']:.2f}",
        f"  L1 in anchor neighbourhood en {cert['l1_neighborhood_en']:.2f}",
        f"  fidelity (cosine)          en {cert['fidelity_en']:.3f}   all {cert['fidelity_all']:.3f}",
        "  checks     " + ", ".join(f"{k}:{mark[v]}" for k, v in cert["checks"].items()),
        f"  RESULT     {cert['grade'].upper()}",
        f"  badge      {cert['badge']}",
        "└──────────────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)
