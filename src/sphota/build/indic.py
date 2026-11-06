# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""How well do Sphota codes hold up across Indian languages?

    python -m sphota.build.indic --scale small

FLORES-200 devtest (1012 sentences, translated by professionals) in English and up to 20 Indian
languages, all encoded with the frozen s1 space. For each language, against the English original:

* region / L1 agreement: the translation lands in the same region (L1 cell) as the English sentence;
* coverage: share of the language's sentences whose region (L1) confidence is at least 0.7;
* confident agreement: agreement among pairs where both sides are confident (D-017's "translation");
* retrieval@1: the English sentence is the nearest canonical vector to its translation;
* top-region share: share of the language's sentences in its single most common region
  (English is the baseline; a large excess means the language collapses into one region).

Writes ``bench/results/indic-<scale>.json`` and prints a table for docs/benchmark.md.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from sphota.build.common import ROOT, log

# FLORES code -> (ISO, English name). Order: by number of speakers (roughly).
INDIC = {
    "hin_Deva": ("hi", "Hindi"),
    "ben_Beng": ("bn", "Bengali"),
    "mar_Deva": ("mr", "Marathi"),
    "tel_Telu": ("te", "Telugu"),
    "tam_Taml": ("ta", "Tamil"),
    "urd_Arab": ("ur", "Urdu"),
    "guj_Gujr": ("gu", "Gujarati"),
    "kan_Knda": ("kn", "Kannada"),
    "ory_Orya": ("or", "Odia"),
    "mal_Mlym": ("ml", "Malayalam"),
    "pan_Guru": ("pa", "Punjabi"),
    "asm_Beng": ("as", "Assamese"),
    "mai_Deva": ("mai", "Maithili"),
    "npi_Deva": ("ne", "Nepali"),
    "snd_Arab": ("sd", "Sindhi"),
    "san_Deva": ("sa", "Sanskrit"),
    "kas_Arab": ("ks", "Kashmiri (Arabic)"),
    "mni_Beng": ("mni", "Manipuri (Bengali script)"),
    "sat_Olck": ("sat", "Santali (Ol Chiki)"),
}
N = {"tiny": 100, "small": 1012, "medium": 1012, "full": 1012}
CONF = 0.7


def _metrics(en: list, xx: list, ven: np.ndarray, vxx: np.ndarray) -> dict:
    reg_en = np.array([c.region for c in en])
    reg_xx = np.array([c.region for c in xx])
    rc_en = np.array([c.region_confidence for c in en])
    rc_xx = np.array([c.region_confidence for c in xx])
    l1_en = np.array([c.levels[0] for c in en])
    l1_xx = np.array([c.levels[0] for c in xx])
    lc_en = np.array([c.confidence[0] for c in en])
    lc_xx = np.array([c.confidence[0] for c in xx])
    both_r = (rc_en >= CONF) & (rc_xx >= CONF)
    both_l = (lc_en >= CONF) & (lc_xx >= CONF)
    sims = vxx @ ven.T
    r1 = float((sims.argmax(1) == np.arange(len(xx))).mean())
    top = Counter(reg_xx.tolist()).most_common(1)[0]
    return {
        "region_agree": float((reg_en == reg_xx).mean()),
        "region_coverage": float((rc_xx >= CONF).mean()),
        "region_agree_confident": float((reg_en == reg_xx)[both_r].mean())
        if both_r.any()
        else None,
        "region_both_confident": float(both_r.mean()),
        "l1_agree": float((l1_en == l1_xx).mean()),
        "l1_coverage": float((lc_xx >= CONF).mean()),
        "l1_agree_confident": float((l1_en == l1_xx)[both_l].mean()) if both_l.any() else None,
        "retrieval_at_1": r1,
        "top_region": top[0],
        "top_region_share": top[1] / len(xx),
    }


def run(scale: str) -> dict:
    import pandas as pd

    from sphota import encode_batch
    from sphota.build.data import _hf

    df = pd.read_parquet(_hf("mteb/flores", "devtest.parquet")).head(N[scale])
    texts_en = df["eng_Latn"].astype(str).tolist()
    log(f"encoding {len(texts_en)} English sentences")
    en = encode_batch(texts_en)
    ven = np.stack([c.vector for c in en])
    ven /= np.linalg.norm(ven, axis=1, keepdims=True)
    reg_en = Counter(c.region for c in en).most_common(1)[0]
    out: dict = {
        "scale": scale,
        "n": len(texts_en),
        "space": "s1",
        "confidence_threshold": CONF,
        "english": {
            "region_coverage": float(np.mean([c.region_confidence >= CONF for c in en])),
            "l1_coverage": float(np.mean([c.confidence[0] >= CONF for c in en])),
            "top_region": reg_en[0],
            "top_region_share": reg_en[1] / len(en),
        },
        "languages": {},
    }
    for flores, (iso, name) in INDIC.items():
        if flores not in df.columns:
            log(f"{name}: not in FLORES, skipped")
            continue
        log(f"encoding {name} ({iso})")
        xx = encode_batch(df[flores].astype(str).tolist())
        vxx = np.stack([c.vector for c in xx])
        vxx /= np.linalg.norm(vxx, axis=1, keepdims=True)
        m = _metrics(en, xx, ven, vxx)
        m["name"] = name
        m["flores"] = flores
        out["languages"][iso] = m
    return out


def table(res: dict) -> str:
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    e = res["english"]
    lines = [
        "| language | region agrees with English | region coverage ≥ 0.7 | agreement when both"
        " confident | L1 agrees | retrieval@1 | top-region share |",
        "|---|---|---|---|---|---|---|",
        f"| English (reference) | | {pct(e['region_coverage'])} | | | | {pct(e['top_region_share'])}"
        f" ({e['top_region']}) |",
    ]
    for iso, m in res["languages"].items():
        lines.append(
            f"| {m['name']} (`{iso}`) | {pct(m['region_agree'])} | {pct(m['region_coverage'])} | "
            f"{pct(m['region_agree_confident'])} | {pct(m['l1_agree'])} | {pct(m['retrieval_at_1'])}"
            f" | {pct(m['top_region_share'])} ({m['top_region']}) |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=list(N), default="small")
    a = ap.parse_args()
    res = run(a.scale)
    p = ROOT / "bench" / "results" / f"indic-{a.scale}.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(table(res))
    log(f"wrote {p}")
