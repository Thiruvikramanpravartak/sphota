# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare the open datasets used to build the Sphota v1 space.

All sources are permissively licensed (see docs/LICENSES-THIRD-PARTY.md):

* FLORES-200 (CC-BY-SA-4.0) via ``mteb/flores``: parallel sentences, 200+ languages.
* SIB-200 (CC-BY-SA-4.0): topic labels for FLORES sentences (evaluation only).
* Tatoeba (CC-BY-2.0) via ``mteb/tatoeba-bitext-mining``: short parallel pairs.
* STS Benchmark via ``sentence-transformers/stsb``: high-score pairs used as paraphrases.
* Wikipedia (CC-BY-SA-3.0) via ``wikimedia/wikipedia``: topical diversity.

Every function is cached under ``data/`` and is safe to re-run.

Usage::

    python -m sphota.build.data --scale small
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re

import pandas as pd

from sphota.build.common import DATA_DIR, HF_CACHE, LANGS, SCALES, log

FLORES_TO_ISO = {
    "eng_Latn": "en",
    "hin_Deva": "hi",
    "tam_Taml": "ta",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "zho_Hans": "zh",
    "arb_Arab": "ar",
    "rus_Cyrl": "ru",
    "jpn_Jpan": "ja",
    "por_Latn": "pt",
    "ben_Beng": "bn",
    "tel_Telu": "te",
    "mar_Deva": "mr",
    "swh_Latn": "sw",
    "ind_Latn": "id",
    "kor_Hang": "ko",
    "tur_Latn": "tr",
    "ita_Latn": "it",
    "vie_Latn": "vi",
}
ISO_TO_FLORES = {v: k for k, v in FLORES_TO_ISO.items()}
TATOEBA_ISO3 = {
    "hi": "hin",
    "ta": "tam",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "zh": "cmn",
    "ar": "ara",
    "ru": "rus",
    "ja": "jpn",
    "pt": "por",
    "bn": "ben",
    "te": "tel",
    "mr": "mar",
    "sw": "swh",
    "id": "ind",
    "ko": "kor",
    "tr": "tur",
    "it": "ita",
    "vi": "vie",
}


def _hf(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", cache_dir=str(HF_CACHE))


def flores(split: str, langs: list[str]) -> pd.DataFrame:
    """FLORES-200 ``dev`` or ``devtest`` as a wide frame: one column per ISO language."""
    df = pd.read_parquet(_hf("mteb/flores", f"{split}.parquet"))
    out = pd.DataFrame({iso: df[ISO_TO_FLORES[iso]].astype(str) for iso in langs})
    out.index.name = "meaning_id"
    return out


def sib200(split: str = "test") -> pd.DataFrame:
    """SIB-200 English topic labels (index_id aligns with FLORES rows)."""
    p = _hf("Davlan/sib200", f"data/eng_Latn/{split}.tsv")
    return pd.read_csv(p, sep="\t")


def tatoeba(langs: list[str], per_lang: int) -> pd.DataFrame:
    rows = []
    for iso in langs:
        if iso == "en" or iso not in TATOEBA_ISO3:
            continue
        try:
            p = _hf("mteb/tatoeba-bitext-mining", f"test/{TATOEBA_ISO3[iso]}-eng.jsonl.gz")
        except Exception as e:  # pragma: no cover - network dependent
            log(f"tatoeba {iso}: skipped ({type(e).__name__})")
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        for it in items[:per_lang]:
            rows.append({"lang": iso, "text": it["sentence1"], "en": it["sentence2"]})
    return pd.DataFrame(rows)


def stsb_paraphrases(min_score: float = 0.9) -> pd.DataFrame:
    """English sentence pairs scored >= 4.5/5 in STS-B (score normalised to 0..1)."""
    frames = []
    for split in ["train", "validation", "test"]:
        p = _hf("sentence-transformers/stsb", f"data/{split}-00000-of-00001.parquet")
        d = pd.read_parquet(p)
        d["split"] = split
        frames.append(d)
    d = pd.concat(frames)
    d = d[d["score"] >= min_score].rename(columns={"sentence1": "a", "sentence2": "b"})
    return d[["a", "b", "score", "split"]].reset_index(drop=True)


_SENT_SPLIT = re.compile(r"(?<=[.!?。！？।])\s+|(?<=[。！？])")


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return parts


def _good_sentence(s: str) -> bool:
    n = len(s)
    if n < 25 or n > 300:
        return False
    if s.count("|") or s.count("=") > 1 or s.startswith(("*", "#", "{", "!")):
        return False
    letters = sum(ch.isalpha() for ch in s)
    return letters / n > 0.6


def wikipedia(lang: str, n_sentences: int, seed: int = 0, max_per_article: int = 3) -> list[dict]:
    """Sample sentences from random row groups of the 20231101 Wikipedia dump."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/wikimedia/wikipedia/20231101.{lang}/*.parquet"))
    rng = random.Random(f"{seed}-{lang}")
    out: list[dict] = []
    tried: set[tuple[str, int]] = set()
    while len(out) < n_sentences and len(tried) < 200:
        fpath = rng.choice(files)
        with fs.open(fpath, "rb", block_size=2**20) as fh:
            pf = pq.ParquetFile(fh)
            rg = rng.randrange(pf.num_row_groups)
            if (fpath, rg) in tried:
                continue
            tried.add((fpath, rg))
            table = pf.read_row_group(rg, columns=["title", "text"])
        titles = table.column("title").to_pylist()
        texts = table.column("text").to_pylist()
        idx = list(range(len(texts)))
        rng.shuffle(idx)
        for i in idx:
            sents = [s for s in split_sentences(texts[i][:6000]) if _good_sentence(s)]
            rng.shuffle(sents)
            for s in sents[:max_per_article]:
                out.append({"lang": lang, "title": titles[i], "text": s})
            if len(out) >= n_sentences:
                break
    return out[:n_sentences]


def build(scale: str) -> None:
    cfg = SCALES[scale]
    langs = LANGS[: cfg["n_langs"]]
    out = DATA_DIR / scale
    out.mkdir(parents=True, exist_ok=True)

    for split in ["dev", "devtest"]:
        p = out / f"flores_{split}.parquet"
        if not p.exists():
            flores(split, langs).to_parquet(p)
            log(f"wrote {p}")

    p = out / "sib200_test.parquet"
    if not p.exists():
        sib = pd.concat([sib200(s).assign(split=s) for s in ["train", "dev", "test"]])
        sib.to_parquet(p)
        log(f"wrote {p}")

    p = out / "tatoeba.parquet"
    if not p.exists():
        tatoeba(langs, cfg["tatoeba_per_lang"]).to_parquet(p)
        log(f"wrote {p}")

    p = out / "stsb_paraphrases.parquet"
    if not p.exists():
        stsb_paraphrases().to_parquet(p)
        log(f"wrote {p}")

    p = out / "wiki.parquet"
    if not p.exists():
        rows: list[dict] = []
        for lang in langs:
            n = cfg["wiki_en"] if lang == "en" else cfg["wiki_per_lang"]
            got = wikipedia(lang, n)
            log(f"wikipedia {lang}: {len(got)} sentences")
            rows.extend(got)
        pd.DataFrame(rows).to_parquet(p)
        log(f"wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", choices=list(SCALES), default="small")
    build(ap.parse_args().scale)


if __name__ == "__main__":
    main()
