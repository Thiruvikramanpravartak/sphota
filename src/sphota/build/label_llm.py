# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Human-readable labels: name cells with a small open LLM, translate with MarianMT.

    python -m sphota.build.label_llm --scale small            # name (resumable), translate, merge

1. English names for every L1 cell (Qwen2.5-1.5B-Instruct) and every L2 cell with >= 4 members
   (Qwen2.5-0.5B-Instruct), both Apache-2.0, greedy decoding, from up to 6 member sentences
   + c-TF-IDF keywords.
   L2 prompts include the parent's name and ask for a more specific sub-topic.
2. Translation of the English names with Helsinki-NLP opus-mt models (Apache-2.0).
3. Merge into ``labels.json`` (display labels) and ``keywords.json`` (native c-TF-IDF
   keyphrases per language, kept as independent evidence).

Every step caches to ``data/<scale>/build/`` so the job can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from sphota.build.common import DATA_DIR, HF_CACHE, ROOT, log

LLM = "Qwen/Qwen2.5-1.5B-Instruct"  # L1 names
LLM_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"  # L2 names (4x faster)
# L2 labels are translated into a subset of languages (CPU budget); others fall back to English.
L2_LANGS = ["hi", "ta", "bn", "es", "fr", "zh"]
MT = {
    "hi": ("Helsinki-NLP/opus-mt-en-hi", ""),
    "ta": ("Helsinki-NLP/opus-mt-en-dra", ">>tam<< "),
    "bn": ("Helsinki-NLP/opus-mt-en-inc", ">>ben<< "),
    "mr": ("Helsinki-NLP/opus-mt-en-inc", ">>mar<< "),
    "es": ("Helsinki-NLP/opus-mt-en-es", ""),
    "fr": ("Helsinki-NLP/opus-mt-en-fr", ""),
    "de": ("Helsinki-NLP/opus-mt-en-de", ""),
    "pt": ("Helsinki-NLP/opus-mt-en-ROMANCE", ">>pt<< "),
    "zh": ("Helsinki-NLP/opus-mt-en-zh", ">>cmn_Hans<< "),
    "ar": ("Helsinki-NLP/opus-mt-en-ar", ">>arb<< "),
    "ru": ("Helsinki-NLP/opus-mt-en-ru", ""),
    # "ja": opus-mt-en-jap is trained on Bible text and produced unusable labels (D-014);
    # Japanese falls back to English.
}


def _clean(s: str) -> str:
    s = s.strip().split("\n")[0].strip().strip("\"'*.:").strip()
    s = re.sub(r"^(label|topic)\s*[:\-]\s*", "", s, flags=re.I)
    return s[:60]


def name_cells(
    cells: dict, cache: Path, parent: dict | None = None, batch: int = 16, llm: str = LLM
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done: dict = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [k for k in cells if k not in done]
    if not todo:
        return done
    torch.set_num_threads(int(os.environ.get("SPHOTA_THREADS", "4")))
    tok = AutoTokenizer.from_pretrained(llm, cache_dir=str(HF_CACHE), padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        llm, cache_dir=str(HF_CACHE), torch_dtype=torch.bfloat16
    )

    def prompt(k: str) -> str:
        c = cells[k]
        ex = "\n".join("- " + e[:160] for e in c["examples"])
        kw = f"\nKeywords: {c['kw']}" if c.get("kw") else ""
        if parent is not None:
            p = parent.get(k.split(".")[0], "")
            msg = (
                f'These sentences belong to a sub-topic of the broader topic "{p}":\n{ex}{kw}\n\n'
                "Give a short label (2-5 words, Title Case) for this more specific sub-topic. "
                "Reply with the label only."
            )
        else:
            msg = (
                f"Sentences from one topic cluster:\n{ex}{kw}\n\nGive a short, general topic label "
                "(2-5 words, Title Case) for this cluster. Reply with the label only."
            )
        return tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
        )

    t0 = time.time()
    for s in range(0, len(todo), batch):
        keys = todo[s : s + batch]
        enc = tok([prompt(k) for k in keys], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=14, do_sample=False)
        res = tok.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        for k, r in zip(keys, res):
            done[k] = _clean(r)
        cache.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        n = s + len(keys)
        log(
            f"named {n}/{len(todo)} ({(time.time() - t0) / n:.2f}s each) e.g. {keys[0]}: {done[keys[0]]}"
        )
    return done


def name_continents(cache: Path) -> dict:
    """Name the Atlas 'continents' (agglomerative groups of L1 cells) from member names."""
    members = json.loads((cache / "continent_members.json").read_text(encoding="utf-8"))
    cells = {k: {"examples": v, "kw": ""} for k, v in members.items()}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLM, cache_dir=str(HF_CACHE), padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        LLM, cache_dir=str(HF_CACHE), torch_dtype=torch.bfloat16
    )
    out = {}
    for k, c in cells.items():
        msg = (
            "These are names of related topic areas on a map of meaning:\n"
            + "\n".join("- " + e for e in c["examples"])
            + "\n\nGive one short umbrella topic name (1-3 words, Title Case) that covers all of them, "
            "like 'Football', 'Biology' or 'European History'. "
            "Reply with the name only."
        )
        enc = tok(
            [
                tok.apply_chat_template(
                    [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
                )
            ],
            return_tensors="pt",
        )
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=8, do_sample=False)
        out[k] = _clean(tok.decode(o[0, enc.input_ids.shape[1] :], skip_special_tokens=True))
        log(f"continent {k}: {out[k]}")
    (cache / "continent_names.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8"
    )
    return out


def translate(names: dict, cache: Path, langs: list[str], batch: int = 64) -> dict:
    from transformers import MarianMTModel, MarianTokenizer

    done: dict = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    for lang in langs:
        repo, tag = MT[lang]
        have = done.setdefault(lang, {})
        todo = [k for k in names if k not in have and names[k]]
        if not todo:
            continue
        tok = MarianTokenizer.from_pretrained(repo, cache_dir=str(HF_CACHE))
        model = MarianMTModel.from_pretrained(repo, cache_dir=str(HF_CACHE))
        t0 = time.time()
        for s in range(0, len(todo), batch):
            keys = todo[s : s + batch]
            enc = tok(
                [tag + names[k] for k in keys], return_tensors="pt", padding=True, truncation=True
            )
            out = model.generate(**enc, max_new_tokens=32, num_beams=2)
            for k, r in zip(keys, tok.batch_decode(out, skip_special_tokens=True)):
                have[k] = r.strip().rstrip("।.。")[:80]
        cache.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        log(f"translated {len(todo)} labels → {lang} in {time.time() - t0:.0f}s")
    return done


def merge(scale: str, release_dir: Path, langs: list[str]) -> None:
    cache = DATA_DIR / scale / "build"
    names = {
        **json.loads((cache / "names_l1.json").read_text(encoding="utf-8")),
        **json.loads((cache / "names_l2.json").read_text(encoding="utf-8")),
    }
    tr = json.loads((cache / "translations.json").read_text(encoding="utf-8"))
    kw_path = release_dir / "keywords.json"
    lab_path = release_dir / "labels.json"
    # the c-TF-IDF output currently in labels.json becomes keywords.json (first run only)
    keywords = (
        json.loads(kw_path.read_text(encoding="utf-8"))
        if kw_path.exists()
        else json.loads(lab_path.read_text(encoding="utf-8"))
    )
    labels: dict[str, dict[str, str]] = {}
    for k, en in names.items():
        # the small model sometimes answers in the examples' language: keep Latin-script names only
        if not en or sum(ch.isascii() for ch in en) < 0.9 * len(en):
            continue
        labels[k] = {"en": en}
        for lg in langs:
            v = tr.get(lg, {}).get(k)
            if v:
                labels[k][lg] = v
    kw_path.write_text(
        json.dumps(keywords, ensure_ascii=False, sort_keys=True, indent=0), encoding="utf-8"
    )
    lab_path.write_text(
        json.dumps(labels, ensure_ascii=False, sort_keys=True, indent=0), encoding="utf-8"
    )
    log(f"merged {len(labels)} labels; keywords for {len(keywords)} keys")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    ap.add_argument("--release", default=str(ROOT / "src" / "sphota" / "data" / "v1"))
    ap.add_argument(
        "--langs",
        nargs="*",
        default=["hi", "ta", "bn", "es", "fr", "de", "pt", "zh", "ar", "ru"],
    )
    ap.add_argument(
        "--step", choices=["all", "name", "translate", "merge", "continents"], default="all"
    )
    a = ap.parse_args()
    cache = DATA_DIR / a.scale / "build"
    if a.step in ("all", "name"):
        l1 = json.loads((cache / "cells_l1.json").read_text(encoding="utf-8"))
        n1 = name_cells(l1, cache / "names_l1.json")
        l2 = json.loads((cache / "cells_l2.json").read_text(encoding="utf-8"))
        name_cells(l2, cache / "names_l2.json", parent=n1, llm=LLM_SMALL)
    if a.step == "continents":
        name_continents(cache)
        return
    if a.step in ("all", "translate"):
        n1 = json.loads((cache / "names_l1.json").read_text(encoding="utf-8"))
        n2 = json.loads((cache / "names_l2.json").read_text(encoding="utf-8"))
        translate(n1, cache / "translations.json", a.langs)  # L1: every label language
        translate(n2, cache / "translations.json", [lg for lg in L2_LANGS if lg in a.langs])
    if a.step in ("all", "merge"):
        merge(a.scale, Path(a.release), a.langs)
        from sphota.build.release import refresh_manifest

        refresh_manifest(Path(a.release))


if __name__ == "__main__":
    main()
