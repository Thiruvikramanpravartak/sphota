# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Automatic quality gate for machine-generated labels (run before any public release).

    python -m sphota.build.label_filter            # filters src/sphota/data/v1/labels.json

Every label is checked; failing ones are removed from ``labels.json`` (display falls back to
English, or to no label) and written with the reason to ``data/<scale>/build/labels_review.json``
for human review. Checks:

* **offensive / sensitive**: English labels matching a blocklist of sexual, profane or slur
  terms; the whole cell (all languages) is held for review;
* **wrong script**: e.g. a Hindi label that is not mostly Devanagari;
* **degenerate output**: a repeated token, untranslated copy of the English label in a
  non-Latin language, too long, or mostly digits/punctuation.

This is a filter, not a guarantee: native-speaker review is still required before launch.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from sphota.build.common import DATA_DIR, ROOT, log

# Deliberately broad: false positives only send a label to human review.
BLOCKLIST = {
    "sex", "sexual", "sexy", "porn", "pornography", "porno", "erotic", "erotica", "nude", "nudity",
    "naked", "fetish", "ejaculation", "ejaculatory", "orgasm", "genital", "genitals", "penis",
    "vagina", "masturbation", "prostitution", "prostitute", "escort", "brothel", "rape", "incest",
    "pedophile", "pedophilia", "bdsm", "xxx", "fuck", "shit", "bitch", "slut", "whore", "cunt",
    "nigger", "faggot", "retard", "kike", "chink", "spic", "tranny", "suicide", "self-harm",
    "nazi", "terrorist", "terrorism", "jihad", "genocide", "massacre",
}
SCRIPTS = {
    "hi": "DEVANAGARI", "mr": "DEVANAGARI", "bn": "BENGALI", "ta": "TAMIL", "te": "TELUGU",
    "ar": "ARABIC", "ru": "CYRILLIC", "zh": "CJK", "ja": "CJK",
}
LATIN = {"en", "es", "fr", "de", "pt", "it", "id", "tr", "vi", "sw"}


def _script_share(text: str, script: str) -> float:
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return 0.0
    hit = 0
    for ch in letters:
        name = unicodedata.name(ch, "")
        if script == "CJK":
            hit += name.startswith(("CJK", "HIRAGANA", "KATAKANA"))
        else:
            hit += name.startswith(script)
    return hit / len(letters)


def check(lang: str, text: str, english: str | None = None) -> str | None:
    """Return a rejection reason, or None if the label passes."""
    t = text.strip()
    if not t:
        return "empty"
    if len(t) > 70:
        return "too long"
    # letters + combining marks (Indic vowel signs are category M)
    letters = sum(unicodedata.category(ch)[0] in "LM" for ch in t)
    if letters < 0.6 * len(t.replace(" ", "")):
        return "mostly non-letters"
    words = re.findall(r"\w+", t.lower())
    if len(words) >= 2 and len(set(words)) <= len(words) / 2:
        return "repeated tokens"
    # CJK labels often contain Latin brand/place names; require less of the native script
    if lang in SCRIPTS and _script_share(t, SCRIPTS[lang]) < (0.3 if SCRIPTS[lang] == "CJK" else 0.6):
        return f"wrong script for {lang}"
    if lang in LATIN and _script_share(t, "LATIN") < 0.8:
        return f"wrong script for {lang}"
    if lang != "en" and english and lang in SCRIPTS and t.lower() == english.lower():
        return "untranslated"
    return None


def offensive(english: str) -> bool:
    words = set(re.findall(r"[a-z][a-z-]*", english.lower()))
    return bool(words & BLOCKLIST)


def filter_labels(labels: dict[str, dict[str, str]]) -> tuple[dict, dict]:
    kept: dict[str, dict[str, str]] = {}
    review: dict[str, dict] = {}
    for key, entry in labels.items():
        en = entry.get("en", "")
        if en and offensive(en):
            review[key] = {"reason": "offensive/sensitive (held for review)", "labels": entry}
            continue
        out = {}
        for lang, text in entry.items():
            reason = check(lang, text, en)
            if reason:
                review.setdefault(key, {"labels": entry, "rejected": {}}).setdefault("rejected", {})[lang] = reason
            else:
                out[lang] = text
        if out:
            kept[key] = out
    return kept, review


def run(release: Path, scale: str = "small") -> dict:
    p = release / "labels.json"
    labels = json.loads(p.read_text(encoding="utf-8"))
    kept, review = filter_labels(labels)
    p.write_text(json.dumps(kept, ensure_ascii=False, sort_keys=True, indent=0), encoding="utf-8")
    out = DATA_DIR / scale / "build" / "labels_review.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    n_in = sum(len(v) for v in labels.values())
    n_out = sum(len(v) for v in kept.values())
    per_lang: dict[str, int] = {}
    for v in review.values():
        for lang in v.get("rejected", {}):
            per_lang[lang] = per_lang.get(lang, 0) + 1
    stats = {"labels_in": n_in, "labels_kept": n_out,
             "cells_held_offensive": sum(1 for v in review.values() if "reason" in v),
             "rejected_per_language": dict(sorted(per_lang.items()))}
    log(f"label filter: {stats}")
    return stats


def main() -> None:
    from sphota.build.release import refresh_manifest

    rel = ROOT / "src" / "sphota" / "data" / "v1"
    stats = run(rel)
    (ROOT / "bench" / "results" / "label-filter-small.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    refresh_manifest(rel)


if __name__ == "__main__":
    main()
