# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration for the offline build pipeline (not needed at runtime)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("SPHOTA_DATA_DIR", ROOT / "data"))
HF_CACHE = Path(os.environ.get("SPHOTA_HF_CACHE", ROOT / ".cache" / "hf"))

# Language order matters: smaller scales take a prefix of this list.
LANGS = [
    "en",
    "hi",
    "ta",
    "es",
    "fr",
    "de",
    "zh",
    "ar",
    "ru",
    "ja",
    "pt",
    "bn",
    "te",
    "mr",
    "sw",
    "id",
    "ko",
    "tr",
    "it",
    "vi",
]

SCALES: dict[str, dict] = {
    # tiny: bake-offs and CI smoke runs (minutes on a laptop CPU)
    "tiny": {
        "n_langs": 6,
        "wiki_en": 1500,
        "wiki_per_lang": 300,
        "tatoeba_per_lang": 200,
        "k": (256, 64, 64, 64),
    },
    # small: what the v1 release in this repo is built with (CPU, < 1 h total)
    "small": {
        "n_langs": 12,
        "wiki_en": 24000,
        "wiki_per_lang": 4000,
        "tatoeba_per_lang": 1000,
        "k": (4096, 256, 256, 256),
    },
    # full: GPU scale-up recipe (millions of sentences)
    "full": {
        "n_langs": 20,
        "wiki_en": 1_000_000,
        "wiki_per_lang": 200_000,
        "tatoeba_per_lang": 5000,
        "k": (4096, 256, 256, 256),
    },
}

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", file=sys.stderr, flush=True)
