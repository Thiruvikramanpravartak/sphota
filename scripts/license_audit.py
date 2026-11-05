"""License audit: check Hub license tags of every model/dataset Sphota depends on, and the
licenses of installed Python dependencies. Exit 1 on anything non-permissive.

    python scripts/license_audit.py            # needs network for the Hub part
"""

from __future__ import annotations

import sys
from importlib import metadata

OK = {
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "cc-by-2.0",
    "gfdl",
    "isc",
    "psf-2.0",
    "mpl-2.0",
}
MODELS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/LaBSE",
    "BAAI/bge-m3",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Xenova/all-MiniLM-L6-v2",
    "Helsinki-NLP/opus-mt-en-hi",
    "Helsinki-NLP/opus-mt-en-dra",
    "Helsinki-NLP/opus-mt-en-inc",
    "Helsinki-NLP/opus-mt-en-es",
    "Helsinki-NLP/opus-mt-en-fr",
    "Helsinki-NLP/opus-mt-en-de",
    "Helsinki-NLP/opus-mt-en-ROMANCE",
    "Helsinki-NLP/opus-mt-en-zh",
    "Helsinki-NLP/opus-mt-en-ar",
    "Helsinki-NLP/opus-mt-en-ru",
    "Helsinki-NLP/opus-mt-en-jap",
]
DATASETS = ["mteb/flores", "Davlan/sib200", "mteb/tatoeba-bitext-mining", "wikimedia/wikipedia"]
# Xenova ONNX conversions often carry no tag; they inherit the source model's license.
INHERITS = {"Xenova/all-MiniLM-L6-v2": "apache-2.0"}


def hub() -> list[str]:
    from huggingface_hub import HfApi

    api, bad = HfApi(), []
    for kind, ids in [("model", MODELS), ("dataset", DATASETS)]:
        for i in ids:
            info = api.model_info(i) if kind == "model" else api.dataset_info(i)
            lic = {t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")} or {
                INHERITS.get(i, "?")
            }
            status = "ok" if lic & OK else "CHECK"
            print(f"{status:5} {kind:7} {i:60} {','.join(sorted(lic))}")
            if status != "ok":
                bad.append(i)
    return bad


def python_deps() -> list[str]:
    bad = []
    for d in ["numpy", "onnxruntime", "tokenizers", "huggingface_hub", "typer", "rich"]:
        try:
            m = metadata.metadata(d)
        except metadata.PackageNotFoundError:
            continue
        lic = ((m.get("License-Expression") or m.get("License") or "").splitlines() or [""])[0][:40]
        cls = [c for c in m.get_all("Classifier") or [] if c.startswith("License")]
        print(f"      python  {d:20} {lic or (cls[0] if cls else '?')}")
    return bad


if __name__ == "__main__":
    bad = python_deps() + hub()
    sys.exit(1 if bad else 0)
