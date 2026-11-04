"""Recipe (NOT run in CI): train a Sphota s1 adapter for Cohere embeddings with YOUR key.

    export CO_API_KEY=...
    pip install "sphota[encode,build]" cohere
    python adapters/recipes/openai_adapter.py --model embed-multilingual-v3.0 --n 20000

Cost: ~20k short texts ≈ 0.5M tokens (well under US$0.05 with embed-multilingual-v3.0 at
the time of writing). The resulting adapter is yours: Sphota does not distribute adapters
trained on proprietary API outputs (check the provider's terms before sharing one).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from sphota.certify import _default_texts, fit_adapter, render_certificate
from sphota.encoders import default_encoder
from sphota.release import load_release
from sphota.space import project


def embed_openai(texts: list[str], model: str, batch: int = 96) -> np.ndarray:
    """Cohere embeddings (name kept for symmetry with the OpenAI recipe)."""
    import cohere

    client = cohere.Client(os.environ["CO_API_KEY"])
    out = []
    for s in range(0, len(texts), batch):
        resp = client.embed(texts=texts[s : s + batch], model=model, input_type="clustering")
        out.extend(resp.embeddings)
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="embed-multilingual-v3.0")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="cohere-embed-multilingual-v3.npz")
    a = ap.parse_args()
    texts = _default_texts(a.n)
    Y = project(load_release(), default_encoder().encode(texts))
    X = embed_openai(texts, a.model)
    ad = fit_adapter(f"cohere/{a.model}", X, Y, meta={"model_license": "proprietary (Cohere ToS)"})
    print("saved", ad.save(a.out))
    # Evaluate on FLORES devtest like `sphota certify` (embeds ~12k sentences, ~0.3M tokens)
    from sphota.certify import evaluate, grade, load_eval

    data = load_eval(quick=True)
    Ym = {k: ad(embed_openai(v, a.model)) for k, v in data.items()}
    Ya = {k: project(load_release(), default_encoder().encode(v)) for k, v in data.items()}
    m = evaluate(Ym, Ya)
    cert = {
        "model": f"cohere/{a.model}",
        "space": "s1",
        "date": "",
        "test_set": "FLORES-200 devtest (quick)",
        **m,
        **grade(m),
        "badge": "",
    }
    print(render_certificate(cert))


if __name__ == "__main__":
    main()
