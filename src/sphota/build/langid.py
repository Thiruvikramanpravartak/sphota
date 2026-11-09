# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Train the anchor-space language identifier used by per-file refinement (D-038).

    python -m sphota.build.langid --scale small

Multinomial logistic regression on L2-normalized raw anchor vectors (multilingual-e5-small),
trained on MASSIVE **train** utterances (51 locales; texts that also occur in the test split are
dropped) plus FLORES-200 dev (12 languages). No evaluation split is used for training. Accuracy is
reported on a held-out 10% of the training pool and on FLORES-200 devtest (report only).

Output: ``src/sphota/data/langid/langid.npz`` (W, b, classes) + ``langid.json`` (sha256, metrics).
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np

from sphota.build import embed
from sphota.build.common import ROOT, log
from sphota.build.domains import _hf

ANCHOR = "intfloat/multilingual-e5-small"
OUT = ROOT / "src" / "sphota" / "data" / "langid"
FLORES_TO_MASSIVE = {"zh": "zh-CN"}


def massive_locales() -> list[str]:
    from huggingface_hub import list_repo_files

    fs = list_repo_files("mteb/amazon_massive_intent", repo_type="dataset")
    return sorted(f.split("/")[1].split(".")[0] for f in fs if f.startswith("train/"))


def run(scale: str, per_lang: int) -> dict:
    from sklearn.linear_model import LogisticRegression

    from sphota.build.release import load

    texts, labels = [], []
    for lg in massive_locales():
        with gzip.open(_hf("mteb/amazon_massive_intent", f"test/{lg}.json.gz"), "rt") as f:
            test = {json.loads(line)["text"] for line in f}
        with gzip.open(_hf("mteb/amazon_massive_intent", f"train/{lg}.json.gz"), "rt") as f:
            rows = [json.loads(line)["text"] for line in f]
        rows = [t for t in dict.fromkeys(rows) if t not in test]
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(rows), min(per_lang, len(rows)), replace=False).tolist())
        texts += [rows[i] for i in pick]
        labels += [lg] * len(pick)
    X = embed.cached(ANCHOR, f"langid-massive-train-{len(texts)}", texts, scale)
    langs, dev, test, _wiki, _tat = load(scale)
    Xf = [embed.cached(ANCHOR, f"flores_dev_{lg}", dev[lg].tolist(), scale) for lg in langs]
    X = np.concatenate([X, *Xf])
    y = np.array(labels + [FLORES_TO_MASSIVE.get(lg, lg) for lg in langs for _ in range(len(dev))])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    perm = np.random.default_rng(1).permutation(len(X))
    cut = int(0.9 * len(X))
    tr, ho = perm[:cut], perm[cut:]
    clf = LogisticRegression(C=20.0, max_iter=2000)
    clf.fit(X[tr], y[tr])
    acc_holdout = float((clf.predict(X[ho]) == y[ho]).mean())
    Xt = np.concatenate(
        [embed.cached(ANCHOR, f"flores_devtest_{lg}", test[lg].tolist(), scale) for lg in langs]
    )
    Xt = Xt / np.linalg.norm(Xt, axis=1, keepdims=True)
    yt = np.array([FLORES_TO_MASSIVE.get(lg, lg) for lg in langs for _ in range(len(test))])
    acc_flores = float((clf.predict(Xt) == yt).mean())
    clf.fit(X, y)  # final model on the whole training pool
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "langid.npz"
    np.savez(
        p,
        W=clf.coef_.T.astype(np.float32),
        b=clf.intercept_.astype(np.float32),
        classes=np.array(clf.classes_.tolist()),
    )
    from sphota.release import sha256_file

    meta = {
        "model": "multinomial logistic regression on L2-normalized anchor vectors",
        "anchor": ANCHOR,
        "classes": clf.classes_.tolist(),
        "train_rows": len(X),
        "data": [
            "MASSIVE train (51 locales, test texts excluded)",
            "FLORES-200 dev (12 languages)",
        ],
        "accuracy_heldout_10pct": round(acc_holdout, 4),
        "accuracy_flores_devtest": round(acc_flores, 4),
        "sha256": sha256_file(p),
    }
    (OUT / "langid.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(
        f"langid: {len(meta['classes'])} classes, held-out acc {acc_holdout:.4f}, FLORES devtest acc {acc_flores:.4f}"
    )
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", default="small")
    ap.add_argument("--per-lang", type=int, default=1500)
    a = ap.parse_args()
    run(a.scale, a.per_lang)
