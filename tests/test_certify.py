import json

import numpy as np

from sphota.certify import THRESHOLDS, badge, evaluate, fit_adapter, grade, render_certificate
from sphota.release import load_release
from sphota.space import project

GOLD = json.loads(open("tests/vectors/golden.json", encoding="utf-8").read())


def test_fit_adapter_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 32)).astype(np.float32)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    M = rng.standard_normal((32, 256)).astype(np.float32)
    ad = fit_adapter("toy", X, Xn @ M, alpha=0.01)
    Y = ad(X[:50])
    assert np.corrcoef(Y.ravel(), (Xn[:50] @ M).ravel())[0, 1] > 0.99
    assert ad.meta["kind"] == "linear"


def test_evaluate_grade_render():
    rel = load_release()
    V = np.array([g["vector"] for g in GOLD["items"]], dtype=np.float32)
    Ya = {lg: project(rel, V) for lg in ["en", "hi"]}
    noisy = {
        lg: y + 0.01 * np.random.default_rng(1).standard_normal(y.shape).astype(np.float32)
        for lg, y in Ya.items()
    }
    m = evaluate(noisy, Ya)
    assert (
        m["agreement_same_text_en"][0] > 0.9
        and m["l1_neighborhood_en"] > 0.9
        and m["fidelity_en"] > 0.99
    )
    g = grade(m)
    assert g["pass"] and g["grade"] == "certified-multilingual"
    cert = {
        "model": "toy",
        "space": "s1",
        "date": "2026-11-05",
        "test_set": "golden",
        "thresholds": THRESHOLDS,
        **m,
        **g,
    }
    cert["badge"] = badge(cert)
    assert "img.shields.io" in cert["badge"]
    text = render_certificate(cert)
    assert "CERTIFIED-MULTILINGUAL" in text and "neighbourhood" in text
    bad = {
        lg: np.random.default_rng(2).standard_normal(y.shape).astype(np.float32)
        for lg, y in Ya.items()
    }
    assert not grade(evaluate(bad, Ya))["pass"]
