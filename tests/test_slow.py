"""Tests that download models (run with: pytest -m slow)."""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow
GOLDEN = json.loads((Path(__file__).parent / "vectors" / "golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def onnx_encoder():
    pytest.importorskip("onnxruntime")
    from sphota.encoders import default_encoder

    return default_encoder("fp32")


def test_reference_encoder_reproduces_golden_vectors(onnx_encoder):
    texts = [g["text"] for g in GOLDEN["items"]]
    X = onnx_encoder.encode(texts)
    ref = np.array([g["vector"] for g in GOLDEN["items"]], dtype=np.float32)
    assert np.abs(X - ref).max() < 1e-4


def test_encode_text_end_to_end(onnx_encoder):
    import sphota

    for g in GOLDEN["items"][:8]:
        assert str(sphota.encode(g["text"])) == g["code"]


def test_int8_variant_is_close():
    from sphota.encoders import default_encoder

    X = default_encoder("int8").encode([g["text"] for g in GOLDEN["items"]])
    ref = np.array([g["vector"] for g in GOLDEN["items"]], dtype=np.float32)
    assert float(np.mean(np.sum(X * ref, 1))) > 0.99


def test_onnx_adapter_matches_registry_model():
    import sphota

    c = sphota.encode("The CEO resigned", model="sentence-transformers/all-MiniLM-L6-v2")
    assert c.depth == 4 and c.confidence[0] > 0
    near = sphota.encode("The chief executive stepped down", model="minilm-l6")
    assert c.similarity(near) > 0.3


def test_offline_mode_uses_cache(monkeypatch, onnx_encoder):
    from sphota import encoders

    monkeypatch.setenv("SPHOTA_OFFLINE", "1")
    monkeypatch.setattr(encoders, "_DEFAULT", {})
    enc = encoders.default_encoder("fp32")  # must load from the HF cache without network
    assert enc.encode(["hello"]).shape == (1, 384)
