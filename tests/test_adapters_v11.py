import numpy as np
import pytest

from sphota.adapters import Adapter, AdapterError, _set_name, load_adapter


def _gelu_tanh(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


def test_mlp2_roundtrip_and_forward(tmp_path):
    rng = np.random.default_rng(0)
    d, h, D = 8, 16, 4
    ad = Adapter(
        "toy/model",
        rng.normal(size=(d, D)).astype(np.float32),
        rng.normal(size=D).astype(np.float32),
        {"kind": "mlp2"},
        W1=rng.normal(size=(d, h)).astype(np.float16),
        c1=rng.normal(size=h).astype(np.float16),
        W2=rng.normal(size=(h, D)).astype(np.float16),
        c2=rng.normal(size=D).astype(np.float32),
    )
    p = ad.save(tmp_path / "toy.npz")
    back = load_adapter(p)
    assert back.W1 is not None and back.W1.dtype == np.float32
    x = rng.normal(size=(3, d)).astype(np.float32)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    want = (
        xn @ ad.A
        + ad.b
        + _gelu_tanh(xn @ ad.W1.astype(np.float32) + ad.c1.astype(np.float32))
        @ ad.W2.astype(np.float32)
        + ad.c2
    )
    np.testing.assert_allclose(back(x), want, rtol=1e-4, atol=1e-4)


def test_linear_adapter_unchanged_without_hidden_layer():
    A = np.eye(3, dtype=np.float32)
    ad = Adapter("toy", A, np.zeros(3, np.float32))
    x = np.array([[3.0, 4.0, 0.0]], np.float32)
    np.testing.assert_allclose(ad(x), [[0.6, 0.8, 0.0]], atol=1e-6)


def test_adapter_set_selection(monkeypatch):
    monkeypatch.delenv("SPHOTA_ADAPTERS", raising=False)
    assert _set_name(None) == "v1"
    assert _set_name("v1.1") == "v1.1"
    monkeypatch.setenv("SPHOTA_ADAPTERS", "v1.1")
    assert _set_name(None) == "v1.1"
    with pytest.raises(AdapterError, match="unknown adapter set"):
        _set_name("v9")
