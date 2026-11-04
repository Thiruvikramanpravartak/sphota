import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN = json.loads((Path(__file__).parent / "vectors" / "golden.json").read_text())


class FakeEncoder:
    """Deterministic offline stand-in for the anchor: golden texts get their real vectors,
    other texts a hash-seeded mixture of golden vectors (so they land in realistic cells)."""

    name = "fake-anchor"
    dim = 384

    def __init__(self):
        self.known = {g["text"]: np.array(g["vector"], np.float32) for g in GOLDEN["items"]}
        self.basis = np.stack(list(self.known.values()))

    def encode(self, texts):
        out = []
        for t in texts:
            if t in self.known:
                out.append(self.known[t])
                continue
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            w = rng.dirichlet(np.ones(len(self.basis)) * 0.3)
            v = w @ self.basis + 0.01 * rng.standard_normal(self.dim)
            out.append((v / np.linalg.norm(v)).astype(np.float32))
        return np.stack(out)


@pytest.fixture
def fake_encoder(monkeypatch):
    from sphota import encoders

    enc = FakeEncoder()
    monkeypatch.setitem(encoders._DEFAULT, "fp32", enc)
    monkeypatch.delenv("SPHOTA_ENCODER", raising=False)
    return enc
