import pytest

import sphota
from sphota import compare, diff, encode, encode_batch, fingerprint


def test_encode_golden_text(fake_encoder):
    c = encode("The CEO resigned")
    assert isinstance(c, sphota.Sphota) and c.depth == 4
    assert c.text == "The CEO resigned"
    assert len(c.confidence) == 4 and c.vector.shape == (256,)


def test_batch_matches_single(fake_encoder):
    texts = ["The CEO resigned", "Quantum computers use qubits.", "I love you"]
    batch = encode_batch(texts)
    assert [str(b) for b in batch] == [str(encode(t)) for t in texts]


def test_truncation_and_privacy(fake_encoder):
    c = encode("I love you", min_confidence=0.9)
    assert c.depth >= 1
    assert encode("I love you", max_level=2).depth == 2


def test_similarity_texts(fake_encoder):
    s_same = sphota.similarity("The CEO resigned", "The CEO resigned")
    assert s_same == pytest.approx(1.0)
    assert 0 <= sphota.similarity("The CEO resigned", "Quantum computers use qubits.") < 1


def test_path_from_texts(fake_encoder):
    p = sphota.path("The CEO resigned", "Quantum computers use qubits.")
    assert len(p) >= 2


def test_fingerprint_compare_diff(fake_encoder):
    doc_a = "The CEO resigned. The central bank raised interest rates by half a percentage point. Quantum computers use qubits."
    doc_b = "The CEO resigned. Add two cups of flour and stir until smooth. The museum exhibits paintings from the Renaissance."
    fa = fingerprint(doc_a)
    assert fa.n_sentences == 3
    assert sum(fa.weights.values()) == pytest.approx(1.0)
    assert fa.top(2) and "Fingerprint(" in repr(fa)
    assert compare(doc_a, doc_a) == pytest.approx(1.0)
    s = compare(doc_a, doc_b)
    assert 0 <= s < 1
    d = diff(doc_a, doc_b)
    assert d.added and d.removed
    assert "FingerprintDiff(" in repr(d)
    assert fingerprint("").n_sentences == 0
    assert fingerprint(["one sentence here", "another sentence here"]).n_sentences == 2


def test_explain_text(fake_encoder):
    info = sphota.explain("The CEO resigned")
    assert info["levels"][0]["label"]


def test_from_vector_unknown_adapter():
    import numpy as np

    from sphota.adapters import AdapterError

    with pytest.raises(AdapterError, match="train-adapter"):
        sphota.from_vector(np.ones(10), model="no/such-model")
