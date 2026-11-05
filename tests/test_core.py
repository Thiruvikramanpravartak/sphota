"""Core behaviour on precomputed vectors: no model download, numpy only."""

import json
from pathlib import Path

import numpy as np
import pytest

import sphota
from sphota import Sphota, from_vector, parse
from sphota.codes import from_indices
from sphota.distance import distance, neighbors, path, reconstruct, similarity
from sphota.release import load_release
from sphota.space import project, quantize, topk_codes

GOLDEN = json.loads((Path(__file__).parent / "vectors" / "golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rel():
    return load_release()


@pytest.fixture(scope="module")
def vecs():
    return np.array([g["vector"] for g in GOLDEN["items"]], dtype=np.float32)


def test_release_manifest(rel):
    assert rel.version == "s1"
    assert rel.sizes == (4096, 256, 256, 256)
    assert rel.dim == 256
    assert rel.anchor["model"] == "intfloat/multilingual-e5-small"
    assert len(rel.labels) > 1000


def test_golden_codes_are_frozen(vecs):
    """THE most important test: the standard must never change."""
    codes = from_vector(vecs)
    for g, c in zip(GOLDEN["items"], codes):
        assert str(c) == g["code"], g["text"]


def test_golden_stable_mode_frozen(vecs):
    codes = from_vector(vecs, stable=True)
    for g, c in zip(GOLDEN["items"], codes):
        assert str(c) == g["code_stable"], g["text"]


def test_projection_shape_and_error(rel, vecs):
    y = project(rel, vecs)
    assert y.shape == (len(vecs), 256)
    with pytest.raises(ValueError, match="adapter"):
        project(rel, np.zeros((1, 100)))


def test_confidence_monotone_and_bounded(rel, vecs):
    q = quantize(rel, project(rel, vecs))
    c = q.confidence
    assert np.all(c <= 1 + 1e-9) and np.all(c >= 0)
    assert np.all(np.diff(c, axis=1) <= 1e-12)


def test_min_confidence_truncates(vecs):
    full = from_vector(vecs[0])
    for tau in [0.0, 0.3, 0.6, 0.9, 0.999]:
        c = from_vector(vecs[0], min_confidence=tau)
        assert full.startswith(str(c))
        assert c.depth >= 1
        if c.depth > 1:
            assert full.confidence[c.depth - 1] >= tau


def test_max_level_privacy(vecs):
    c = from_vector(vecs[0], max_level=1)
    assert c.depth == 1 and len(str(c)) == 4
    with pytest.raises(ValueError):
        from_vector(vecs[0], max_level=0)


def test_sphota_is_a_str(vecs):
    c = from_vector(vecs[0])
    assert isinstance(c, str)
    assert repr(c).startswith("Sphota('")
    assert c == str(c)
    assert json.dumps({"c": c}) == json.dumps({"c": str(c)})
    assert c.prefix(2) == str(c)[:7]
    assert c.startswith_code(str(c.prefix(1)))
    d = c.to_dict()
    assert d["code"] == str(c) and len(d["confidence"]) == 4
    import pickle

    assert pickle.loads(pickle.dumps(c)) == c


def test_alternates_and_candidates(vecs):
    seen_alt = False
    for v in vecs:
        c = from_vector(v)
        cands = c.candidates(3)
        assert 1 <= len(cands) <= 3
        ps = [p for _, p in cands]
        assert ps == sorted(ps, reverse=True)
        assert str(cands[0][0]) == str(c.full_depth_code)
        for a, p in c.alternates:
            seen_alt = True
            assert a != c and 0 < p < 1
            assert a.depth == c.depth
    assert seen_alt


def test_labels(vecs):
    c = from_vector(vecs[0])
    assert isinstance(c.label, str) and c.label
    assert c.label_in("xx") == c.label  # falls back to English
    assert sphota.label(str(c.prefix(1)))


def test_distance_geometry(rel):
    a, b = from_indices([10, 1, 2, 3]), from_indices([10, 1, 2, 4])
    assert distance(a, a, rel) == 0
    assert similarity(a, a, rel) == pytest.approx(1.0)
    assert 0 <= similarity(a, b, rel) <= 1
    far = from_indices(
        [int(np.argmax(np.linalg.norm(rel.codebooks[0] - rel.codebooks[0][10], axis=1)))]
    )
    assert distance(a.prefix(1), far, rel) > distance(
        a.prefix(1), neighbors(a.prefix(1), 1, rel)[0][0], rel
    )
    np.testing.assert_allclose(reconstruct(a.prefix(1), rel), rel.codebooks[0][10], rtol=1e-6)


def test_neighbors(rel):
    nb = neighbors(from_indices([5]), 5, rel)
    assert len(nb) == 5 and all(c.depth == 1 for c, _ in nb)
    dists = [d for _, d in nb]
    assert dists == sorted(dists)
    nb2 = neighbors(from_indices([5, 7]), 4, rel)
    assert all(c.depth == 2 and c.levels[0] == 5 and c.levels[1] != 7 for c, _ in nb2)


def test_path(rel):
    a, b = from_indices([1, 2]), from_indices([3000, 4])
    p = path(a, b, rel)
    assert p[0] == a and p[-1] == b
    assert path(a, a, rel) == [a]
    p2 = sphota.path(str(a), str(from_indices([1, 9])))
    assert [str(x) for x in p2] == [str(a), str(from_indices([1, 9]))]


def test_topk_beam(rel, vecs):
    q = quantize(rel, project(rel, vecs[:3]))
    k = topk_codes(q, 0, 5)
    assert len(k) <= 5 and abs(sum(p for _, p in k)) <= 1 + 1e-9


def test_api_code_utilities():
    a, b = str(from_indices([1, 2, 3, 4])), str(from_indices([1, 2, 3, 5]))
    assert sphota.similarity(a, b) > 0.5
    assert sphota.distance(a, a) == 0
    assert len(sphota.neighbors(a, 3)) == 3
    info = sphota.explain(a)
    assert info["code"] == a and len(info["levels"]) == 4 and info["neighbors"]
    assert sphota.prefix(a, 1) == a[:4]
    assert isinstance(sphota.prefix(a, 2), Sphota)
    assert sphota.is_valid(a) and parse(a).depth == 4


def test_encode_type_errors():
    with pytest.raises(TypeError):
        sphota.encode(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sphota.encode_batch([1, 2])  # type: ignore[list-item]
    assert sphota.encode_batch([]) == []
