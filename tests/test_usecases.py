import json

import numpy as np
import pytest

from sphota.usecases import (
    CodeIndex,
    PrefixRouter,
    SemanticCache,
    analyze_file,
    build_report,
    compare_documents,
    cross_lingual_search,
    dedupe,
    group_by_prefix,
    prefix_counts,
    render_html,
)

GOLD = [
    "The CEO resigned",
    "El director ejecutivo dimitió",
    "Der Vorstandschef ist zurückgetreten",
    "How do I reset my password?",
    "¿Cómo restablezco mi contraseña?",
    "Quantum computers use qubits.",
    "The museum exhibits paintings from the Renaissance.",
    "Add two cups of flour and stir until smooth.",
]


def test_code_index_search_and_dedupe(fake_encoder):
    idx = CodeIndex()
    codes = idx.add_texts(GOLD)
    assert len(idx) == len(GOLD)
    hits = idx.search("The CEO resigned", k=3)
    assert hits[0].item == "The CEO resigned" and hits[0].score == pytest.approx(1.0)
    assert "Hit(" in repr(hits[0])
    idx.add_codes(["dup"], [codes[0]])
    assert any("dup" in g for g in idx.duplicates(level=4))
    idx.remove("dup")
    assert "dup" not in idx.codes()
    assert dedupe(GOLD + ["The CEO resigned"])
    assert cross_lingual_search("The CEO resigned", GOLD, k=2)[0].item == "The CEO resigned"


def test_code_index_codes_only_and_within(fake_encoder):
    idx = CodeIndex(store_vectors=False)
    codes = idx.add_texts(GOLD)
    hits = idx.search("The CEO resigned", k=3)
    assert hits[0].item == "The CEO resigned"
    region = str(codes[0].prefix(1))
    vidx = CodeIndex()
    vidx.add_texts(GOLD)
    inside = vidx.search("Quantum computers use qubits.", k=5, within=region)
    assert inside and all(h.code.startswith(region) for h in inside)
    assert idx.search("The CEO resigned", k=5, within=region)[0].code.startswith(region)
    assert vidx.search("x", within=str(codes[5].prefix(4))) is not None


def test_code_index_vectors(fake_encoder):
    X = fake_encoder.encode(GOLD)
    idx = CodeIndex()
    idx.add_vectors(X, ids=list(range(len(GOLD))), items=GOLD)
    assert idx.search_vector(X[5], k=1)[0].item == GOLD[5]


def test_semantic_cache_memory_and_sqlite(fake_encoder, tmp_path):
    for cache in [
        SemanticCache(depth=3),
        SemanticCache(depth=2, path=tmp_path / "c.db", ttl_seconds=60),
    ]:
        assert cache.get("The CEO resigned") is None
        assert cache.set("The CEO resigned", {"answer": 42})
        assert cache.get("The CEO resigned") == {"answer": 42}
        calls = []
        v = cache.get_or_compute(
            "Quantum computers use qubits.", lambda t: calls.append(t) or "computed"
        )
        assert v == "computed" and calls
        assert (
            cache.get_or_compute("Quantum computers use qubits.", lambda t: "again") == "computed"
        )
        assert cache.stats.hits >= 2 and 0 < cache.stats.hit_rate < 1
        assert len(cache) == 2
    strict = SemanticCache(depth=4, min_confidence=0.999)
    assert not strict.set("I love you", 1)
    assert strict.stats.skipped_low_confidence == 1
    probing = SemanticCache(depth=1, probe_alternates=True)
    probing.get("I love you")


def test_router(fake_encoder):
    r = PrefixRouter(default="other")
    r.fit(GOLD[:3] + GOLD[3:5], ["exec"] * 3 + ["it"] * 2)
    assert r.route("The CEO resigned").route == "exec"
    assert r.route("How do I reset my password?").route == "it"
    out = r.route_batch(["Quantum computers use qubits."])
    assert out[0].route in {"exec", "it", "other"}
    assert "Route(" in repr(out[0])
    r2 = PrefixRouter(neighbor_fallback=0)
    from sphota import encode

    r2.add_rule(str(encode("The CEO resigned").prefix(1)), "x")
    assert r2.route("The CEO resigned").route == "x"


def test_analytics(fake_encoder, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "id,text,lang\n" + "\n".join(f'{i},"{t}",en' for i, t in enumerate(GOLD * 2)),
        encoding="utf-8",
    )
    res = analyze_file(p, out=tmp_path / "r.html")
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert res["n"] == 16 and "<details" in html and "Sphota" in html
    rep = build_report(GOLD, min_count=3)
    assert rep.suppressed > 0 or all(r.count >= 3 for r in rep.roots)
    assert rep.to_dict()["n"] == len(GOLD)
    assert render_html(build_report(GOLD))
    j = tmp_path / "t.jsonl"
    j.write_text("\n".join(json.dumps({"body": t}) for t in GOLD), encoding="utf-8")
    assert analyze_file(j, out=tmp_path / "r2.html")["n"] == len(GOLD)
    t = tmp_path / "t.txt"
    t.write_text("\n".join(GOLD), encoding="utf-8")
    assert analyze_file(t, out=tmp_path / "r3.html", limit=3)["n"] == 3
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="--column"):
        analyze_file(bad, out=tmp_path / "x.html")


def test_prefix_helpers():
    codes = ["79Q9-AX-69-Q1", "79Q9-AX-69-Q2", "JS08-AX"]
    assert prefix_counts(codes)["79Q9"] == 2
    assert group_by_prefix(["a", "b", "c"], codes, level=2)["79Q9-AX"] == ["a", "b"]


def test_compare_documents(fake_encoder, tmp_path):
    docs = {"a": " ".join(GOLD[:3]), "b": " ".join(GOLD[3:6]), "c": " ".join(GOLD[:2])}
    cmp = compare_documents(docs)
    assert cmp.similarity.shape == (3, 3)
    assert np.allclose(np.diag(cmp.similarity), 1)
    assert cmp.most_similar("a", 1)[0][0] == "c"
    assert cmp.diff("a", "b").similarity < 1
    assert "a" in cmp.table()
    f = tmp_path / "d.txt"
    f.write_text(docs["a"], encoding="utf-8")
    assert compare_documents([f, f]).similarity[0, 1] == pytest.approx(1.0)
