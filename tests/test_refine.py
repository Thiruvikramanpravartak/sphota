import json
from pathlib import Path

import numpy as np
import pytest

from sphota.refine import Mulberry32, centre_by_group, refine, script_of

FIX = Path(__file__).parent / "fixtures" / "refine_parity.json"


def test_mulberry32_matches_the_javascript_reference():
    r = Mulberry32(0)
    assert [round(r(), 8) for _ in range(4)] == [0.26642921, 0.00032975, 0.22327203, 0.14620215]


@pytest.mark.parametrize(
    "detail,rule", [("fine", "2sqrt"), ("medium", "sqrt"), ("coarse", "0.5sqrt")]
)
def test_refinement_matches_the_parity_fixture(detail, rule):
    f = json.loads(FIX.read_text(encoding="utf-8"))
    r = refine(
        np.array(f["V"]),
        "kmeans",
        scripts=np.array(f["keys"]),
        language="centre",
        k_rule=rule,
        l1=np.array(f["l1"]),
    )
    exp = f["expected"][detail]
    assert r.labels.tolist() == exp["labels"]
    assert r.parent == exp["parent"] and r.representative == exp["representative"]


def test_refinement_ignores_row_order_and_respects_detail():
    f = json.loads(FIX.read_text(encoding="utf-8"))
    V, keys, l1 = np.array(f["V"]), np.array(f["keys"]), np.array(f["l1"])
    a = refine(V, "kmeans", scripts=keys, language="centre", k_rule="2sqrt", l1=l1)
    p = np.random.default_rng(3).permutation(len(V))
    b = refine(V[p], "kmeans", scripts=keys[p], language="centre", k_rule="2sqrt", l1=l1[p])
    # same partition up to group numbering
    pairs = {(int(x), int(y)) for x, y in zip(a.labels[p], b.labels)}
    assert len(pairs) == a.n_groups == b.n_groups
    n = len(V)
    assert a.n_groups <= int(np.ceil(2 * np.sqrt(n)))
    c = refine(V, "kmeans", scripts=keys, language="centre", k_rule="0.5sqrt", l1=l1)
    assert c.n_groups <= int(np.ceil(0.5 * np.sqrt(n))) < a.n_groups


def test_language_centring_brings_translations_together():
    f = json.loads(FIX.read_text(encoding="utf-8"))
    V, keys = np.array(f["V"]), np.array(f["keys"])
    plain = refine(V, "kmeans", k_rule="sqrt").labels
    centred = refine(V, "kmeans", scripts=keys, language="centre", k_rule="sqrt").labels

    # groups that mix the 4 synthetic "languages" are more common after centring
    def mixed(lab):
        return np.mean(
            [len({k for k, x in zip(keys, lab) if x == g}) > 1 for g in set(lab.tolist())]
        )

    assert mixed(centred) > mixed(plain)


def test_centring_is_a_noop_for_one_language_and_script_names():
    V = np.random.default_rng(0).normal(size=(50, 8))
    assert centre_by_group(V, np.array(["LATIN"] * 50)) is V
    assert [script_of(t) for t in ["hello", "नमस्ते", "你好", "123"]] == [
        "LATIN",
        "DEVANAGARI",
        "CJK",
        "NONE",
    ]
