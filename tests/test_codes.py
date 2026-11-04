import json
from pathlib import Path

import pytest

from sphota.codes import (
    ALPHABET,
    BITS,
    V1_SIZES,
    Code,
    SphotaCodeError,
    check_symbol,
    from_indices,
    index_to_symbols,
    is_valid,
    parse,
    symbols_to_index,
)

VECTORS = json.loads((Path(__file__).parent / "vectors" / "codes.json").read_text())


@pytest.mark.parametrize("case", VECTORS)
def test_vectors_roundtrip(case):
    c = from_indices(case["indices"])
    assert str(c) == case["display"]
    assert c.format(version=True, check=True) == case["full"]
    assert parse(case["full"]) == c
    assert parse(case["display"].lower()) == c
    assert parse(case["display"].replace("-", "")) == c


def test_permutation_is_bijective_on_valid_range():
    for lvl in range(4):
        seen = set()
        for i in range(V1_SIZES[lvl]):
            s = index_to_symbols(lvl, i)
            assert symbols_to_index(lvl, s) == i
            seen.add(s)
        assert len(seen) == V1_SIZES[lvl]


def test_aliases_and_whitespace():
    c = from_indices([5, 6, 7, 8])
    s = str(c)
    noisy = " " + s.lower().replace("0", "o").replace("1", "l") + " \n"
    assert parse(noisy) == c


def test_prefix_semantics():
    c = from_indices([100, 2, 3, 4])
    p = c.prefix(2)
    assert p.depth == 2
    assert p.is_prefix_of(c)
    assert not c.is_prefix_of(p)
    assert c.common_prefix(from_indices([100, 2, 9])) == 2
    assert str(c).startswith(str(p))
    with pytest.raises(SphotaCodeError):
        c.prefix(5)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "XYZ",
        "7F3A-C2-91-0B-11",
        "7F3A--C2",
        "7F3-AC2",
        "s2:7F3A",
        "7F3A-C2-91-0B.!",
        "UUUU",
        "12345678901",
    ],
)
def test_rejects_malformed(bad):
    assert not is_valid(bad)
    with pytest.raises(SphotaCodeError):
        parse(bad)


def test_error_messages_are_actionable():
    with pytest.raises(SphotaCodeError, match="XXXX-XX-XX-XX"):
        parse("7F3-AC2")
    with pytest.raises(SphotaCodeError, match="migrate"):
        parse("s9:79Q9")


def test_check_symbol_detects_errors_and_depth():
    c = from_indices([123, 45, 6, 255])
    good = c.format(check=True)
    assert is_valid(good)
    chk = check_symbol(c.levels)
    other = next(x for x in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" if x != chk)
    assert not is_valid(f"{c}.{other}")
    # the depth is folded into the check value: a prefix carries its own check symbol
    p = c.prefix(2)
    assert parse(p.format(check=True)) == p


def test_typo_detection_rate():
    import random

    rnd = random.Random(0)
    n = detected = 0
    for _ in range(2000):
        c = from_indices([rnd.randrange(V1_SIZES[0])] + [rnd.randrange(256) for _ in range(3)])
        s = list(str(c))
        j = rnd.choice([k for k, ch in enumerate(s) if ch != "-"])
        s[j] = rnd.choice([a for a in ALPHABET if a != s[j]])
        n += 1
        detected += not is_valid("".join(s))
    assert detected / n > 0.8


def test_code_validation():
    with pytest.raises(SphotaCodeError):
        Code(())
    with pytest.raises(SphotaCodeError):
        Code((V1_SIZES[0],))
    with pytest.raises(SphotaCodeError):
        index_to_symbols(1, 256)
    with pytest.raises(SphotaCodeError):
        parse(123)  # type: ignore[arg-type]
    assert BITS == (20, 10, 10, 10)


def test_ordering_and_hash():
    a, b = from_indices([1]), from_indices([1, 2])
    assert {a: 1, b: 2}[from_indices([1])] == 1
    assert sorted([b, a])[0] == a
