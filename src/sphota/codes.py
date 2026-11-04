# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Sphota code format (spec: docs/spec/01-code-format.md). Pure Python, no numpy needed.

A full code has 4 levels written ``XXXX-XX-XX-XX`` in Crockford base32. Level 1 uses 4
symbols (20 bits), levels 2-4 use 2 symbols (10 bits) each. A codebook index ``i`` at level
``l`` is written as the symbol value ``s = (a_l * i + b_l) mod 2**bits_l``; ``a_l`` is odd so
the map is a bijection. Only ``K_l`` of the ``2**bits_l`` symbol values are valid, so most
typos are detected even without the optional check symbol.

Accepted input forms (case-insensitive, whitespace ignored)::

    7F3A-C2-91-0B        full code
    7F3A-C2              level-2 prefix
    s1:7F3A-C2-91-0B     with version tag
    7F3A-C2-91-0B.Q      with check symbol (mod-37 over the code's integer value)
    7F3AC2910B           without hyphens (widths are fixed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
CHECK_ALPHABET = ALPHABET + "*~$=U"
_DECODE = {c: i for i, c in enumerate(ALPHABET)}
# Crockford aliases: O -> 0, I/L -> 1
_DECODE.update({"O": 0, "I": 1, "L": 1})

WIDTHS: tuple[int, ...] = (4, 2, 2, 2)
BITS: tuple[int, ...] = tuple(5 * w for w in WIDTHS)
MAX_LEVELS = len(WIDTHS)

# v1 symbol permutation constants (frozen; see spec section 3.3)
PERM_A: tuple[int, ...] = (0x5BD1F, 0x2A5, 0x1B3, 0x3C7)
PERM_B: tuple[int, ...] = (0x3A6E9, 0x15D, 0x0C9, 0x2E1)
# v1 codebook sizes (frozen; must match release/v1/manifest.json)
V1_SIZES: tuple[int, ...] = (4096, 256, 256, 256)

_RE = re.compile(r"^(?:(?P<ver>S\d+):)?(?P<body>[0-9A-Z-]+?)(?:\.(?P<chk>[0-9A-Z*~$=]))?$")


class SphotaCodeError(ValueError):
    """Raised for malformed codes. The message always says how to fix the input."""


def _inv(a: int, m: int) -> int:
    return pow(a, -1, m)


def index_to_symbols(level: int, index: int, sizes: tuple[int, ...] = V1_SIZES) -> str:
    if not 0 <= index < sizes[level]:
        raise SphotaCodeError(f"level {level + 1} index {index} out of range [0, {sizes[level]})")
    m = 1 << BITS[level]
    s = (PERM_A[level] * index + PERM_B[level]) % m
    out = []
    for _ in range(WIDTHS[level]):
        out.append(ALPHABET[s & 31])
        s >>= 5
    return "".join(reversed(out))


def symbols_to_index(level: int, sym: str, sizes: tuple[int, ...] = V1_SIZES) -> int:
    s = 0
    for ch in sym:
        if ch not in _DECODE:
            raise SphotaCodeError(
                f"invalid character {ch!r}; Sphota codes use Crockford base32 ({ALPHABET})")
        s = (s << 5) | _DECODE[ch]
    m = 1 << BITS[level]
    i = ((s - PERM_B[level]) * _inv(PERM_A[level], m)) % m
    if i >= sizes[level]:
        raise SphotaCodeError(
            f"{sym!r} is not a valid level-{level + 1} symbol in Sphota v1 "
            "(probably a typo: check each character, or copy the code again)")
    return i


def _int_value(levels: tuple[int, ...]) -> int:
    v = 0
    for lvl, i in enumerate(levels):
        v = (v << BITS[lvl]) | i
    return v


def check_symbol(levels: tuple[int, ...]) -> str:
    """Crockford mod-37 check symbol over the code's integer value (level indices)."""
    v = (_int_value(levels) << 3) | len(levels)
    return CHECK_ALPHABET[v % 37]


@dataclass(frozen=True, order=True)
class Code:
    """A parsed Sphota code: a tuple of per-level codebook indices (1..4 levels)."""

    levels: tuple[int, ...]
    version: str = "s1"

    def __post_init__(self) -> None:
        if not 1 <= len(self.levels) <= MAX_LEVELS:
            raise SphotaCodeError(f"a code has 1 to {MAX_LEVELS} levels, got {len(self.levels)}")
        for lvl, i in enumerate(self.levels):
            if not 0 <= int(i) < V1_SIZES[lvl]:
                raise SphotaCodeError(f"level {lvl + 1} index {i} out of range")

    # ---- formatting
    @property
    def depth(self) -> int:
        return len(self.levels)

    def format(self, *, version: bool = False, check: bool = False, hyphens: bool = True) -> str:
        parts = [index_to_symbols(lvl, i) for lvl, i in enumerate(self.levels)]
        s = ("-" if hyphens else "").join(parts)
        if version:
            s = f"{self.version}:{s}"
        if check:
            s = f"{s}.{check_symbol(self.levels)}"
        return s

    def __str__(self) -> str:
        return self.format()

    # ---- hierarchy
    def prefix(self, depth: int) -> Code:
        if not 1 <= depth <= self.depth:
            raise SphotaCodeError(f"prefix depth must be in 1..{self.depth}, got {depth}")
        return Code(self.levels[:depth], self.version)

    def is_prefix_of(self, other: Code | str) -> bool:
        o = parse(other) if isinstance(other, str) else other
        return o.levels[: self.depth] == self.levels

    def common_prefix(self, other: Code | str) -> int:
        """Number of leading levels shared with ``other``."""
        o = parse(other) if isinstance(other, str) else other
        n = 0
        for a, b in zip(self.levels, o.levels):
            if a != b:
                break
            n += 1
        return n


def parse(text: str | Code) -> Code:
    """Parse and validate a Sphota code string. Raises :class:`SphotaCodeError`."""
    if isinstance(text, Code):
        return text
    if not isinstance(text, str):
        raise SphotaCodeError(f"expected a code string like '7F3A-C2-91-0B', got {type(text).__name__}")
    raw = "".join(text.split()).upper()
    m = _RE.match(raw)
    if not m:
        raise SphotaCodeError(
            f"cannot parse {text!r}: expected something like '7F3A-C2-91-0B' "
            "(optionally 's1:' in front or '.X' check symbol at the end)")
    ver = (m.group("ver") or "S1").lower()
    if ver != "s1":
        raise SphotaCodeError(
            f"code version {ver!r} is not supported by this library (supports s1); "
            "upgrade sphota or use `sphota migrate`")
    body = m.group("body")
    if "-" in body:
        segs = body.split("-")
        if any(not s for s in segs):
            raise SphotaCodeError(f"empty segment in {text!r}; use the form XXXX-XX-XX-XX")
    else:
        segs, pos = [], 0
        for w in WIDTHS:
            if pos >= len(body):
                break
            segs.append(body[pos : pos + w])
            pos += w
        if pos < len(body):
            raise SphotaCodeError(f"{text!r} is too long: a full code has 10 symbols")
    if len(segs) > MAX_LEVELS:
        raise SphotaCodeError(f"{text!r} has {len(segs)} levels; Sphota v1 codes have at most 4")
    levels = []
    for lvl, seg in enumerate(segs):
        if len(seg) != WIDTHS[lvl]:
            raise SphotaCodeError(
                f"level {lvl + 1} of {text!r} should have {WIDTHS[lvl]} symbols, got {seg!r}; "
                "the layout is XXXX-XX-XX-XX")
        levels.append(symbols_to_index(lvl, seg))
    code = Code(tuple(levels), ver)
    chk = m.group("chk")
    if chk is not None and chk != check_symbol(code.levels):
        raise SphotaCodeError(f"check symbol mismatch in {text!r}: the code was mistyped")
    return code


def is_valid(text: str) -> bool:
    try:
        parse(text)
    except SphotaCodeError:
        return False
    return True


def from_indices(levels: list[int] | tuple[int, ...]) -> Code:
    return Code(tuple(int(i) for i in levels))
