# Sphota Specification v1, Part 1: Code Format

```
Status:     Frozen (space version s1)
Editors:    The Sphota Authors
Normative:  Yes. Key words MUST, SHOULD, MAY are per RFC 2119 / RFC 8174.
```

## 1. Overview

A **Sphota code** names a region of the Sphota v1 canonical meaning space. It is a sequence
of 1 to 4 **levels**; each level is an index into that level's codebook. Shorter codes
(prefixes) name larger regions that contain all longer codes starting with them.

```
7F3A-C2-91-0B
└─┬┘ └┬┘ └┬┘ └┬┘
 L1  L2  L3  L4      L1 = broad region (4096 cells)  …  L4 = fine cell
```

## 2. Alphabet

Codes use the **Crockford base32** alphabet:

```
0123456789ABCDEFGHJKMNPQRSTVWXYZ      (value 0..31; no I, L, O, U)
```

* Encoders MUST emit upper-case symbols from this alphabet.
* Decoders MUST accept lower case, and MUST map `O`/`o` → `0` and `I`/`i`/`L`/`l` → `1`.
* Decoders MUST ignore whitespace anywhere in the input.

## 3. Levels, widths and symbol values

### 3.1 Widths

| Level | Symbols | Bits | v1 codebook size `K` | Valid fraction of symbol space |
|---|---|---|---|---|
| L1 | 4 | 20 | 4096 | 0.39 % |
| L2 | 2 | 10 | 256 | 25 % |
| L3 | 2 | 10 | 256 | 25 % |
| L4 | 2 | 10 | 256 | 25 % |

The widths are fixed by the format; the codebook sizes are fixed by the space version.

### 3.2 Symbol value

The symbols of a level are read as a big-endian base-32 integer `s` (`0 ≤ s < 2^bits`).

### 3.3 Index permutation

The codebook index `i` (`0 ≤ i < K`) of level `l` is written as

```
s = (a_l · i + b_l) mod 2^bits_l
i = ((s − b_l) · a_l⁻¹) mod 2^bits_l        (a_l is odd, so a_l⁻¹ exists mod 2^bits_l)
```

with the frozen v1 constants

| Level | `a_l` | `b_l` |
|---|---|---|
| L1 | `0x5BD1F` | `0x3A6E9` |
| L2 | `0x2A5` | `0x15D` |
| L3 | `0x1B3` | `0x0C9` |
| L4 | `0x3C7` | `0x2E1` |

A decoded index `i ≥ K` is **invalid** and the decoder MUST reject the code.

*Rationale.* The permutation spreads indices over the whole symbol space, so codes look
varied rather than all starting with `0`, and the sparse valid set detects most typos
without a check symbol (measured: 85 % of random single-symbol substitutions are rejected).
Symbols *within* a level carry no ordering semantics; meaning lives at level granularity.

## 4. Textual forms

### 4.1 Canonical display form

Levels are joined by hyphen-minus (`-`), e.g. `7F3A-C2-91-0B` (full), `7F3A-C2` (L2 prefix),
`7F3A` (L1 prefix). Encoders MUST produce this form unless asked otherwise.

### 4.2 Compact form

Without hyphens, e.g. `7F3AC2910B`. Since widths are fixed, decoders MUST accept compact
forms of length 4, 6, 8 or 10 symbols.

### 4.3 Version tag

An optional space-version tag precedes the code, separated by a colon:
`s1:7F3A-C2-91-0B`. Decoders MUST treat a missing tag as `s1`. A decoder that does not
support the tagged version MUST reject the code with an error that names the version.
Systems that store codes long-term SHOULD store the tag (or record the space version in
their schema).

### 4.4 Check symbol

An optional check symbol follows a full stop: `7F3A-C2-91-0B.K`.

```
v = (i_1 ‖ i_2 ‖ … ‖ i_n) as an integer, concatenating each index in its level's bit width
c = ((v << 3) | n) mod 37
check = "0123456789ABCDEFGHJKMNPQRSTVWXYZ*~$=U"[c]
```

The depth `n` is folded in so that a truncated code never carries a valid check symbol of
the longer code. Decoders MUST verify a present check symbol.

### 4.5 Grammar (ABNF, RFC 5234)

```
sphota-code  = [ version ":" ] body [ "." check ]
version      = ("s" / "S") 1*DIGIT
body         = l1 [ "-" l2 [ "-" l3 [ "-" l4 ] ] ] / compact
l1           = 4sym
l2 = 2sym ; l3 = 2sym ; l4 = 2sym
compact      = 4sym / 6sym / 8sym / 10sym
sym          = DIGIT / %x41-48 / %x4A-4B / %x4D-4E / %x50-54 / %x56-5A   ; A-H J K M N P-T V-Z
check        = sym / "*" / "~" / "$" / "=" / "U"
```

(Decoders additionally accept the lower-case and alias forms of §2.)

## 5. Comparison and storage

* Two codes are **equal** iff their level indices are equal (textual form is irrelevant).
* Code `p` **is a prefix of** `c` iff `p`'s levels equal the first `depth(p)` levels of `c`.
* Storing the canonical display form as text makes SQL prefix queries trivial and
  index-friendly: `code LIKE '7F3A-C2%'` (B-tree with `text_pattern_ops` in PostgreSQL). See
  `integrations/postgres/`.
* Lexicographic order of display strings is a valid total order but has **no semantic
  meaning** beyond grouping by prefix.

## 6. Test vectors

| Indices | Display | With tag and check |
|---|---|---|
| (0,0,0,0) | `79Q9-AX-69-Q1` | `s1:79Q9-AX-69-Q1.4` |
| (123,45,6,255) | `B16E-2Y-QV-GT` | `s1:B16E-2Y-QV-GT.B` |

Implementations MUST round-trip all test vectors in `tests/vectors/codes.json`.
