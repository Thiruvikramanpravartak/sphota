# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""The public, friendly API. Everything is one function call.

>>> from sphota import encode
>>> c = encode("The CEO resigned")          # doctest: +SKIP
>>> c                                       # doctest: +SKIP
Sphota('7F3A-C2-91-0B')
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sphota import distance as _dist
from sphota.codes import Code, from_indices, parse
from sphota.release import Release, load_release
from sphota.space import Quantized, complete_path, project, quantize, topk_codes

DEFAULT_LANG = "en"
MAX_ALTERNATES = 5


# ------------------------------------------------------------------ labels


def label_parts(
    code: Code | str, lang: str = DEFAULT_LANG, rel: Release | None = None
) -> list[str]:
    """Human-readable labels for the L1 and L2 prefixes of ``code``.

    Falls back to English when ``lang`` has no label; L2 labels exist only for cells that were
    populated in the training data (otherwise only the L1 label is returned).
    """
    rel = rel or load_release()
    c = parse(code)
    out = []
    keys = [f"{c.levels[0]}"]
    if c.depth >= 2:
        keys.append(f"{c.levels[0]}.{c.levels[1]}")
    for k in keys:
        entry = rel.labels.get(k)
        if not entry:
            continue
        text = entry.get(lang) or entry.get(DEFAULT_LANG)
        if text:
            out.append(text)
    return out


def label(code: Code | str, lang: str = DEFAULT_LANG) -> str:
    return " › ".join(label_parts(code, lang))


# ------------------------------------------------------------------ result object


class Sphota(str):
    """A Sphota code. It *is* a string (``'7F3A-C2-91-0B'``) with extra, lazy attributes.

    Attributes:
        code: the parsed :class:`~sphota.codes.Code`.
        confidence: cumulative per-level confidence (probability the prefix matches).
        level_confidence: per-level conditional confidence.
        text: the input text (if encoded from text).
    """

    code: Code
    text: str | None
    confidence: list[float]
    level_confidence: list[float]
    full_depth_code: Code | None
    _q: Quantized | None
    _row: int
    _vector: np.ndarray | None

    def __new__(cls, code: Code | str, **kw: Any) -> Sphota:
        c = parse(code)
        obj = super().__new__(cls, str(c))
        obj.code = c
        obj.text = kw.get("text")
        obj.confidence = list(kw.get("confidence") or [])
        obj.level_confidence = list(kw.get("level_confidence") or [])
        obj.full_depth_code = kw.get("full_depth_code")
        obj._q = kw.get("q")
        obj._row = int(kw.get("row", 0))
        obj._vector = kw.get("vector")
        return obj

    def __repr__(self) -> str:
        return f"Sphota('{self!s}')"

    def __reduce__(self) -> tuple[Any, ...]:
        return (Sphota, (str(self),))

    # ---- hierarchy
    @property
    def depth(self) -> int:
        return self.code.depth

    @property
    def levels(self) -> tuple[int, ...]:
        return self.code.levels

    def prefix(self, depth: int) -> Sphota:
        conf = self.confidence[:depth] if self.confidence else None
        return Sphota(
            self.code.prefix(depth),
            confidence=conf,
            level_confidence=self.level_confidence[:depth] or None,
        )

    def startswith_code(self, other: str) -> bool:
        return parse(other).is_prefix_of(self.code)

    def common_prefix(self, other: str) -> int:
        return self.code.common_prefix(parse(other))

    # ---- regions (v1.1): coarse cross-language / cross-model layer
    @property
    def region(self) -> str:
        """One of 32 coarse meaning regions ('R00'..'R31'); stable across languages and models."""
        from sphota.regions import region_of

        return region_of(self.vector)[0]

    @property
    def region_confidence(self) -> float:
        from sphota.regions import region_of

        return region_of(self.vector)[1]

    @property
    def region_label(self) -> str:
        from sphota.regions import region_label

        return region_label(self.region)

    def region_label_in(self, lang: str) -> str:
        from sphota.regions import region_label

        return region_label(self.region, lang)

    # ---- labels
    @property
    def label(self) -> str:
        return label(self.code)

    def label_in(self, lang: str) -> str:
        return label(self.code, lang)

    # ---- geometry
    @property
    def vector(self) -> np.ndarray:
        """Canonical vector of the input (if encoded) else the code's reconstruction."""
        if self._vector is not None:
            return self._vector
        return _dist.reconstruct(self.code)

    def similarity(self, other: str) -> float:
        return _dist.similarity(self.code, parse(other))

    def distance(self, other: str) -> float:
        return _dist.distance(self.code, parse(other))

    def neighbors(self, k: int = 10) -> list[Sphota]:
        return [Sphota(c) for c, _ in _dist.neighbors(self.code, k)]

    # ---- uncertainty
    @property
    def alternates(self) -> list[tuple[Sphota, float]]:
        """Other likely codes for the *first uncertain level* (with probabilities)."""
        if self._q is None:
            return []
        q, r = self._q, self._row
        out: list[tuple[Sphota, float]] = []
        for lvl in range(self.depth):
            cands, ps = q.cand[lvl][r], q.cand_p[lvl][r]
            if ps[0] >= 0.98 and lvl < self.depth - 1:
                continue
            prefix = [int(v) for v in q.indices[r, :lvl]]
            depth = self.depth
            rel = load_release()
            for c, p in zip(cands, ps):
                if p < 0.02:
                    break
                if self._vector is not None:
                    levels = complete_path(rel, self._vector, [*prefix, int(c)], depth)
                else:
                    levels = [*prefix, int(c)]
                cs = Sphota(from_indices(levels))
                if cs.code != self.code and all(cs.code != o.code for o, _ in out):
                    out.append((cs, float(p)))
                if len(out) == MAX_ALTERNATES:
                    break
            break
        return out

    @property
    def is_boundary(self) -> bool:
        """True if some level up to this depth is a near-tie (``level_confidence < 0.6``)."""
        return any(p < 0.6 for p in self.level_confidence[: self.depth])

    def candidates(self, k: int = 3) -> list[tuple[Sphota, float]]:
        """Top-k full codes with probabilities (multi-code for ambiguous text)."""
        if self._q is None:
            return [(self, 1.0)]
        return [
            (Sphota(from_indices(list(c))), float(p)) for c, p in topk_codes(self._q, self._row, k)
        ]

    def explain(self, lang: str = DEFAULT_LANG) -> dict[str, Any]:
        return explain(self, lang=lang)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": str(self),
            "label": self.label,
            "confidence": [round(x, 4) for x in self.confidence],
            "region": self.region,
            "region_label": self.region_label,
            "region_confidence": round(self.region_confidence, 4),
            "alternates": [{"code": str(c), "p": round(p, 4)} for c, p in self.alternates],
            **({"text": self.text} if self.text is not None else {}),
        }


# ------------------------------------------------------------------ encoding


def _to_codes(
    Y: np.ndarray,
    texts: Sequence[str | None],
    min_confidence: float | None,
    max_level: int,
    stable: bool,
    rel: Release,
    temp_scale: float = 1.0,
) -> list[Sphota]:
    if not 1 <= max_level <= len(rel.codebooks):
        raise ValueError(f"max_level must be between 1 and {len(rel.codebooks)}")
    q = quantize(rel, Y, levels=len(rel.codebooks), stable=stable, temp_scale=temp_scale)
    conf = q.confidence
    out = []
    for i in range(len(Y)):
        depth = max_level
        if min_confidence is not None:
            # cumulative confidence is non-increasing: cut at the first level below threshold;
            # always keep level 1 (a code has >= 1 level); its confidence tells the story
            below = conf[i, :max_level] < min_confidence
            depth = max(1, int(np.argmax(below)) if below.any() else max_level)
        full = from_indices(q.indices[i])
        out.append(
            Sphota(
                full.prefix(depth),
                text=texts[i],
                confidence=conf[i].tolist(),
                level_confidence=q.level_conf[i].tolist(),
                full_depth_code=full,
                q=q,
                row=i,
                vector=Y[i],
            )
        )
    return out


_ENCODERS: dict[str, Any] = {}


def _encoder_for(model: str | None):
    """(encoder, adapter) for ``model``; adapter is None for the anchor.

    Adapter models run through ONNX Runtime when the registry lists an ONNX export (no torch
    needed), otherwise through sentence-transformers.
    """
    from sphota.encoders import EncoderError, OnnxEncoder, TorchEncoder, default_encoder

    rel = load_release()
    if model is None or model in {rel.anchor["model"], "anchor", "default"}:
        return default_encoder(), None
    from sphota.adapters import get_adapter, registry_entry

    ad = get_adapter(model)
    if ad.model not in _ENCODERS:
        onnx = (registry_entry(model) or {}).get("onnx")
        enc = None
        if onnx:
            try:
                enc = OnnxEncoder(
                    repo=onnx["repo"],
                    onnx_file=onnx.get("file", "onnx/model.onnx"),
                    prefix=ad.prefix,
                    max_length=int(onnx.get("max_length", 512)),
                    pooling=onnx.get("pooling", "mean"),
                    name=ad.model,
                )
            except EncoderError:
                enc = None
        _ENCODERS[ad.model] = enc or TorchEncoder(ad.model, prefix=ad.prefix)
    return _ENCODERS[ad.model], ad


# Scripts where the anchor model fails (FLORES benchmark, D-021): codes look confident but are
# wrong. Encoding is unchanged (frozen behaviour); callers get a warning.
_UNSUPPORTED_SCRIPTS = {
    "Ol Chiki (Santali)": (0x1C50, 0x1C7F),
    "Meetei Mayek (Manipuri)": (0xABC0, 0xABFF),
}


def unsupported_script(text: str) -> str | None:
    """Name of an unsupported script that makes up most of ``text``'s letters, else ``None``."""
    letters = [ord(ch) for ch in text if ch.isalpha()]
    if not letters:
        return None
    for name, (lo, hi) in _UNSUPPORTED_SCRIPTS.items():
        if sum(lo <= o <= hi for o in letters) >= 0.5 * len(letters):
            return name
    return None


def _warn_unsupported(texts: list[str]) -> None:
    hits = {n for t in texts if (n := unsupported_script(t))}
    if hits:
        warnings.warn(
            f"Sphota codes are unreliable for {', '.join(sorted(hits))}: the anchor model does not "
            "understand this script, so codes can look confident and still be wrong (see "
            "docs/benchmark.md, Indian languages). Transliterate or translate the text first.",
            UserWarning,
            stacklevel=3,
        )


def encode_batch(
    texts: Iterable[str],
    *,
    min_confidence: float | None = None,
    max_level: int = 4,
    stable: bool = False,
    model: str | None = None,
) -> list[Sphota]:
    """Encode many texts (batched, cached). See :func:`encode` for parameters."""
    texts = list(texts)
    if not texts:
        return []
    if not all(isinstance(t, str) for t in texts):
        raise TypeError("encode_batch expects strings; for vectors use from_vector(vec, model=...)")
    _warn_unsupported(texts)
    rel = load_release()
    enc, ad = _encoder_for(model)
    X = enc.encode(texts)
    Y = project(rel, X) if ad is None else ad(X)
    ts = float(ad.meta.get("temperature_scale", 1.0)) if ad is not None else 1.0
    return _to_codes(Y, texts, min_confidence, max_level, stable, rel, ts)


def encode(
    text: str,
    *,
    min_confidence: float | None = None,
    max_level: int = 4,
    stable: bool = False,
    model: str | None = None,
) -> Sphota:
    """Encode text (any language) into a Sphota code.

    Args:
        text: the text.
        min_confidence: truncate the code where the cumulative confidence drops below this
            (GPS-style precision). ``None`` keeps all ``max_level`` levels.
        max_level: 1-4. Shorter codes reveal less (privacy by depth).
        stable: stable mode: near-boundary points pick the cell most robust to small
            perturbations (fewer flips across models/paraphrases).
        model: encode with another model through its certified adapter
            (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``). Default: the v1 anchor.
    """
    if not isinstance(text, str):
        raise TypeError(
            "encode expects a string; for many texts use encode_batch, for vectors use from_vector"
        )
    return encode_batch(
        [text], min_confidence=min_confidence, max_level=max_level, stable=stable, model=model
    )[0]


def from_vector(
    vec: np.ndarray | Sequence[float],
    *,
    model: str | None = None,
    min_confidence: float | None = None,
    max_level: int = 4,
    stable: bool = False,
) -> Sphota | list[Sphota]:
    """Code(s) for precomputed embedding(s). ``model`` names the model that produced them.

    A 1-d vector returns one :class:`Sphota`; a 2-d array returns a list.
    """
    rel = load_release()
    X = np.asarray(vec, dtype=np.float32)
    single = X.ndim == 1
    X = np.atleast_2d(X)
    ts = 1.0
    if model is None or model in {rel.anchor["model"], "anchor", "default"}:
        Y = project(rel, X)
    else:
        from sphota.adapters import get_adapter

        ad = get_adapter(model)
        Y = ad(X)
        ts = float(ad.meta.get("temperature_scale", 1.0))
    out = _to_codes(Y, [None] * len(Y), min_confidence, max_level, stable, rel, ts)
    return out[0] if single else out


# ------------------------------------------------------------------ code utilities


def prefix(code: str, depth: int) -> Sphota:
    return Sphota(parse(code).prefix(depth))


def similarity(a: str, b: str) -> float:
    """Similarity of two codes (or two texts, which are encoded first) in [0, 1]."""
    ca, cb = _as_code(a), _as_code(b)
    return _dist.similarity(ca, cb)


def distance(a: str, b: str) -> float:
    return _dist.distance(_as_code(a), _as_code(b))


def neighbors(code: str, k: int = 10) -> list[Sphota]:
    return [Sphota(c) for c, _ in _dist.neighbors(_as_code(code), k)]


def path(a: str, b: str) -> list[Sphota]:
    """Semantic route between two codes (or texts) through neighbouring regions."""
    return [Sphota(c) for c in _dist.path(_as_code(a), _as_code(b))]


def _as_code(x: str | Code) -> Code:
    if isinstance(x, Sphota):
        return x.code
    if isinstance(x, Code):
        return x
    from sphota.codes import is_valid

    if is_valid(x):
        return parse(x)
    return encode(x).code


def explain(code: str, lang: str = DEFAULT_LANG) -> dict[str, Any]:
    """Decode a code: per-level labels, confidence (if known), neighbours."""
    s = code if isinstance(code, Sphota) else Sphota(_as_code(code))
    levels = []
    for d in range(1, s.depth + 1):
        p = s.code.prefix(d)
        item: dict[str, Any] = {"prefix": str(p), "level": d}
        if d <= 2:
            parts = label_parts(p, lang)
            item["label"] = parts[d - 1] if len(parts) >= d else None
        if s.confidence:
            item["confidence"] = round(s.confidence[d - 1], 4)
        levels.append(item)
    return {
        "code": str(s),
        "label": label(s.code, lang),
        "levels": levels,
        "neighbors": [{"code": str(c), "label": label(c, lang)} for c in neighbors(s.prefix(1), 5)],
    }


# ------------------------------------------------------------------ documents


_SENT = re.compile(r"(?<=[.!?。！？।])\s+|(?<=[。！？])|\n+")


def sentences(doc: str, min_chars: int = 12) -> list[str]:
    out = [s.strip() for s in _SENT.split(doc) if s and s.strip()]
    return [s for s in out if len(s) >= min_chars] or ([doc.strip()] if doc.strip() else [])


@dataclass
class Fingerprint:
    """A document's meaning fingerprint: weighted histogram of code prefixes."""

    level: int
    weights: dict[str, float]  # prefix -> weight (sums to 1)
    n_sentences: int
    codes: list[Sphota] = field(default_factory=list, repr=False)

    def top(self, n: int = 5, lang: str = DEFAULT_LANG) -> list[tuple[str, float, str]]:
        items = sorted(self.weights.items(), key=lambda kv: -kv[1])[:n]
        return [(k, w, label(k, lang)) for k, w in items]

    def at_level(self, level: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, w in self.weights.items():
            p = str(parse(k).prefix(min(level, parse(k).depth)))
            out[p] = out.get(p, 0.0) + w
        return out

    def __repr__(self) -> str:
        tops = ", ".join(f"{k} {w:.0%}" for k, w, _ in self.top(3))
        return f"Fingerprint(level={self.level}, n_sentences={self.n_sentences}, top=[{tops}])"


def fingerprint(doc: str | Sequence[str], level: int = 2, model: str | None = None) -> Fingerprint:
    """Meaning fingerprint of a document (a string, or a list of sentences/chunks)."""
    sents = sentences(doc) if isinstance(doc, str) else [s for s in doc if s.strip()]
    if not sents:
        return Fingerprint(level, {}, 0)
    codes = encode_batch(sents, model=model)
    w: dict[str, float] = {}
    for s, c in zip(sents, codes):
        # weight: longer sentences say more; unsure levels count less
        wt = np.log1p(len(s)) * c.confidence[0] ** 0.5
        k = str(c.code.prefix(level))
        w[k] = w.get(k, 0.0) + float(wt)
    tot = sum(w.values()) or 1.0
    return Fingerprint(level, {k: v / tot for k, v in w.items()}, len(sents), codes)


def _fp(x: str | Sequence[str] | Fingerprint, level: int) -> Fingerprint:
    return x if isinstance(x, Fingerprint) else fingerprint(x, level=level)


def compare(doc_a: str | Fingerprint, doc_b: str | Fingerprint, level: int = 2) -> float:
    """Similarity of two documents' meaning fingerprints in [0, 1].

    Average of the cosine similarity of the L1 histograms and of the ``level`` histograms, so
    documents on related-but-not-identical topics score in between.
    """
    fa, fb = _fp(doc_a, level), _fp(doc_b, level)

    def cos(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) | set(b)
        va = np.array([a.get(k, 0.0) for k in keys])
        vb = np.array([b.get(k, 0.0) for k in keys])
        den = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(va @ vb / den) if den else 0.0

    return 0.5 * cos(fa.at_level(1), fb.at_level(1)) + 0.5 * cos(
        fa.at_level(level), fb.at_level(level)
    )


@dataclass
class FingerprintDiff:
    added: list[tuple[str, float, str]]
    removed: list[tuple[str, float, str]]
    shared: list[tuple[str, float, str]]
    similarity: float

    def __repr__(self) -> str:
        return (
            f"FingerprintDiff(similarity={self.similarity:.2f}, added={len(self.added)}, "
            f"removed={len(self.removed)}, shared={len(self.shared)})"
        )


def diff(
    doc_a: str | Fingerprint,
    doc_b: str | Fingerprint,
    level: int = 2,
    threshold: float = 0.05,
    lang: str = DEFAULT_LANG,
) -> FingerprintDiff:
    """Which topics were added in ``doc_b`` / removed from ``doc_a`` (weight change > threshold)."""
    fa, fb = _fp(doc_a, level), _fp(doc_b, level)
    added, removed, shared = [], [], []
    for k in sorted(set(fa.weights) | set(fb.weights)):
        wa, wb = fa.weights.get(k, 0.0), fb.weights.get(k, 0.0)
        item = (k, round(wb - wa, 4), label(k, lang))
        if wb - wa > threshold:
            added.append(item)
        elif wa - wb > threshold:
            removed.append(item)
        elif wa > 0 and wb > 0:
            shared.append(item)
    added.sort(key=lambda t: -t[1])
    removed.sort(key=lambda t: t[1])
    return FingerprintDiff(added, removed, shared, compare(fa, fb, level))
