# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Sphota (स्फोट): readable, multilingual meaning codes.

Any text, in any language, maps to a short, frozen, hierarchical code plus one of 32 broad
regions, for topic analytics, routing, caching and model migration. The coarse levels agree
across languages most of the time; finer levels are precision within one model.

>>> from sphota import encode                # doctest: +SKIP
>>> c = encode("The CEO resigned")            # doctest: +SKIP
>>> c, c.label, c.confidence                  # doctest: +SKIP

The light core (``parse``, ``similarity`` on codes, ``neighbors``, labels) needs only numpy.
Encoding text needs ``pip install 'sphota[encode]'`` (ONNX Runtime, no torch).
"""

from sphota.api import (
    Fingerprint,
    FingerprintDiff,
    Sphota,
    compare,
    diff,
    distance,
    encode,
    encode_batch,
    explain,
    fingerprint,
    from_vector,
    label,
    neighbors,
    path,
    prefix,
    similarity,
    unsupported_script,
)
from sphota.codes import Code, SphotaCodeError, is_valid, parse

__version__ = "1.0.0rc1"
SPACE_VERSION = "s1"

__all__ = [
    "SPACE_VERSION",
    "Code",
    "Fingerprint",
    "FingerprintDiff",
    "Sphota",
    "SphotaCodeError",
    "__version__",
    "compare",
    "diff",
    "distance",
    "encode",
    "encode_batch",
    "explain",
    "fingerprint",
    "from_vector",
    "is_valid",
    "label",
    "neighbors",
    "parse",
    "path",
    "prefix",
    "similarity",
    "unsupported_script",
]
