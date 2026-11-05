# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Ready-made use cases built on Sphota codes."""

from sphota.usecases.analytics import (
    analyze_file,
    build_report,
    group_by_prefix,
    prefix_counts,
    render_html,
)
from sphota.usecases.cache import SemanticCache
from sphota.usecases.documents import compare_documents
from sphota.usecases.index import CodeIndex, Hit, cross_lingual_search, dedupe
from sphota.usecases.router import PrefixRouter, Route

__all__ = [
    "CodeIndex",
    "Hit",
    "PrefixRouter",
    "Route",
    "SemanticCache",
    "analyze_file",
    "build_report",
    "compare_documents",
    "cross_lingual_search",
    "dedupe",
    "group_by_prefix",
    "prefix_counts",
    "render_html",
]
