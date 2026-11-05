# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Prefix router: route texts (tickets, prompts, queries) by the longest matching code prefix.

Rules can be written by hand (``{"JYZA": "finance-team", "JYZA-7M": "exec-comms"}``) or
learned from a few labelled examples with :meth:`PrefixRouter.fit` (majority route per L1 and
L2 prefix, with a neighbour fallback for unseen L1 regions). Because codes are language- and
model-independent, rules written once apply to every language and every certified model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from sphota.api import Sphota, encode_batch
from sphota.codes import parse
from sphota.distance import neighbors


@dataclass
class Route:
    route: str
    prefix: str | None
    code: str
    confidence: float

    def __repr__(self) -> str:
        return f"Route({self.route!r}, via={self.prefix}, code={self.code}, conf={self.confidence:.2f})"


@dataclass
class PrefixRouter:
    rules: dict[str, str] = field(default_factory=dict)
    default: str = "default"
    model: str | None = None
    neighbor_fallback: int = 5

    def add_rule(self, prefix: str, route: str) -> None:
        self.rules[str(parse(prefix))] = route

    def fit(
        self, texts: Sequence[str], routes: Sequence[str], min_support: int = 1
    ) -> PrefixRouter:
        codes = encode_batch(list(texts), model=self.model)
        votes: dict[str, Counter] = defaultdict(Counter)
        for c, r in zip(codes, routes):
            votes[str(c.code.prefix(1))][r] += 1
            if c.depth >= 2:
                votes[str(c.code.prefix(2))][r] += 1
        for p, cnt in votes.items():
            route, n = cnt.most_common(1)[0]
            if n >= min_support:
                parent = p.split("-")[0]
                if "-" in p and self.rules.get(parent) == route:
                    continue  # redundant with the L1 rule
                self.rules[p] = route
        return self

    def route_code(self, c: Sphota) -> Route:
        for d in range(c.depth, 0, -1):
            p = str(c.code.prefix(d))
            if p in self.rules:
                return Route(self.rules[p], p, str(c), c.confidence[d - 1] if c.confidence else 1.0)
        if self.neighbor_fallback:
            for nb, _ in neighbors(c.code.prefix(1), self.neighbor_fallback):
                p = str(nb)
                if p in self.rules:
                    return Route(
                        self.rules[p],
                        p + "~",
                        str(c),
                        0.5 * (c.confidence[0] if c.confidence else 1.0),
                    )
        return Route(self.default, None, str(c), 0.0)

    def route(self, text: str) -> Route:
        return self.route_code(encode_batch([text], model=self.model)[0])

    def route_batch(self, texts: Sequence[str]) -> list[Route]:
        return [self.route_code(c) for c in encode_batch(list(texts), model=self.model)]
