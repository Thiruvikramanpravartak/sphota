"""Route tickets by code prefix: learn rules from a few labelled examples, in any language."""

import csv
from pathlib import Path

from sphota.usecases import PrefixRouter

rows = list(
    csv.DictReader(open(Path(__file__).parent / "data" / "headlines.csv", encoding="utf-8"))
)
train = [r for r in rows if r["lang"] == "en"]
test = [r for r in rows if r["lang"] != "en"]

router = PrefixRouter(default="triage").fit([r["text"] for r in train], [r["topic"] for r in train])
print(f"learned {len(router.rules)} prefix rules from {len(train)} English examples")
correct = 0
for r in test:
    route = router.route(r["text"])
    correct += route.route == r["topic"]
print(f"routed {len(test)} non-English texts; correct topic: {correct}/{len(test)}")
