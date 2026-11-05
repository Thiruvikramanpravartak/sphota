"""Meaning fingerprints for long documents: compare and diff topics."""

from sphota import compare, diff, fingerprint

v1 = (
    "The company reported strong quarterly earnings. Revenue grew in every region. "
    "The board approved a new dividend. Heavy rain disrupted deliveries in March."
)
v2 = (
    "The company reported strong quarterly earnings. Revenue grew in every region. "
    "The chief executive resigned unexpectedly. A new vaccine trial began at the company's lab."
)

fp = fingerprint(v1)
print(fp)
for prefix, weight, label in fp.top(3):
    print(f"  {prefix}  {weight:.0%}  {label}")
print(f"similarity v1 vs v2: {compare(v1, v2):.2f}")
d = diff(v1, v2)
for prefix, change, label in d.added:
    print(f"  + {prefix} {change:+.0%} {label}")
for prefix, change, label in d.removed:
    print(f"  - {prefix} {change:+.0%} {label}")
