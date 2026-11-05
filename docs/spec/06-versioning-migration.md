# Sphota Specification v1, Part 6: Versioning and Migration

```
Status: Frozen (s1)
```

## 1. Two version numbers

* **Space version** (`s1`, `s2`, …): identifies projection + codebooks + quantization rule.
  It is the only thing that determines what a code *means*.
* **Library version** (SemVer, e.g. `1.0.0`): the Python package. Any library version that
  supports `s1` MUST produce identical s1 codes (golden tests).

## 2. Frozen forever

A published space version is immutable. Its artifacts are content-addressed by the SHA-256
values in `manifest.json`; implementations MUST verify them before use. Bugs in a space are
fixed by publishing a new space version, never by editing an old one.

## 3. Migration tables

A new space version `sN+1` ships a migration table from `sN`:

```json
{
  "from": "s1", "to": "s2",
  "level": 2,
  "map": { "<s1 L2 prefix>": [["<s2 L2 prefix>", 0.83], ["<s2 L2 prefix>", 0.17]] }
}
```

Built by `sphota migrate --build --from <dir> --to <dir>`: encode a reference corpus in both
versions and record, for every source prefix (at levels 1 and 2), the distribution of target
prefixes. Consumers can

* re-encode raw text (best, if text is available),
* re-map vectors through the new space (if anchor vectors were stored), or
* map codes through the table (`sphota migrate codes.csv --table s1-s2.json`), taking the
  most probable target and keeping the probability as confidence.

## 4. Stored-data recommendation

Store `space_version` with every code (column or tag `s1:`). Store the anchor vector too if
you can afford it; codes are the portable index, vectors the lossless backup.
