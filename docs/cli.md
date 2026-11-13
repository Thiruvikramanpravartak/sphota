# Command line

`pip install "sphota[encode,cli]"`, then:

| Command | What it does |
|---|---|
| `sphota encode "text" ["text" ...] [-c 0.6] [-l 2] [--stable] [-m MODEL] [-a] [--top-k 3] [--json]` | encode (use `-` to read lines from stdin) |
| `sphota explain CODE_OR_TEXT [--lang hi]` | labels per level, nearby regions |
| `sphota similarity A B` | similarity of two codes or texts |
| `sphota neighbors CODE [-k 10]` | nearest codes |
| `sphota path A B` | semantic route between two codes/texts |
| `sphota validate CODE` | check a code (exit 1 if invalid) |
| `sphota fingerprint FILE [--json]` | document fingerprint |
| `sphota compare FILE_A FILE_B` / `sphota diff FILE_A FILE_B` | document comparison |
| `sphota analyze data.csv [-o report.html] [--csv rows.csv] [--min-count 10] [--by auto\|refined\|theme\|region\|l1]` | topic analytics report; `auto` (default) fits groups to the file (≥ 200 messages, ids like `WJX1.2`) or uses themes for small files; `--csv` writes one row per message; `--detail fine\|medium\|coarse` or `--groups N` sets how many groups |
| `sphota adapters` | certified adapters |
| `sphota train-adapter --model HF_ID` | fit an adapter |
| `sphota certify --model HF_ID [--adapter f.npz] [--out cert.json]` | conformance certificate |
| `sphota migrate --build --from DIR --to DIR` / `sphota migrate codes.csv --table t.json` | version migration |
| `sphota bundle -o DIR` | offline bundle |
| `sphota info` | installed space version and settings |

Environment: `SPHOTA_OFFLINE=1`, `SPHOTA_BUNDLE=dir`, `SPHOTA_ENCODER=fp32|int8|torch`,
`SPHOTA_RELEASE_DIR=dir`.
