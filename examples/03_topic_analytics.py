"""Topic analytics: turn a CSV into a standalone HTML report (also: `sphota analyze file.csv`)."""

from pathlib import Path

from sphota.usecases import analyze_file

here = Path(__file__).parent
out = here / "headlines-report.html"
res = analyze_file(here / "data" / "headlines.csv", out=out)
print(f"{res['n']} texts → {res['n_l1']} topics → {out}")
for prefix, count, label in res["report"].top(5):
    print(f"{prefix}  {count:3d}  {label}")
