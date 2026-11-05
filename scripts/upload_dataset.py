"""Upload the tagged Wikipedia sample to the Hugging Face Hub. NOT run automatically.

huggingface-cli login
python -m sphota.build.open_data --scale small
python scripts/upload_dataset.py --repo <user>/sphota-wiki-s1 --scale small
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--scale", default="small")
ap.add_argument("--private", action="store_true")
a = ap.parse_args()
api = HfApi()
api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
data = ROOT / "data" / a.scale / "open" / f"sphota-wiki-s1-{a.scale}.parquet"
api.upload_file(
    path_or_fileobj=str(data), path_in_repo=f"data/{data.name}", repo_id=a.repo, repo_type="dataset"
)
api.upload_file(
    path_or_fileobj=str(ROOT / "datasets" / "sphota-wiki" / "README.md"),
    path_in_repo="README.md",
    repo_id=a.repo,
    repo_type="dataset",
)
print(f"https://huggingface.co/datasets/{a.repo}")
