from __future__ import annotations

import argparse
from pathlib import Path

from .common import project_root

DEFAULT_DATASET_REPO = "NJU-LINK/CoVEBench"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the CoVEBench Hugging Face dataset.")
    parser.add_argument("--repo", default=DEFAULT_DATASET_REPO, help=f"Hugging Face dataset repo. Default: {DEFAULT_DATASET_REPO}")
    parser.add_argument(
        "--local-dir",
        default=str(project_root() / "data" / "covebench_hf"),
        help="Download destination. The default keeps checklist.json and data/ under data/covebench_hf.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--hf-endpoint", default="", help="Optional HF mirror endpoint, e.g. https://hf-mirror.com.")
    args = parser.parse_args()

    if args.hf_endpoint:
        import os

        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Run `uv sync` first.") from exc

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_dir),
    )
    print(f"Downloaded {args.repo} to {local_dir}")
    print(f"Checklist: {local_dir / 'checklist.json'}")
    print(f"Source videos: {local_dir / 'data'}")


if __name__ == "__main__":
    main()
