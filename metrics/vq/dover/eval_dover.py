#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from covebench_eval.common import load_id_list, task_id_from_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="TQ: DOVER++ technical score.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repo", default=str(ROOT / "external" / "T2AV-Compass" / "t2av-compass" / "Objective" / "Video" / "DOVER"))
    parser.add_argument("--weight", default=str(ROOT / "weights" / "dover" / "DOVER_plus_plus.pth"))
    parser.add_argument("--id-list")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch
    import yaml

    repo = Path(args.repo)
    weight = Path(args.weight)
    if not (repo / "evaluate_a_set_of_videos.py").exists():
        raise SystemExit("DOVER repo not found. Run: uv run scripts/download_weights.py")
    if not weight.exists():
        raise SystemExit("DOVER++ weight not found. Run: uv run scripts/download_weights.py")

    state_dict_path = weight.with_name(weight.stem + "_state_dict.pth")
    if not state_dict_path.exists():
        ckpt = torch.load(weight, map_location="cpu")
        torch.save(ckpt["state_dict"], state_dict_path)

    cfg = yaml.safe_load((repo / "dover.yml").read_text())
    cfg["test_load_path"] = str(state_dict_path)
    tmp_cfg = repo / f"dover_batch_{Path(args.video_dir).name}.yml"
    tmp_raw = Path(args.output).with_suffix(".raw.csv")
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    tmp_raw.parent.mkdir(parents=True, exist_ok=True)

    gpu = args.device.split(":", 1)[1] if args.device.startswith("cuda:") else "0"
    cmd = (
        f"cd {repo} && CUDA_VISIBLE_DEVICES={gpu} "
        f"'{sys.executable}' evaluate_a_set_of_videos.py "
        f"-o {tmp_cfg.name} -in '{Path(args.video_dir).resolve()}' -out '{tmp_raw.resolve()}'"
    )
    subprocess.run(["bash", "-lc", cmd], check=True)

    with open(tmp_raw, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    id_filter = load_id_list(args.id_list)
    written = 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "score", "error"])
        writer.writeheader()
        for row in rows:
            path = Path((row.get("path") or row.get("video") or "").strip())
            task_id = task_id_from_path(path)
            if task_id is None:
                continue
            if id_filter is not None and task_id not in id_filter:
                continue
            if args.limit and written >= args.limit:
                break
            score = ""
            error = ""
            try:
                score = f"{float(row.get(' technical score') or row.get('technical score') or row.get('technical_score')):.8f}"
            except Exception as exc:
                error = f"{type(exc).__name__}: cannot parse technical score"
            writer.writerow({"task_id": task_id, "score": score, "error": error})
            written += 1


if __name__ == "__main__":
    main()
