from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path

from .common import discover_videos, project_root, read_metric_csv, run_command

METRICS = [
    ("AES", "metrics/vq/aes/eval_aes.py", "vq_aes.csv", ["--video-dir", "{edited}", "--output", "{out}", "--device", "{device}", "--frames", "{frames}"]),
    ("TQ", "metrics/vq/dover/eval_dover.py", "vq_tq.csv", ["--video-dir", "{edited}", "--output", "{out}", "--device", "{device}"]),
    ("MSM", "metrics/vq/msm/eval_msm.py", "vq_msm.csv", ["--video-dir", "{edited}", "--output", "{out}", "--frames", "{frames}"]),
    ("SSIM", "metrics/vf/ssim/eval_ssim.py", "vf_ssim.csv", ["--source-dir", "{source}", "--video-dir", "{edited}", "--output", "{out}", "--frames", "{frames}"]),
    ("MF", "metrics/vf/cotracker/eval_cotracker.py", "vf_mf.csv", ["--source-dir", "{source}", "--video-dir", "{edited}", "--output", "{out}", "--device", "{device}"]),
    ("VQR", "metrics/vq/vqr/eval_vqr.py", "vq_vqr.csv", ["--video-dir", "{edited}", "--output", "{out}", "--frames", "{frames}"]),
    ("SRC", "metrics/vf/src/eval_src.py", "vf_src.csv", ["--checklist", "{checklist}", "--source-dir", "{source}", "--video-dir", "{edited}", "--output", "{out}", "--work-dir", "{work}/src_cache", "--source-mask-root", "{source_mask_root}", "--device", "{device}"]),
]

DEFAULT_COLUMNS = ["AES", "TQ", "MSM", "SSIM", "MF", "VQR", "SRC"]


def expand_args(template: list[str], values: dict[str, str]) -> list[str]:
    return [item.format(**values) for item in template]


def write_id_list(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in ids) + "\n", encoding="utf-8")


def load_scores(work_dir: Path) -> dict[str, dict[int, str]]:
    return {
        "AES": read_metric_csv(work_dir / "vq_aes.csv", "score"),
        "TQ": read_metric_csv(work_dir / "vq_tq.csv", "score"),
        "MSM": read_metric_csv(work_dir / "vq_msm.csv", "score"),
        "SSIM": read_metric_csv(work_dir / "vf_ssim.csv", "score"),
        "MF": read_metric_csv(work_dir / "vf_mf.csv", "score"),
        "VQR": read_metric_csv(work_dir / "vq_vqr.csv", "score"),
        "SRC": read_metric_csv(work_dir / "vf_src.csv", "score"),
    }


def write_final_csv(output_csv: Path, task_ids: list[int], scores: dict[str, dict[int, str]], columns: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for task_id in task_ids:
            writer.writerow({column: scores[column].get(task_id, "") for column in columns})


def write_index_csv(output_csv: Path, task_ids: list[int], scores: dict[str, dict[int, str]], columns: list[str]) -> None:
    indexed_columns = ["task_id", *columns]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=indexed_columns)
        writer.writeheader()
        for task_id in task_ids:
            row = {"task_id": task_id}
            row.update({column: scores[column].get(task_id, "") for column in columns})
            writer.writerow(row)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run objective CoVEBench metrics for one edited-video folder.")
    parser.add_argument("--edited-dir", required=True, help="Directory containing edited videos named by task id.")
    parser.add_argument("--source-dir", required=True, help="Directory containing source videos named by the same task ids.")
    parser.add_argument("--output-csv", required=True, help="Final user-facing CSV with metric score columns only.")
    parser.add_argument("--work-dir", default="", help="Intermediate metric CSV/log/cache directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=10, help="Equal-spaced frame count for frame/image metrics.")
    parser.add_argument("--metrics", default=",".join(DEFAULT_COLUMNS), help="Comma-separated subset for debugging.")
    parser.add_argument("--checklist", default=str(root / "data" / "checklist.json"), help="Checklist JSON required by SRC.")
    parser.add_argument("--source-mask-root", default="", help="Optional source-side SRC mask cache for SRC.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    edited = Path(args.edited_dir).resolve()
    source = Path(args.source_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else output_csv.with_suffix("").parent / f"{output_csv.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    edited_ids = {tid for tid, _ in discover_videos(edited)}
    source_ids = {tid for tid, _ in discover_videos(source)}
    task_ids = sorted(edited_ids & source_ids)
    if args.limit:
        task_ids = task_ids[: args.limit]
    if not task_ids:
        raise SystemExit("No matching task ids between source_dir and edited_dir.")
    missing_sources = sorted(edited_ids - source_ids)
    missing_edits = sorted(source_ids - edited_ids)
    if missing_sources:
        print(f"WARNING: {len(missing_sources)} edited videos have no source match. Example: {missing_sources[:10]}", file=sys.stderr)
    if missing_edits:
        print(f"WARNING: {len(missing_edits)} source videos have no edited match. Example: {missing_edits[:10]}", file=sys.stderr)

    id_list = work_dir / "task_ids.txt"
    write_id_list(id_list, task_ids)
    requested = {x.strip().upper() for x in args.metrics.split(",") if x.strip()}
    unknown = requested - set(DEFAULT_COLUMNS)
    if unknown:
        raise SystemExit(f"Unknown metric(s): {', '.join(sorted(unknown))}. Choose from {', '.join(DEFAULT_COLUMNS)}")
    values = {
        "edited": str(edited),
        "source": str(source),
        "device": args.device,
        "frames": str(args.frames),
        "ids": str(id_list),
        "checklist": str(Path(args.checklist).resolve()),
        "work": str(work_dir),
        "source_mask_root": str(Path(args.source_mask_root).resolve()),
    }
    manifest = {"edited_dir": str(edited), "source_dir": str(source), "task_count": len(task_ids), "metrics": []}
    for metric_name, script, out_name, arg_template in METRICS:
        if metric_name not in requested:
            continue
        out_path = work_dir / out_name
        log_path = work_dir / f"{metric_name.lower()}.log"
        metric_values = {**values, "out": str(out_path)}
        cmd = [sys.executable, str(root / script), *expand_args(arg_template, metric_values), "--id-list", str(id_list)]
        manifest["metrics"].append({"metric": metric_name, "cmd": " ".join(shlex.quote(x) for x in cmd), "output": str(out_path), "log": str(log_path)})
        if args.dry_run:
            print(" ".join(shlex.quote(x) for x in cmd))
        else:
            print(f"Running {metric_name} -> {out_path}", flush=True)
            run_command(cmd, log_path=log_path, cwd=root)

    (work_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    scores = load_scores(work_dir)
    final_columns = [column for column in DEFAULT_COLUMNS if column in requested]
    write_final_csv(output_csv, task_ids, scores, final_columns)
    write_index_csv(output_csv.with_name(output_csv.stem + "_with_task_id.csv"), task_ids, scores, final_columns)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_csv.with_name(output_csv.stem + '_with_task_id.csv')} for sanity checking.")


if __name__ == "__main__":
    main()
