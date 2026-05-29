from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path

from .common import discover_videos, project_root, run_command

SUBJECTIVE_COLUMNS = ["UAS", "IFS", "VRS", "SEM"]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_checklist(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Checklist must be a JSON list of task objects.")
    return data


def write_id_list(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in ids) + "\n", encoding="utf-8")


def copy_without_task_id(input_csv: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(input_csv, newline="", encoding="utf-8") as src, open(output_csv, "w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=SUBJECTIVE_COLUMNS)
        writer.writeheader()
        for row in reader:
            writer.writerow({column: row.get(column, "") for column in SUBJECTIVE_COLUMNS})


def materialize_checklist(
    checklist: list[dict],
    source_dir: Path | None,
    edited_dir: Path | None,
    limit: int,
) -> tuple[list[dict], list[int]]:
    source_map = dict(discover_videos(source_dir)) if source_dir else {}
    edited_map = dict(discover_videos(edited_dir)) if edited_dir else {}

    selected: list[dict] = []
    task_ids: list[int] = []
    for item in checklist:
        raw_id = item.get("id")
        try:
            task_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if source_map and task_id not in source_map:
            continue
        if edited_map and task_id not in edited_map:
            continue
        copied = dict(item)
        if source_map:
            copied["videoA_path"] = str(source_map[task_id].resolve())
        if edited_map:
            copied["videoB_path"] = str(edited_map[task_id].resolve())
        selected.append(copied)
        task_ids.append(task_id)
        if limit and len(selected) >= limit:
            break

    if not selected:
        raise SystemExit("No checklist tasks matched the requested source/edited videos.")
    return selected, task_ids


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run CoVEBench subjective MLLM-checklist metrics for one edited-video folder.")
    parser.add_argument("--checklist", required=True, help="Original checklist JSON.")
    parser.add_argument("--output-csv", required=True, help="Final user-facing CSV with UAS, IFS, VRS, SEM columns only.")
    parser.add_argument("--work-dir", default="", help="Intermediate checklist/log/cache directory.")
    parser.add_argument("--source-dir", help="Optional source-video directory named by task id.")
    parser.add_argument("--edited-dir", help="Optional edited-video directory named by task id.")
    parser.add_argument("--model-path", default="", help="Qwen video-MLLM path used by vLLM. Required unless --aggregate-only or --dry-run.")
    parser.add_argument("--prompt-dir", default=str(root / "metrics" / "subjective" / "mllm_checklist" / "prompts"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--allowed-media-dir", default="", help="Optional vLLM local media allowlist root. Defaults to inferred common video root.")
    parser.add_argument("--accuracy-scale", choices=["percent", "unit"], default="percent", help="CSV scale for UAS/IFS/VRS.")
    parser.add_argument("--sem-scale", choices=["raw", "percent"], default="raw", help="CSV scale for SEM.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true", help="Skip vLLM inference and aggregate an existing checklist_evaluated.json in work-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running inference/aggregation.")
    args = parser.parse_args()

    checklist_path = Path(args.checklist).resolve()
    output_csv = Path(args.output_csv).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else output_csv.with_suffix("").parent / f"{output_csv.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.source_dir).resolve() if args.source_dir else None
    edited_dir = Path(args.edited_dir).resolve() if args.edited_dir else None
    checklist, task_ids = materialize_checklist(load_checklist(checklist_path), source_dir, edited_dir, args.limit)

    work_checklist = work_dir / "checklist.json"
    id_list = work_dir / "task_ids.txt"
    write_json(work_checklist, checklist)
    write_id_list(id_list, task_ids)

    evaluated_json = work_dir / "checklist_evaluated.json"
    judge_log = work_dir / "judge.log"
    aggregate_log = work_dir / "aggregate.log"
    indexed_csv = output_csv.with_name(output_csv.stem + "_with_task_id.csv")

    if not args.aggregate_only:
        if not args.model_path and not args.dry_run:
            raise SystemExit("--model-path is required unless --aggregate-only or --dry-run is set.")
        judge_cmd = [
            sys.executable,
            str(root / "metrics" / "subjective" / "mllm_checklist" / "eval_checklist.py"),
            "--base-path",
            str(work_dir),
            "--model-path",
            args.model_path,
            "--prompt-dir",
            str(Path(args.prompt_dir).resolve()),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--batch-size",
            str(args.batch_size),
            "--max-retries",
            str(args.max_retries),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--id-list",
            str(id_list),
        ]
        if args.allowed_media_dir:
            judge_cmd.extend(["--allowed-media-dir", str(Path(args.allowed_media_dir).resolve())])
        if args.dry_run:
            print(" ".join(shlex.quote(x) for x in judge_cmd))
        else:
            run_command(judge_cmd, log_path=judge_log, cwd=root)

    aggregate_cmd = [
        sys.executable,
        str(root / "metrics" / "subjective" / "mllm_checklist" / "aggregate_subjective.py"),
        "--input",
        str(evaluated_json),
        "--output",
        str(indexed_csv),
        "--accuracy-scale",
        args.accuracy_scale,
        "--sem-scale",
        args.sem_scale,
        "--id-list",
        str(id_list),
    ]
    if args.dry_run:
        print(" ".join(shlex.quote(x) for x in aggregate_cmd))
        return

    run_command(aggregate_cmd, log_path=aggregate_log, cwd=root)
    copy_without_task_id(indexed_csv, output_csv)

    manifest = {
        "checklist": str(checklist_path),
        "source_dir": str(source_dir) if source_dir else "",
        "edited_dir": str(edited_dir) if edited_dir else "",
        "task_count": len(task_ids),
        "judge_log": str(judge_log),
        "aggregate_log": str(aggregate_log),
        "evaluated_json": str(evaluated_json),
        "output_csv": str(output_csv),
        "indexed_output_csv": str(indexed_csv),
    }
    write_json(work_dir / "run_manifest.json", manifest)
    print(f"Wrote {output_csv}")
    print(f"Wrote {indexed_csv} for sanity checking.")


if __name__ == "__main__":
    main()
