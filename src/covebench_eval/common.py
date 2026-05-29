from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base or Path.cwd()) / p


def task_id_from_path(path: Path) -> int | None:
    stem = path.stem
    patterns = [
        r"task_(\d+)_output$",
        r"output_id(\d+)$",
        r"video[_-](\d+)$",
        r"^(\d+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    matches = re.findall(r"\d+", stem)
    return int(matches[-1]) if matches else None


def discover_videos(video_dir: Path) -> list[tuple[int, Path]]:
    videos: list[tuple[int, Path]] = []
    for root, _, files in os.walk(video_dir):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            task_id = task_id_from_path(path)
            if task_id is not None:
                videos.append((task_id, path))
    videos.sort(key=lambda item: item[0])
    return videos


def load_id_list(path: str | Path | None) -> set[int] | None:
    if not path:
        return None
    ids: set[int] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(line))
    return ids


def filter_videos(
    videos: Iterable[tuple[int, Path]],
    id_list: str | Path | None = None,
    limit: int = 0,
) -> list[tuple[int, Path]]:
    id_filter = load_id_list(id_list)
    filtered = [(tid, path) for tid, path in videos if id_filter is None or tid in id_filter]
    return filtered[:limit] if limit else filtered


def equal_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    count = min(total, count)
    if count <= 1:
        return [total // 2]
    return sorted({int(round(i * (total - 1) / (count - 1))) for i in range(count)})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_metric_csv(path: Path, score_col: str) -> dict[int, str]:
    scores: dict[int, str] = {}
    if not path.exists():
        return scores
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_id = row.get("task_id") or row.get("id")
            if raw_id is None:
                path_value = row.get("path") or row.get("video_path") or row.get("video")
                if not path_value:
                    continue
                task_id = task_id_from_path(Path(path_value))
            else:
                task_id = int(raw_id)
            if task_id is None:
                continue
            value = (row.get(score_col) or "").strip()
            error = (row.get("error") or "").strip()
            scores[task_id] = "" if error else value
    return scores


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return math.nan if not values else float(sum(values) / len(values))


def run_command(cmd: list[str], log_path: Path | None = None, cwd: Path | None = None) -> None:
    printable = " ".join(str(x) for x in cmd)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"$ {printable}\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=log, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=cwd, text=True)
    if proc.returncode != 0:
        where = f" See log: {log_path}" if log_path else ""
        raise RuntimeError(f"Command failed ({proc.returncode}): {printable}.{where}")


def python_executable() -> str:
    return sys.executable

