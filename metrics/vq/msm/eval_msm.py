#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from covebench_eval.common import discover_videos, equal_indices, filter_videos  # noqa: E402


def sample_gray_frames(video_path: Path, count: int, size: tuple[int, int]) -> list[np.ndarray]:
    import cv2
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    idxs = equal_indices(len(vr), count)
    if not idxs:
        raise ValueError("empty video")
    batch = vr.get_batch(idxs).asnumpy()
    grays = []
    for frame in batch:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        grays.append(cv2.resize(gray, size, interpolation=cv2.INTER_AREA))
    return grays


def motion_smoothness(frames: list[np.ndarray], motion_threshold: float = 0.05) -> float:
    import cv2
    import numpy as np

    if len(frames) < 3:
        return math.nan
    flows = [
        cv2.calcOpticalFlowFarneback(frames[i], frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        for i in range(len(frames) - 1)
    ]
    scores = []
    eps = 1e-6
    for f0, f1 in zip(flows[:-1], flows[1:]):
        m0 = np.linalg.norm(f0, axis=2)
        m1 = np.linalg.norm(f1, axis=2)
        mask = (m0 > motion_threshold) | (m1 > motion_threshold)
        if not np.any(mask):
            scores.append(1.0)
            continue
        dot = np.sum(f0 * f1, axis=2)
        cos = dot / (m0 * m1 + eps)
        dir_score = (np.clip(cos, -1.0, 1.0) + 1.0) / 2.0
        mag_score = 1.0 - np.abs(m0 - m1) / (m0 + m1 + eps)
        score = 0.7 * dir_score[mask] + 0.3 * mag_score[mask]
        scores.append(float(np.mean(np.clip(score, 0.0, 1.0))))
    return float(np.mean(scores)) if scores else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="MSM: edited-only optical-flow motion smoothness.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--id-list")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    videos = filter_videos(discover_videos(Path(args.video_dir)), args.id_list, args.limit)
    if not videos:
        raise SystemExit(f"No numbered videos found in {args.video_dir}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "score", "error"])
        writer.writeheader()
        for index, (task_id, path) in enumerate(videos, 1):
            row = {"task_id": task_id, "score": "", "error": ""}
            try:
                frames = sample_gray_frames(path, args.frames, (args.width, args.height))
                row["score"] = f"{motion_smoothness(frames):.8f}"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(row)
            f.flush()
            if index == 1 or index % 50 == 0:
                print(f"MSM {index}/{len(videos)} task={task_id} score={row['score']} error={row['error']}", flush=True)


if __name__ == "__main__":
    main()
