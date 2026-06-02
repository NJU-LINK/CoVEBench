#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from covebench_eval.common import discover_videos, equal_indices, filter_videos, load_id_list  # noqa: E402

DEFAULT_EXCLUDE_LIST = ROOT / "configs" / "filters" / "ssim_exclude_indices.txt"


def sample_grays(path: Path, count: int, size: tuple[int, int]) -> list[np.ndarray]:
    import cv2
    from decord import VideoReader, cpu

    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    idxs = equal_indices(len(vr), count)
    if not idxs:
        raise ValueError("empty video")
    batch = vr.get_batch(idxs).asnumpy()
    frames = []
    for frame in batch:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(cv2.resize(gray, size, interpolation=cv2.INTER_AREA))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="SSIM: structural fidelity between source and edited videos.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--id-list")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sources = dict(discover_videos(Path(args.source_dir)))
    videos = filter_videos(discover_videos(Path(args.video_dir)), args.id_list, args.limit)
    exclude_ids = load_id_list(DEFAULT_EXCLUDE_LIST) or set()
    if exclude_ids:
        videos = [(task_id, path) for task_id, path in videos if task_id not in exclude_ids]
    if not videos:
        raise SystemExit(f"No numbered videos found in {args.video_dir}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "score", "error"])
        writer.writeheader()
        for index, (task_id, video_path) in enumerate(videos, 1):
            row = {"task_id": task_id, "score": "", "error": ""}
            source_path = sources.get(task_id)
            if not source_path:
                row["error"] = "missing source video"
            else:
                try:
                    import numpy as np
                    from skimage.metrics import structural_similarity

                    src = sample_grays(source_path, args.frames, (args.width, args.height))
                    edt = sample_grays(video_path, args.frames, (args.width, args.height))
                    n = min(len(src), len(edt))
                    score = float(np.mean([structural_similarity(a, b, data_range=255) for a, b in zip(src[:n], edt[:n])]))
                    row["score"] = f"{score:.8f}"
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(row)
            f.flush()
            if index == 1 or index % 50 == 0:
                print(f"SSIM {index}/{len(videos)} task={task_id} score={row['score']} error={row['error']}", flush=True)


if __name__ == "__main__":
    main()
