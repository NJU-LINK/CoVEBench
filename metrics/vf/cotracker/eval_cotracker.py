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

DEFAULT_EXCLUDE_LIST = ROOT / "configs" / "filters" / "mf_ssim_exclude_indices.txt"


def load_video_tensor(path: Path, frames: int, size: int) -> torch.Tensor:
    import cv2
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    idxs = equal_indices(len(vr), frames)
    if not idxs:
        raise ValueError("empty video")
    arr = vr.get_batch(idxs).asnumpy()
    resized = [cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA) for rgb in arr]
    return torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2)[None].float()


def summarize_tracks(tracks: torch.Tensor, visibilities: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    tr = tracks[0].detach().cpu().numpy()
    vis = visibilities[0].detach().cpu().numpy() > 0.5
    pos = []
    vel = []
    for idx in range(tr.shape[1]):
        good = vis[:, idx]
        pts = tr[good, idx, :] if good.any() else tr[:, idx, :]
        pos.append(pts.mean(axis=0))
        diffs = np.diff(tr[:, idx, :], axis=0)
        if good[:-1].any():
            diffs = diffs[good[:-1]]
        vel.append(diffs.mean(axis=0) if len(diffs) else np.zeros(2, dtype=np.float32))
    return np.stack(pos), np.stack(vel)


def motion_similarity(pos_a, vel_a, pos_b, vel_b, alpha: float, dmax: float) -> float:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    cpos = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=2) / dmax
    dot = np.sum(vel_a[:, None, :] * vel_b[None, :, :], axis=2)
    denom = np.linalg.norm(vel_a[:, None, :], axis=2) * np.linalg.norm(vel_b[None, :, :], axis=2) + 1e-6
    cdir = 1.0 - dot / denom
    cost = alpha * cpos + (1.0 - alpha) * cdir
    r, c = linear_sum_assignment(cost)
    return float(1.0 - np.mean(cost[r, c]))


def main() -> None:
    parser = argparse.ArgumentParser(description="MF: CoTracker motion fidelity between source and edited videos.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "weights" / "cotracker" / "scaled_offline.pth"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--id-list")
    parser.add_argument("--exclude-list", default=str(DEFAULT_EXCLUDE_LIST), help="Task ids to exclude from MF evaluation.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch

    try:
        from cotracker.predictor import CoTrackerPredictor
    except Exception as exc:
        raise RuntimeError("Cannot import cotracker. Install it in the covebench uv environment.") from exc
    if not Path(args.checkpoint).exists():
        raise SystemExit("CoTracker checkpoint missing. Run: uv run scripts/download_weights.py")

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    predictor = CoTrackerPredictor(checkpoint=args.checkpoint, offline=True).to(device).eval()
    sources = dict(discover_videos(Path(args.source_dir)))
    videos = filter_videos(discover_videos(Path(args.video_dir)), args.id_list, args.limit)
    exclude_ids = load_id_list(args.exclude_list) or set()
    if exclude_ids:
        videos = [(task_id, path) for task_id, path in videos if task_id not in exclude_ids]
    if not videos:
        raise SystemExit(f"No numbered videos found in {args.video_dir}")
    dmax = math.sqrt(2.0) * args.size

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
                    with torch.inference_mode():
                        src = load_video_tensor(source_path, args.frames, args.size).to(device)
                        edt = load_video_tensor(video_path, args.frames, args.size).to(device)
                        tracks_a, vis_a = predictor(src, grid_size=args.grid_size, grid_query_frame=0)
                        tracks_b, vis_b = predictor(edt, grid_size=args.grid_size, grid_query_frame=0)
                    pos_a, vel_a = summarize_tracks(tracks_a, vis_a)
                    pos_b, vel_b = summarize_tracks(tracks_b, vis_b)
                    row["score"] = f"{motion_similarity(pos_a, vel_a, pos_b, vel_b, args.alpha, dmax):.8f}"
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(row)
            f.flush()
            if index == 1 or index % 10 == 0:
                print(f"MF {index}/{len(videos)} task={task_id} score={row['score']} error={row['error']}", flush=True)


if __name__ == "__main__":
    main()
