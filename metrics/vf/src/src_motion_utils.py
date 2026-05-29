#!/usr/bin/env python3
"""
Masked motion similarity for edited videos.

This script reuses the preservation masks produced by pipeline.py:
  source video A: data/<basename(checklist.videoA_path)>
  masks:          outputs/<item_id>/masks/mask_0000.png ...
  target video B: ../datasets/<dataset>/<dataset-specific-name>.mp4

It computes sparse trajectory similarity inside the preservation mask. The
default backend uses OpenCV Lucas-Kanade optical flow so it is lightweight and
runnable without installing CoTracker. The scoring step follows the same shape
as the proposed CoTracker metric: positional cost + directional cost + matching.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKLIST = SCRIPT_DIR / "checklist.json"

# Layout: data/ and dataset folders (wan/, kiwi/...) live as SIBLINGS of cut/
#   parent/
#     cut/         ← this file
#     data/        ← source videos
#     wan/         ← target videos (dataset)
# Override with CUT_DATA_DIR / CUT_DATASETS_ROOT env vars when needed.
import os
DATA_DIR = Path(os.environ.get("CUT_DATA_DIR", SCRIPT_DIR.parent / "data"))
MASK_ROOT = SCRIPT_DIR / "outputs"
DATASETS_ROOT = Path(os.environ.get("CUT_DATASETS_ROOT", SCRIPT_DIR.parent))

DATASETS = ("kiwi", "lucy", "insv2v", "icve", "vacepackage", "reco", "ditto", "omini", "wan", "happyhorse")

# `lucy` lives inside cut/ (not as a sibling of cut/) and uses task_<id>.mp4 naming.
LUCY_DIR = SCRIPT_DIR / "lucy_edit_results"
HAPPYHORSE_DIR = DATASETS_ROOT / "happyhorse"


def choose_dataset() -> str:
    print("请选择目标视频文件夹：")
    for i, name in enumerate(DATASETS, start=1):
        print(f"  {i}. {name}")
    while True:
        choice = input("输入序号或名称: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(DATASETS):
                return DATASETS[idx - 1]
        if choice in DATASETS:
            return choice
        print("无效选择，请重新输入。")


def source_video_path(item: dict[str, Any]) -> Path:
    # Must match pipeline.py: only the basename of videoA_path is used.
    return DATA_DIR / Path(item["videoA_path"]).name


def target_video_candidates(dataset: str, item: dict[str, Any]) -> list[Path]:
    item_id = str(item["id"])

    if dataset == "lucy":
        return [LUCY_DIR / f"task_{item_id}.mp4"]
    if dataset == "happyhorse":
        return [HAPPYHORSE_DIR / f"video-edit-v1-{int(item_id):03d}.mp4"]

    dataset_dir = DATASETS_ROOT / dataset
    if dataset == "kiwi":
        return [dataset_dir / f"video_{item_id}.mp4"]
    if dataset == "insv2v":
        return [dataset_dir / f"task_{item_id}" / f"task_{item_id}_output.mp4"]
    if dataset == "ditto":
        name = Path(item["videoA_path"]).stem
        return [dataset_dir / f"output_id{item_id}_{name}.mp4"]
    return [dataset_dir / f"{item_id}.mp4"]


def target_video_path(dataset: str, item: dict[str, Any]) -> Path | None:
    for candidate in target_video_candidates(dataset, item):
        if candidate.exists():
            return candidate
    return None


def mask_dir_for_item(item: dict[str, Any]) -> Path:
    return MASK_ROOT / str(item["id"]) / "masks"


def read_video_frames(path: Path, fps: float, max_frames: int) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError:
        raise SystemExit("pip install opencv-python")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, int(round(native_fps / fps)))
    frames: list[np.ndarray] = []
    idx = 0

    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        idx += 1

    cap.release()
    return frames


def read_masks(mask_dir: Path, count: int, size: tuple[int, int]) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError:
        raise SystemExit("pip install opencv-python")

    width, height = size
    masks: list[np.ndarray] = []
    for idx in range(count):
        path = mask_dir / f"mask_{idx:04d}.png"
        if not path.exists():
            masks.append(np.zeros((height, width), dtype=bool))
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            masks.append(np.zeros((height, width), dtype=bool))
            continue
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(mask > 127)
    return masks


def align_frames(
    source_frames: list[np.ndarray],
    target_frames: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    try:
        import cv2
    except ImportError:
        raise SystemExit("pip install opencv-python")

    count = min(len(source_frames), len(target_frames))
    source_frames = source_frames[:count]
    target_frames = target_frames[:count]
    if not source_frames:
        return [], []

    height, width = source_frames[0].shape[:2]
    aligned_target: list[np.ndarray] = []
    for frame in target_frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        aligned_target.append(frame)
    return source_frames, aligned_target


def sample_points(mask: np.ndarray, max_points: int, grid: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    height, width = mask.shape
    points: list[tuple[float, float]] = []
    for y in range(grid // 2, height, grid):
        for x in range(grid // 2, width, grid):
            if mask[y, x]:
                points.append((float(x), float(y)))

    if len(points) < min(16, max_points):
        stride = max(1, len(xs) // max_points)
        points.extend((float(x), float(y)) for x, y in zip(xs[::stride], ys[::stride]))

    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = [points[i] for i in idx]

    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)


def track_lk(frames: list[np.ndarray], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError:
        raise SystemExit("pip install opencv-python")

    if len(frames) == 0 or len(points) == 0:
        return np.empty((0, 0, 2), dtype=np.float32), np.empty((0, 0), dtype=bool)

    grays = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    num_frames = len(grays)
    num_points = points.shape[0]

    tracks = np.zeros((num_frames, num_points, 2), dtype=np.float32)
    visible = np.zeros((num_frames, num_points), dtype=bool)
    tracks[0] = points.reshape(num_points, 2)
    visible[0] = True

    prev_points = points.copy()
    active = np.ones(num_points, dtype=bool)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    for t in range(1, num_frames):
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            grays[t - 1], grays[t], prev_points, None, **lk_params
        )
        if next_points is None or status is None:
            break

        status = status.reshape(-1).astype(bool)
        active &= status
        tracks[t] = next_points.reshape(num_points, 2)
        visible[t] = active
        prev_points = next_points

    return tracks, visible


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ImportError:
        # Greedy fallback. Less exact than Hungarian, but keeps the script usable.
        rows: list[int] = []
        cols: list[int] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for flat_idx in np.argsort(cost, axis=None):
            r, c = np.unravel_index(int(flat_idx), cost.shape)
            if r in used_rows or c in used_cols:
                continue
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)
            if len(rows) == min(cost.shape):
                break
        return np.array(rows), np.array(cols)


def trajectory_cost_matrix(
    tracks_a: np.ndarray,
    vis_a: np.ndarray,
    tracks_b: np.ndarray,
    vis_b: np.ndarray,
    frame_diag: float,
    alpha: float,
    min_common: int,
) -> np.ndarray:
    n_a = tracks_a.shape[1]
    n_b = tracks_b.shape[1]
    cost = np.ones((n_a, n_b), dtype=np.float32)

    vel_a = np.diff(tracks_a, axis=0)
    vel_b = np.diff(tracks_b, axis=0)
    vel_vis_a = vis_a[:-1] & vis_a[1:]
    vel_vis_b = vis_b[:-1] & vis_b[1:]

    for i in range(n_a):
        for j in range(n_b):
            common = vis_a[:, i] & vis_b[:, j]
            if int(common.sum()) < min_common:
                continue

            pos = np.linalg.norm(tracks_a[common, i] - tracks_b[common, j], axis=1)
            pos_cost = float(pos.mean() / max(frame_diag, 1.0))

            vel_common = vel_vis_a[:, i] & vel_vis_b[:, j]
            if int(vel_common.sum()) < max(1, min_common - 1):
                dir_cost = 1.0
            else:
                va = vel_a[vel_common, i]
                vb = vel_b[vel_common, j]
                denom = np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-6
                cos = np.sum(va * vb, axis=1) / denom
                dir_cost = float((1.0 - np.clip(cos, -1.0, 1.0)).mean() / 2.0)

            cost[i, j] = alpha * pos_cost + (1.0 - alpha) * dir_cost

    return np.clip(cost, 0.0, 1.0)


def compute_motion_similarity(
    source_path: Path,
    target_path: Path,
    mask_dir: Path,
    fps: float,
    max_frames: int,
    max_points: int,
    grid: int,
    alpha: float,
    min_visible_ratio: float,
) -> dict[str, Any]:
    source_frames = read_video_frames(source_path, fps=fps, max_frames=max_frames)
    target_frames = read_video_frames(target_path, fps=fps, max_frames=max_frames)
    source_frames, target_frames = align_frames(source_frames, target_frames)
    if len(source_frames) < 2:
        return {"status": "too_few_frames", "frames": len(source_frames)}

    height, width = source_frames[0].shape[:2]
    masks = read_masks(mask_dir, len(source_frames), (width, height))

    start_frame = next((i for i, m in enumerate(masks) if m.any()), None)
    if start_frame is None:
        return {"status": "empty_mask", "frames": len(source_frames)}

    source_frames = source_frames[start_frame:]
    target_frames = target_frames[start_frame:]
    masks = masks[start_frame:]

    points = sample_points(masks[0], max_points=max_points, grid=grid)
    if len(points) < 2:
        return {
            "status": "too_few_points",
            "frames": len(source_frames),
            "sampled_points": int(len(points)),
            "start_frame": start_frame,
        }

    tracks_a, vis_a = track_lk(source_frames, points)
    tracks_b, vis_b = track_lk(target_frames, points)

    min_visible = max(2, int(math.ceil(len(source_frames) * min_visible_ratio)))
    valid_a = vis_a.sum(axis=0) >= min_visible
    valid_b = vis_b.sum(axis=0) >= min_visible
    tracks_a = tracks_a[:, valid_a]
    vis_a = vis_a[:, valid_a]
    tracks_b = tracks_b[:, valid_b]
    vis_b = vis_b[:, valid_b]

    if tracks_a.shape[1] == 0 or tracks_b.shape[1] == 0:
        return {
            "status": "no_valid_tracks",
            "frames": len(source_frames),
            "sampled_points": int(len(points)),
            "valid_tracks_a": int(tracks_a.shape[1]),
            "valid_tracks_b": int(tracks_b.shape[1]),
            "start_frame": start_frame,
        }

    frame_diag = math.sqrt(width * width + height * height)
    cost = trajectory_cost_matrix(
        tracks_a,
        vis_a,
        tracks_b,
        vis_b,
        frame_diag=frame_diag,
        alpha=alpha,
        min_common=min_visible,
    )
    rows, cols = _linear_sum_assignment(cost)
    matched_costs = cost[rows, cols] if len(rows) else np.array([], dtype=np.float32)
    if len(matched_costs) == 0:
        return {"status": "no_matches", "frames": len(source_frames), "start_frame": start_frame}

    mean_cost = float(matched_costs.mean())
    score = max(0.0, min(1.0, 1.0 - mean_cost))

    return {
        "status": "completed",
        "motion_similarity": round(score, 6),
        "mean_cost": round(mean_cost, 6),
        "frames": len(source_frames),
        "start_frame": start_frame,
        "fps": fps,
        "alpha": alpha,
        "sampled_points": int(len(points)),
        "valid_tracks_a": int(tracks_a.shape[1]),
        "valid_tracks_b": int(tracks_b.shape[1]),
        "matched_tracks": int(len(matched_costs)),
        "method": "masked_lucas_kanade_trajectories",
    }


def run_dataset(args: argparse.Namespace) -> int:
    dataset = args.dataset or choose_dataset()
    if dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset: {dataset}. Expected one of: {', '.join(DATASETS)}")

    items: list[dict[str, Any]] = json.loads(CHECKLIST.read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids)
        items = [item for item in items if int(item["id"]) in wanted]
    if args.limit:
        items = items[: args.limit]

    out_dir = SCRIPT_DIR / dataset
    out_dir.mkdir(exist_ok=True)

    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        item_id = int(item["id"])
        video_a_id = Path(item["videoA_path"]).stem
        output_path = out_dir / f"{item_id}.json"

        if output_path.exists() and not args.force:
            result = json.loads(output_path.read_text(encoding="utf-8"))
            status = result.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
            rows.append(result)
            print(f"[{idx}/{len(items)}] id={item_id} videoA={video_a_id}  cached:{status}", flush=True)
            continue

        source = source_video_path(item)
        target = target_video_path(dataset, item)
        masks = mask_dir_for_item(item)

        result: dict[str, Any] = {
            "item_id": item_id,
            "video_a_id": video_a_id,
            "target_id": item_id,
            "dataset": dataset,
            "source_video": str(source),
            "target_video": str(target) if target else None,
            "mask_dir": str(masks),
        }

        try:
            if not source.exists():
                result.update(status="source_not_found")
            elif target is None:
                result.update(
                    status="target_not_found",
                    target_candidates=[str(p) for p in target_video_candidates(dataset, item)],
                )
            elif not masks.exists():
                result.update(status="mask_not_found")
            else:
                result.update(
                    compute_motion_similarity(
                        source,
                        target,
                        masks,
                        fps=args.fps,
                        max_frames=args.max_frames,
                        max_points=args.max_points,
                        grid=args.grid,
                        alpha=args.alpha,
                        min_visible_ratio=args.min_visible_ratio,
                    )
                )
        except Exception as exc:
            result.update(status="error", error=str(exc))

        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(result)
        status = result.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        score = result.get("motion_similarity")
        suffix = f" score={score:.4f}" if isinstance(score, float) else ""
        print(f"[{idx}/{len(items)}] id={item_id} videoA={video_a_id}  {status}{suffix}", flush=True)

    summary = {
        "dataset": dataset,
        "counts": counts,
        "completed": sum(1 for row in rows if row.get("status") == "completed"),
        "mean_motion_similarity": None,
    }
    scores = [row["motion_similarity"] for row in rows if row.get("status") == "completed"]
    if scores:
        summary["mean_motion_similarity"] = round(float(np.mean(scores)), 6)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n── Summary ──────────────────────")
    for key, value in sorted(counts.items()):
        print(f"  {key:<22} {value:>4}")
    if summary["mean_motion_similarity"] is not None:
        print(f"  mean_motion_similarity {summary['mean_motion_similarity']:.6f}")
    print(f"  output_dir             {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute masked motion similarity.")
    parser.add_argument("--dataset", choices=DATASETS, help="Target dataset folder. If omitted, prompt interactively.")
    parser.add_argument("--ids", nargs="+", type=int, help="Run only specific checklist item IDs.")
    parser.add_argument("--limit", type=int, help="Process at most N checklist items.")
    parser.add_argument("--force", action="store_true", help="Recompute existing per-item JSON files.")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling FPS. Default matches pipeline.py masks.")
    parser.add_argument("--max-frames", type=int, default=32, help="Maximum sampled frames per video.")
    parser.add_argument("--max-points", type=int, default=256, help="Maximum mask points to track.")
    parser.add_argument("--grid", type=int, default=16, help="Grid spacing for mask point sampling.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Position vs direction cost weight.")
    parser.add_argument("--min-visible-ratio", type=float, default=0.6, help="Minimum visible track ratio.")
    args = parser.parse_args()
    return run_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
