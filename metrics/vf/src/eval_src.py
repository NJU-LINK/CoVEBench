#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from covebench_eval.common import discover_videos, filter_videos, task_id_from_path  # noqa: E402

import src_motion_utils as motion_utils  # noqa: E402
import src_pipeline as pipeline  # noqa: E402
from src_motion_utils import align_frames, read_masks, read_video_frames  # noqa: E402
from src_pipeline import (  # noqa: E402
    FRAME_FPS,
    GDINO_MIN_SCORE,
    _draw_entity_boxes,
    extract_frames,
    ground_entities,
    load_models,
    sam2_propagate,
    save_outputs,
)


class DinoScorer:
    """L2-normalized CLS token from DINOv2; cosine [-1,1] mapped to [0,1]."""

    def __init__(self, model_name: str, cpu: bool = False) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def cosine(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        inputs = self.processor(images=[crop_a, crop_b], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            feats = outputs.last_hidden_state[:, 0]
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-6)
            sim = float((feats[0] * feats[1]).sum().item())
        return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))


def expanded_bbox(mask: np.ndarray, expand_ratio: float) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x = int(round((x2 - x1) * expand_ratio))
    pad_y = int(round((y2 - y1) * expand_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def masked_crop(frame: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    crop[~crop_mask] = 0
    return crop


def compute_src_score(
    source_video: Path,
    target_video: Path,
    src_mask_dir: Path,
    tgt_mask_dir: Path,
    dino: DinoScorer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    src_frames = read_video_frames(source_video, fps=args.fps, max_frames=args.max_frames)
    tgt_frames = read_video_frames(target_video, fps=args.fps, max_frames=args.max_frames)
    src_frames, tgt_frames = align_frames(src_frames, tgt_frames)
    if not src_frames:
        return {"status": "too_few_frames", "frames": 0}

    h, w = src_frames[0].shape[:2]
    n = min(len(src_frames), len(tgt_frames))
    src_masks = read_masks(src_mask_dir, n, (w, h))
    tgt_masks = read_masks(tgt_mask_dir, n, (w, h))

    sims: list[float] = []
    missing = 0
    used = 0

    for i in range(n):
        src_mask = src_masks[i]
        if int(src_mask.sum()) < args.min_mask_pixels:
            continue

        tgt_mask = tgt_masks[i]
        if int(tgt_mask.sum()) < args.min_mask_pixels:
            missing += 1
            sims.append(0.0)
            used += 1
            continue

        src_bbox = expanded_bbox(src_mask, args.bbox_expand)
        tgt_bbox = expanded_bbox(tgt_mask, args.bbox_expand)
        if src_bbox is None or tgt_bbox is None:
            continue

        crop_a = masked_crop(src_frames[i], src_mask, src_bbox)
        crop_b = masked_crop(tgt_frames[i], tgt_mask, tgt_bbox)
        sims.append(dino.cosine(crop_a, crop_b))
        used += 1

    if used == 0:
        return {"status": "empty_mask", "frames": n}

    return {
        "status": "completed",
        "frames": n,
        "used_frames": used,
        "missing_in_target": missing,
        "missing_rate": round(missing / used, 6),
        "src_score": round(float(np.mean(sims)), 6),
        "fps": args.fps,
        "bbox_expand": args.bbox_expand,
        "min_mask_pixels": args.min_mask_pixels,
        "method": "groundingdino_sam2_bilateral_masked_dinov2",
    }


def load_checklist(path: Path) -> dict[int, dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["id"]): item for item in items}


def metadata_path(mask_root: Path, task_id: int) -> Path:
    return mask_root / str(task_id) / "metadata.json"


def load_meta(mask_root: Path, task_id: int) -> dict[str, Any] | None:
    path = metadata_path(mask_root, task_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def source_mask_dir(mask_root: Path, task_id: int) -> Path:
    return mask_root / str(task_id) / "masks"


def target_mask_dir(mask_root: Path, task_id: int) -> Path:
    return mask_root / str(task_id) / "masks"


def configure_pipeline(args: argparse.Namespace, source_mask_root: Path) -> None:
    pipeline.CHECKLIST = args.checklist
    pipeline.DATA_DIR = args.source_dir
    pipeline.OUTPUTS_DIR = source_mask_root
    pipeline.SAM2_CKPT = str(args.sam2_checkpoint)
    if args.gdino_model:
        pipeline.GDINO_MODEL_ID = args.gdino_model
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)


def copy_existing_source_cache(existing_root: Path, work_root: Path, task_ids: list[int]) -> None:
    if not existing_root.exists():
        return
    for task_id in task_ids:
        src = existing_root / str(task_id)
        dst = work_root / str(task_id)
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)


def generate_source_mask(
    item: dict[str, Any],
    source_video: Path,
    source_mask_root: Path,
    gdino_processor,
    gdino_model,
    sam2,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = int(item["id"])
    meta = load_meta(source_mask_root, task_id)
    if meta and meta.get("status") == "completed" and source_mask_dir(source_mask_root, task_id).exists() and not args.force_source_mask:
        return meta

    if meta is None:
        if not args.allow_llm:
            return {"status": "source_metadata_missing", "error": "Run with --allow-llm or provide --source-mask-root cache."}
        client = pipeline._llm_client()
        meta = pipeline.run_stage1(item, client)

    if meta.get("status") not in ("ready_for_gpu", "completed"):
        return meta

    meta = dict(meta)
    meta["local_path"] = str(source_video)
    if meta.get("status") == "completed" and source_mask_dir(source_mask_root, task_id).exists() and not args.force_source_mask:
        return meta
    return pipeline.run_stage2(item, meta, gdino_processor, gdino_model, sam2)


def generate_target_mask(
    task_id: int,
    target_video: Path,
    preserved_entities: list[dict[str, Any]],
    target_root: Path,
    gdino_processor,
    gdino_model,
    sam2,
    args: argparse.Namespace,
) -> dict[str, Any]:
    item_dir = target_root / str(task_id)
    item_dir.mkdir(parents=True, exist_ok=True)
    meta_path = item_dir / "metadata.json"

    if meta_path.exists() and not args.force_target_mask:
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached.get("status") in ("completed", "no_grounded_entities"):
            return cached

    meta: dict[str, Any] = {
        "id": task_id,
        "target_video": str(target_video),
        "preserved_entities": preserved_entities,
        "fps": FRAME_FPS,
    }

    frames_dir = item_dir / "frames"
    frames = extract_frames(target_video, frames_dir)
    meta["num_frames"] = len(frames)
    if not frames:
        meta.update(status="error", error="ffmpeg produced 0 frames from target")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    records = ground_entities(gdino_processor, gdino_model, frames, preserved_entities)
    matched = [r for r in records if r["box"] is not None]
    meta["gdino_records"] = records
    meta["matched_entities"] = len(matched)

    preview = _draw_entity_boxes(frames[0], records)
    (item_dir / "gdino_boxes_preview.jpg").write_bytes(preview)

    if not matched:
        meta.update(
            status="no_grounded_entities",
            error=f"Grounding DINO found no entity in target at score >= {GDINO_MIN_SCORE:.2f}",
        )
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    video_masks = sam2_propagate(sam2, frames_dir, [r["box"] for r in matched])
    save_outputs(item_dir, frames, video_masks)

    meta["status"] = "completed"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="SRC: Static Region Consistency via GroundingDINO + SAM2 + DINOv2.")
    parser.add_argument("--checklist", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--source-mask-root", default="", help="Optional existing source mask cache.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--id-list")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--bbox-expand", type=float, default=0.15)
    parser.add_argument("--dino-model", default=str(ROOT / "weights" / "hf" / "dinov2-small"))
    parser.add_argument("--gdino-model", default=str(ROOT / "weights" / "hf" / "grounding-dino-base"))
    parser.add_argument("--sam2-checkpoint", default=str(ROOT / "weights" / "sam2" / "sam2_hiera_large.pt"))
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--allow-llm", action="store_true", help="Generate missing source preserved-entity metadata with the LLM prompts in src_pipeline.py.")
    parser.add_argument("--force", action="store_true", help="Recompute cached per-task SRC JSON.")
    parser.add_argument("--force-source-mask", action="store_true")
    parser.add_argument("--force-target-mask", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for DINOv2 scoring only; SAM2/GDINO still need GPU.")
    args = parser.parse_args()

    args.checklist = Path(args.checklist).resolve()
    args.source_dir = Path(args.source_dir).resolve()
    args.video_dir = Path(args.video_dir).resolve()
    args.output = Path(args.output).resolve()
    args.work_dir = Path(args.work_dir).resolve() if args.work_dir else args.output.with_suffix("")
    args.sam2_checkpoint = Path(args.sam2_checkpoint).resolve()

    if not args.checklist.exists():
        raise SystemExit(f"Checklist not found: {args.checklist}")
    if not args.sam2_checkpoint.exists():
        raise SystemExit(f"SAM2 checkpoint not found: {args.sam2_checkpoint}. Run scripts/download_weights.py first.")
    if Path(str(args.gdino_model)).is_absolute() and not Path(str(args.gdino_model)).exists():
        args.gdino_model = "IDEA-Research/grounding-dino-base"
    if Path(str(args.dino_model)).is_absolute() and not Path(str(args.dino_model)).exists():
        args.dino_model = "facebook/dinov2-small"

    # Masks are extracted at pipeline.FRAME_FPS, but compute_src_score re-samples the
    # videos at args.fps and pairs frames with masks by index. If the two disagree the
    # masks silently align to the wrong frames and the score is meaningless — so refuse.
    if args.fps != FRAME_FPS:
        raise SystemExit(
            f"--fps {args.fps} must equal the mask sampling rate FRAME_FPS={FRAME_FPS} "
            f"(set in src_pipeline.py); masks and frames are paired by index."
        )

    source_videos = dict(filter_videos(discover_videos(args.source_dir), args.id_list, args.limit))
    edited_videos = dict(filter_videos(discover_videos(args.video_dir), args.id_list, args.limit))
    task_ids = sorted(set(source_videos) & set(edited_videos))
    if args.limit:
        task_ids = task_ids[: args.limit]
    if not task_ids:
        raise SystemExit("No matching task ids between source-dir and video-dir.")

    checklist = load_checklist(args.checklist)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source_mask_root = args.work_dir / "source_masks"
    target_mask_root = args.work_dir / "target_masks"
    result_root = args.work_dir / "results"
    for directory in (source_mask_root, target_mask_root, result_root):
        directory.mkdir(parents=True, exist_ok=True)

    if args.source_mask_root:
        existing_source_root = Path(args.source_mask_root).resolve()
        copy_existing_source_cache(existing_source_root, source_mask_root, task_ids)
    configure_pipeline(args, source_mask_root)
    motion_utils.DATA_DIR = args.source_dir
    motion_utils.MASK_ROOT = source_mask_root

    print(f"SRC tasks: {len(task_ids)}", flush=True)
    print("Loading Grounding DINO + SAM2 ...", flush=True)
    gdino_processor, gdino_model, sam2 = load_models()
    print(f"Loading DINOv2 scorer ({args.dino_model}) ...", flush=True)
    dino = DinoScorer(str(args.dino_model), cpu=args.cpu)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "score", "error"])
        writer.writeheader()
        for idx, task_id in enumerate(task_ids, 1):
            row = {"task_id": task_id, "score": "", "error": ""}
            result_path = result_root / f"{task_id}.json"
            try:
                if result_path.exists() and not args.force:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    item = checklist.get(task_id)
                    if item is None:
                        raise RuntimeError("task id missing from checklist")
                    source_meta = generate_source_mask(
                        item,
                        source_videos[task_id],
                        source_mask_root,
                        gdino_processor,
                        gdino_model,
                        sam2,
                        args,
                    )
                    if source_meta.get("status") != "completed":
                        result = {"status": f"source_{source_meta.get('status', 'failed')}", "source_meta": source_meta}
                    elif not source_meta.get("preserved_entities"):
                        result = {"status": "no_preserved_entities"}
                    else:
                        target_meta = generate_target_mask(
                            task_id,
                            edited_videos[task_id],
                            source_meta["preserved_entities"],
                            target_mask_root,
                            gdino_processor,
                            gdino_model,
                            sam2,
                            args,
                        )
                        if target_meta.get("status") != "completed":
                            result = {"status": f"target_{target_meta.get('status', 'failed')}", "target_meta": target_meta}
                        else:
                            result = compute_src_score(
                                source_videos[task_id],
                                edited_videos[task_id],
                                source_mask_dir(source_mask_root, task_id),
                                target_mask_dir(target_mask_root, task_id),
                                dino,
                                args,
                            )
                    result.update({"task_id": task_id, "source_video": str(source_videos[task_id]), "edited_video": str(edited_videos[task_id])})
                    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

                if result.get("status") == "completed":
                    row["score"] = f"{float(result['src_score']):.8f}"
                else:
                    row["error"] = str(result.get("status", "unknown"))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(row)
            f.flush()
            if idx == 1 or idx % 20 == 0:
                print(f"SRC {idx}/{len(task_ids)} task={task_id} score={row['score']} error={row['error']}", flush=True)

    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
