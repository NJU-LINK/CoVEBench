#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from covebench_eval.common import discover_videos, equal_indices, filter_videos  # noqa: E402


def read_equal_frames(video: Path, count: int) -> list[Image.Image]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    targets = set(equal_indices(total, count))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in targets:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
        if targets and idx > max(targets):
            break
    cap.release()
    if not frames:
        raise ValueError("empty video or failed frame reads")
    return frames


class AESModel:
    def __init__(self, device: str, predictor_path: str | None, batch_size: int):
        import torch

        aes_src = ROOT / "external" / "aesthetic-predictor-v2-5" / "src"
        if aes_src.exists():
            sys.path.insert(0, str(aes_src))
        try:
            from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        except Exception as exc:
            raise RuntimeError(
                "Cannot import aesthetic-predictor-v2-5. Run: uv run scripts/download_weights.py"
            ) from exc

        model, preprocessor = convert_v2_5_from_siglip(
            predictor_name_or_path=predictor_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16 if use_cuda else torch.float32
        self.model = model.to(self.dtype).to(device).eval()
        self.preprocessor = preprocessor
        self.device = device
        self.batch_size = batch_size
        self.torch = torch

    def score(self, images: list[Image.Image]) -> list[float]:
        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                chunk = images[start : start + self.batch_size]
                pixel_values = self.preprocessor(images=chunk, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.dtype).to(self.device)
                logits = self.model(pixel_values).logits.reshape(-1).float().cpu().tolist()
                scores.extend(float(x) for x in logits)
        return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="AES: aesthetic-predictor-v2-5, 10 equal-spaced frames per video.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--predictor-path", default=os.environ.get("AES_PREDICTOR_PATH", str(ROOT / "weights" / "aes" / "aesthetic_predictor_v2_5.pth")))
    parser.add_argument("--id-list")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    videos = filter_videos(discover_videos(Path(args.video_dir)), args.id_list, args.limit)
    if not videos:
        raise SystemExit(f"No numbered videos found in {args.video_dir}")

    model = AESModel(args.device, args.predictor_path, args.batch_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "score", "error"])
        writer.writeheader()
        for index, (task_id, path) in enumerate(videos, 1):
            row = {"task_id": task_id, "score": "", "error": ""}
            try:
                images = read_equal_frames(path, args.frames)
                try:
                    scores = model.score(images)
                finally:
                    for image in images:
                        image.close()
                row["score"] = f"{sum(scores) / len(scores):.8f}"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(row)
            f.flush()
            if index == 1 or index % 50 == 0:
                print(f"AES {index}/{len(videos)} task={task_id} score={row['score']} error={row['error']}", flush=True)


if __name__ == "__main__":
    main()
