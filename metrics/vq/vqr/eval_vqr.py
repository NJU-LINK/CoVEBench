#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from covebench_eval.common import discover_videos, equal_indices, filter_videos  # noqa: E402

PROMPT = (
    "You are doing the image quality assessment task. Here is the question: "
    "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
    "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
)
QUESTION_TEMPLATE = "{Question} Please only output the final answer with only one score in <answer> </answer> tags."


def extract_equal_frames(video_path: Path, sample_count: int, tmp_dir: Path) -> list[str]:
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    idxs = equal_indices(len(vr), sample_count)
    if not idxs:
        raise RuntimeError("no frames decoded")
    batch = vr.get_batch(idxs).asnumpy()
    paths = []
    for idx, frame in zip(idxs, batch):
        out = tmp_dir / f"frame_{idx:06d}.jpg"
        Image.fromarray(frame).save(out, quality=95)
        paths.append(str(out))
    return paths


def make_messages(image_paths: list[str]):
    text = QUESTION_TEMPLATE.format(Question=PROMPT)
    return [
        [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": text}]}]
        for img_path in image_paths
    ]


def parse_score(output: str) -> float:
    matches = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)
    answer = matches[-1].strip() if matches else output.strip()
    score_match = re.search(r"\d+(?:\.\d+)?", answer)
    if not score_match:
        raise ValueError(f"cannot parse score from output: {output!r}")
    return float(score_match.group())


def score_batch(image_paths, model, processor, device, max_new_tokens):
    import torch
    from qwen_vl_utils import process_vision_info

    with torch.inference_mode():
        messages = make_messages(image_paths)
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=1,
        )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [parse_score(output) for output in outputs]


def score_with_retry(image_paths, model, processor, device, batch_size, max_new_tokens):
    import torch

    scores = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        try:
            scores.extend(score_batch(chunk, model, processor, device, max_new_tokens))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise
            scores.extend(score_with_retry(chunk, model, processor, device, max(1, len(chunk) // 2), max_new_tokens))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="VQR: VisualQuality-R1 on 10 equal-spaced frames.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(ROOT / "weights" / "hf" / "VisualQuality-R1-7B"))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--id-list")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    videos = filter_videos(discover_videos(Path(args.video_dir)), args.id_list, args.limit)
    if not videos:
        raise SystemExit(f"No numbered videos found in {args.video_dir}")
    if not Path(args.model).exists() and "/" not in args.model:
        raise SystemExit("VisualQuality-R1 model missing. Run: uv run scripts/download_weights.py")

    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **load_kwargs).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    tmp_root = Path(args.output).parent / "vqr_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "score", "error"])
        writer.writeheader()
        for index, (task_id, path) in enumerate(videos, 1):
            row = {"task_id": task_id, "score": "", "error": ""}
            try:
                with tempfile.TemporaryDirectory(prefix=f"task_{task_id}_", dir=tmp_root) as tmp:
                    image_paths = extract_equal_frames(path, args.frames, Path(tmp))
                    scores = score_with_retry(image_paths, model, processor, device, args.batch_size, args.max_new_tokens)
                row["score"] = f"{float(np.mean(scores)):.8f}"
            except Exception as exc:
                row["error"] = repr(exc)
            writer.writerow(row)
            f.flush()
            if index == 1 or index % 10 == 0:
                print(f"VQR {index}/{len(videos)} task={task_id} score={row['score']} error={row['error']}", flush=True)


if __name__ == "__main__":
    main()
