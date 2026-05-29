# VQR

`eval_vqr.py` computes holistic visual quality with VisualQuality-R1.

Input: edited videos named by task id.

Sampling: 10 equal-spaced frames per video by default.

Output: `task_id,score,error`, where `score` is the average of 10 frame-level VisualQuality-R1 scores.

Example:

```bash
uv run metrics/vq/vqr/eval_vqr.py \
  --video-dir data/my_model \
  --output outputs/work/vq_vqr.csv
```

