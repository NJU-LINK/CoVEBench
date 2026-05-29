# AES

`eval_aes.py` computes aesthetic quality with `aesthetic-predictor-v2-5`.

Input: edited videos named by task id.

Sampling: 10 equal-spaced frames per video by default.

Output: `task_id,score,error`, where `score` is the average frame-level aesthetic score.

Example:

```bash
uv run metrics/vq/aes/eval_aes.py \
  --video-dir data/my_model \
  --output outputs/work/vq_aes.csv \
  --device cuda:0
```

