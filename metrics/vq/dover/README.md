# TQ / DOVER++

`eval_dover.py` computes technical quality using DOVER++.

Input: edited videos named by task id.

Output: `task_id,score,error`, where `score` is DOVER++ `technical score`.

Example:

```bash
uv run metrics/vq/dover/eval_dover.py \
  --video-dir data/my_model \
  --output outputs/work/vq_tq.csv \
  --device cuda:0
```

