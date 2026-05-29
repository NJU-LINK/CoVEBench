# MSM

`eval_msm.py` computes edited-only motion smoothness using optical-flow consistency.

Input: edited videos named by task id.

Sampling: 10 equal-spaced frames per video by default.

Output: `task_id,score,error`, where higher values indicate smoother frame-to-frame motion.

Example:

```bash
uv run metrics/vq/msm/eval_msm.py \
  --video-dir data/my_model \
  --output outputs/work/vq_msm.csv
```

