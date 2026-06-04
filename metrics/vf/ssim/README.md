# SSIM

`eval_ssim.py` computes structural fidelity between source and edited videos.

Input:

- `--source-dir`: original videos named by task id. The top-level one-command runner resolves checklist `videoA_path` entries and creates this materialized layout under `--work-dir`.
- `--video-dir`: edited videos named by the same task id.

Sampling: 10 equal-spaced frame pairs per video by default.

Output: `task_id,score,error`, where `score` is mean SSIM over sampled frame pairs.

Example:

```bash
uv run metrics/vf/ssim/eval_ssim.py \
  --source-dir outputs/my_model_work/source_by_task_id \
  --video-dir data/my_model \
  --output outputs/work/vf_ssim.csv
```

