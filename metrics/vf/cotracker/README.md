# MF / CoTracker

`eval_cotracker.py` computes motion fidelity with CoTracker.

Input:

- `--source-dir`: original videos named by task id. The top-level one-command runner resolves checklist `videoA_path` entries and creates this materialized layout under `--work-dir`.
- `--video-dir`: edited videos named by the same task id.

Sampling: 16 equal-spaced frames per video by default, following the prior CoTracker/VEditBench-style implementation.

Output: `task_id,score,error`, where higher values indicate more similar source/edit motion trajectories.

Example:

```bash
uv run metrics/vf/cotracker/eval_cotracker.py \
  --source-dir outputs/my_model_work/source_by_task_id \
  --video-dir data/my_model \
  --output outputs/work/vf_mf.csv \
  --device cuda:0
```

