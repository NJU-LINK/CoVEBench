# MF / CoTracker

`eval_cotracker.py` computes motion fidelity with CoTracker.

Input:

- `--source-dir`: original videos named by task id.
- `--video-dir`: edited videos named by the same task id.

Sampling: 16 equal-spaced frames per video by default, following the prior CoTracker/VEditBench-style implementation.

Output: `task_id,score,error`, where higher values indicate more similar source/edit motion trajectories.

Example:

```bash
uv run metrics/vf/cotracker/eval_cotracker.py \
  --source-dir data/source \
  --video-dir data/my_model \
  --output outputs/work/vf_mf.csv \
  --device cuda:0
```

