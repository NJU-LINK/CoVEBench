# SRC: Static Region Consistency

`SRC` measures whether source-video entities that should remain unchanged are visually preserved after editing. It reuses the historical CoVEBench pipeline:

1. Extract preserved entities from Semantic Preservation checklist items.
2. Ground those entities with Grounding DINO and propagate masks with SAM2.
3. Compare source/edited masked crops with DINOv2 cosine similarity, mapped to `[0, 1]`.

Higher is better.

## Inputs

- `--checklist`: CoVEBench checklist JSON with `id`, `videoA_path`, and Semantic Preservation questions.
- `--source-dir`: source videos named by task id for direct metric calls. The top-level one-command runner resolves checklist `videoA_path` entries and creates this materialized layout under `--work-dir`.
- `--video-dir`: edited videos named by task id.
- `--source-mask-root`: optional cache of source-side SAM2 masks.

If source masks are missing, run with `--allow-llm` and set `LLM_API_KEY` so the script can extract preserved entities from the checklist before running Grounding DINO + SAM2.

## Run

```bash
uv run metrics/vf/src/eval_src.py \
  --checklist data/covebench_hf/checklist.json \
  --source-dir outputs/my_model_work/source_by_task_id \
  --video-dir data/my_model \
  --output outputs/my_model_work/vf_src.csv \
  --work-dir outputs/my_model_work/src_cache \
  --source-mask-root data/source_masks
```

The CSV schema is `task_id,score,error`; `score` is the SRC value.
