# VF Reproduction

This folder includes `stage1_cache/`, a cache of the Gemini-based Stage 1
decisions and entity extraction. Use it to reproduce the VF masks without
rerunning Gemini.

## Inputs

- `checklist.json`
- `data/checklist.json`
- The source videos referenced by `checklist.json`
- `stage1_cache/`

## Run

```bash
cd metrics/vf/src

CUT_DATA_DIR=/path/to/video/data \
python prepare_stage1_cache.py

CUT_DATA_DIR=/path/to/video/data \
HF_ENDPOINT=https://hf-mirror.com \
LLM_API_KEY=dummy \
python src_pipeline.py --llm-workers 1
```

`prepare_stage1_cache.py` copies `stage1_cache/` to `outputs/` and converts
cached `completed` items to `ready_for_gpu`, so the pipeline skips Gemini and
reruns Grounding DINO + SAM2.

By default, the script reads `data/checklist.json` from the repository root. Set
`CHECKLIST_PATH=/path/to/checklist.json` to override it.

Generated directories such as `cut_outputs/`, `outputs/`, and `checkpoints/`
are intentionally ignored by Git.
