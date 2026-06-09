# CoVEBench

> **CoVEBench: Can Video Editing Models Handle Complex Instructions?**

**NJU-LINK Team, Nanjing University · Kling Team, Kuaishou Technology**

<p align="center">
  <strong>Demo Video</strong>
</p>
https://github.com/NJU-LINK/CoVEBench/blob/main/docs/assets/videos/demo.mp4

**CoVEBench** is a diagnostic benchmark for compositional instruction-guided video editing. Unlike single-operation editing benchmarks, CoVEBench evaluates realistic multi-point instructions that require models to modify requested content while preserving unrelated source-video semantics and temporal coherence.
<p align="center">
  <!-- <a href="https://arxiv.org/abs/2606."><img src="https://img.shields.io/badge/arXiv-2606.-b31b1b.svg" alt="arXiv"></a> -->
  <a href="https://nju-link.github.io/CoVEBench/"><img src="https://img.shields.io/badge/Project-Page-4f7cba.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/NJU-LINK/CoVEBench"><img src="https://img.shields.io/badge/HuggingFace-CoVEBench-f6c343.svg" alt="TELBench"></a>
  <a href="https://github.com/NJU-LINK/CoVEBench"><img src="https://img.shields.io/badge/GitHub-Code-24292f.svg" alt="GitHub"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Docs-Usage-6a8f5f.svg" alt="Usage Docs"></a>
</p>

## Demo

<p align="center">
  <a href="docs/assets/videos/demo.mp4">
    <strong>Watch the demo video</strong>
  </a>
</p>

![CoVEBench overview](docs/assets/figures/overview.png)

## What We Evaluate

CoVEBench measures video editing performance across three complementary dimensions: instruction compliance, video quality, and video fidelity. The benchmark combines MLLM-checklist subjective metrics with objective quality and fidelity metrics.

![Evaluation matrix](docs/assets/tables/evaluation_matrix.png)

The released metrics are:

| Dimension | Metric | Column | Method |
| --- | --- | --- | --- |
| Instruction Compliance | Union Accuracy | `UAS` | MLLM + checklist |
| Instruction Compliance | Instruction Following Score | `IFS` | MLLM + checklist |
| Instruction Compliance | Video Realism Score | `VRS` | MLLM + checklist |
| Video Quality | Comprehensive Quality | `VQR` | VisualQuality-R1 |
| Video Quality | Aesthetics | `AES` | aesthetic-predictor-v2-5 |
| Video Quality | Motion Smoothness | `MSM` | edited-only optical flow |
| Video Quality | Technical Quality | `TQ` | DOVER++ technical branch |
| Video Fidelity | Semantic Consistency | `SEM` | MLLM + checklist |
| Video Fidelity | Structural Fidelity | `SSIM` | SSIM |
| Video Fidelity | Motion Fidelity | `MF` | CoTracker |
| Video Fidelity | Static Region Consistency | `SRC` | SAM2 + DINOv2 |

## Key Findings

- Current video editing models still struggle with compositional instructions: models often satisfy individual edit points but fail the strict union criterion.
- Editing strength and preservation are not automatically aligned: stronger modifications can unintentionally alter regions that should remain unchanged.
- Fine-grained checklist evaluation exposes failures that are hidden by coarse prompt-level or single-metric scoring.

See the full project page in [`docs/`](docs) for qualitative examples, main results, error analysis, and additional figures.

## Quick Start

Download the benchmark dataset from Hugging Face:

```bash
HF_ENDPOINT=https://hf-mirror.com uv run scripts/download_dataset.py
```

By default this downloads [`NJU-LINK/CoVEBench`](https://huggingface.co/datasets/NJU-LINK/CoVEBench) to:

```text
data/covebench_hf/
  checklist.json
  data/
    6.mp4
    19.mp4
    mixkit-....mp4
```

Use `--local-dir` if you want a different location:

```bash
uv run scripts/download_dataset.py --local-dir /path/to/covebench_hf
```

Place one model's edited/generated videos in a separate directory. Edited videos
must be named by checklist task id:

```text
data/
  my_model/
    1.mp4
    2.mp4
    3.mp4
```

For example, after your video editing model reads the checklist prompts and
source videos, save its outputs as `data/my_model/{id}.mp4`. The evaluation
scripts will only score ids that have both a checklist source video and a
generated video in `--edited-dir`.

`checklist.json` is the authoritative mapping from benchmark task id to the
source video. Each checklist item has an `id` and a `videoA_path`; the source
videos are resolved from `videoA_path` and do **not** need to be named by the
task id. Edited/generated videos, however, should be named by the checklist
task id, for example `data/my_model/1.mp4` for checklist item `"id": 1`.

When running objective metrics, `scripts/run_model.py` materializes a temporary
`source_by_task_id` directory under `--work-dir` using hardlinks, symlinks, or
copies, so pairwise metrics such as `SSIM`, `MF`, and `SRC` can compare
`id.mp4` source/edit pairs internally.

Install the objective evaluation environment and download objective weights:

```bash
bash scripts/setup_env.sh
HF_ENDPOINT=https://hf-mirror.com uv run scripts/download_weights.py
```

Run objective metrics for one model:

```bash
uv run scripts/run_model.py \
  --source-dir data/covebench_hf \
  --edited-dir data/my_model \
  --checklist data/covebench_hf/checklist.json \
  --output-csv outputs/my_model_scores.csv \
  --work-dir outputs/my_model_work \
  --metrics AES,TQ,MSM,SSIM,MF,VQR \
  --device cuda:0
```

This command runs the objective metrics except `SRC`. 
**Note:** The source-mask data required for `SRC` evaluation has not been provided yet, but will be released soon. Once available, you can run `SRC` by providing `--source-mask-root`, or you can run the SRC script directly with `--allow-llm` and `LLM_API_KEY` to generate the masks dynamically.

Run subjective MLLM-checklist metrics:

The released subjective judge is `Qwen/Qwen3.5-122B-A10B`. `scripts/run_subjective.py` uses this HF id by default; pass `--model-path` only when using a local checkpoint directory.

```bash
bash scripts/setup_env.sh --subjective
HF_ENDPOINT=https://hf-mirror.com uv run scripts/download_weights.py \
  --include-subjective \
  --subjective-model-repo Qwen/Qwen3.5-122B-A10B
uv run scripts/run_subjective.py \
  --source-dir data/covebench_hf \
  --edited-dir data/my_model \
  --checklist data/covebench_hf/checklist.json \
  --output-csv outputs/my_model_subjective.csv \
  --work-dir outputs/my_model_subjective_work \
  --tensor-parallel-size 2
```

If the judge model is already available locally, skip `--include-subjective` and pass the local checkpoint path, for example `--model-path weights/hf/subjective_judge`.

The released `checklist.json` stores source-video paths as relative placeholders such as `data/6.mp4`. The one-command runners use checklist `id` and `videoA_path` to resolve the source video from `--source-dir`, while edited videos are matched by numeric task id from `--edited-dir`. The subjective runner writes a materialized checklist under `--work-dir` with the actual paths used for evaluation.

## Outputs

Objective runner:

```csv
AES,TQ,MSM,SSIM,MF,VQR,SRC
...
```

Subjective runner:

```csv
UAS,IFS,VRS,SEM
...
```

The objective runner also writes a `_with_task_id.csv` file for task-level inspection. The subjective runner writes model-level aggregate scores. Both runners keep intermediate logs/caches under `--work-dir`.

## Repository Layout

```text
docs/                    # project webpage, figures, tables, paper PDF
metrics/vq/              # objective video-quality metrics
metrics/vf/              # objective video-fidelity metrics
metrics/subjective/      # MLLM-checklist subjective metrics
scripts/                 # setup, download, and one-command runners
src/covebench_eval/      # shared runner utilities
configs/                 # example config and metric filter lists
```

Metric-specific details:

- [`metrics/vq/`](metrics/vq): `AES`, `TQ`, `MSM`, `VQR`
- [`metrics/vf/`](metrics/vf): `SSIM`, `MF`, `SRC`
- [`metrics/subjective/`](metrics/subjective): `UAS`, `IFS`, `VRS`, `SEM`

## Reproducibility Notes

- Frame-level metrics use 10 equally spaced frames by default.
- `AES`, `VQR`, and `MSM` are computed on edited videos only.
- `SSIM`, `MF`, and `SRC` compare source and edited videos by checklist task id. The source side is resolved from checklist `videoA_path`; the edited side is expected to be named by task id.
- `SRC` can use a provided source-mask cache via `--source-mask-root`; otherwise run the SRC script directly with `--allow-llm` and `LLM_API_KEY`.
- `TQ` uses the DOVER++ technical score, not the overall DOVER++ score.
- Subjective `UAS`, `IFS`, `VRS`, and `SEM` are reported on a 0-100 scale by default, using global question-level aggregation.

## Citation

```bibtex
@misc{covebench2026,
  title        = {CoVEBench: A Diagnostic Benchmark for Compositional Instruction-Guided Video Editing},
  author       = {CoVEBench Team},
  year         = {2026},
  howpublished = {\url{https://github.com/NJU-LINK/CoVEBench}}
}
```
