# Subjective MLLM-Checklist Metrics

This directory contains the CoVEBench subjective benchmark metrics. They are implemented separately from `metrics/vq/` and `metrics/vf/` because all four scores share the same MLLM judge, checklist schema, video inputs, and inference cache.

## Metrics

| Dimension | Metric | Column | Range | Direction |
| --- | --- | --- | --- | --- |
| Instruction Compliance | Union Accuracy | `UAS` | 0-100 by default | higher is better |
| Instruction Compliance | Instruction Following Score | `IFS` | 0-100 by default | higher is better |
| Instruction Compliance | Video Realism Score | `VRS` | 0-100 by default | higher is better |
| Video Fidelity | Semantic Consistency | `SEM` | 0-100 by default | higher is better |

`PhysicsLaws` and `AIArtifacts` prompts are not included here because they are not part of the released main benchmark scores.

## Definitions

The released aggregate scores are computed globally over all evaluated checklist questions, matching the historical CoVEBench scoring script.

Instruction Following Score:

```text
IFS = correct Execution Accuracy questions / total Execution Accuracy questions
```

Video Realism Score:

```text
VRS = correct Physical Logic questions / total Physical Logic questions
```

Union Accuracy:

```text
UAS = groups with all Execution Accuracy and Physical Logic questions correct / evaluated objective groups
```

Semantic Consistency:

```text
SEM = mean_{q in Semantic Preservation} (s_q * 10), where s_q is the judge score
```

The default CSV stores all four columns on a 0-100 scale. Use `--accuracy-scale unit` or `--sem-scale raw` if a different display scale is required.

## Run One Model

Install the subjective environment and download or provide a judge model:

```bash
bash scripts/setup_env.sh --subjective
HF_ENDPOINT=https://hf-mirror.com uv run scripts/download_weights.py \
  --include-subjective \
  --subjective-model-repo <released-qwen-judge-repo>
```

```bash
uv run scripts/run_subjective.py \
  --source-dir data/source \
  --edited-dir data/my_model \
  --checklist data/checklist.json \
  --model-path weights/hf/subjective_judge \
  --output-csv outputs/my_model_subjective.csv \
  --work-dir outputs/my_model_subjective_work \
  --tensor-parallel-size 2
```

Replace `<released-qwen-judge-repo>` with the exact released CoVEBench judge model. If your released judge is already available locally, skip model download and pass that path to `--model-path`.

The final user-facing CSV contains one aggregate row and only the four benchmark columns:

```csv
UAS,IFS,VRS,SEM
...
```

A sanity-check copy with task ids is also written:

```text
outputs/my_model_subjective_with_task_id.csv
```

Intermediate files are kept under `--work-dir`:

```text
checklist.json
checklist_evaluated.json
processed_q_ids.txt
failed_evaluation_log.txt
judge.log
aggregate.log
```

## Aggregate Existing Judgments

If `checklist_evaluated.json` already exists, skip MLLM inference and only rebuild the CSV:

```bash
uv run scripts/run_subjective.py \
  --checklist data/checklist.json \
  --output-csv outputs/my_model_subjective.csv \
  --work-dir outputs/my_model_subjective_work \
  --aggregate-only
```

## Checklist Schema

Each task should contain `id`, `videoA_path`, `videoB_path`, `editing_instruction`, and `evaluation_groups`. Each group contains `questions`. Supported question types are:

| Question Type | Used For | Input Videos |
| --- | --- | --- |
| `AB-MCQ` | Instruction or realism checklist accuracy | edited video |
| `Single-TF` | Instruction or realism checklist accuracy | edited video |
| `Dual-TF` | Instruction or realism checklist accuracy | source and edited videos |
| `Score-MCQ` | Semantic consistency | source and edited videos |

The aggregator first uses the checklist `dimension` field: `Execution Accuracy` maps to `IFS`, `Physical Logic` maps to `VRS`, and `Semantic Preservation` maps to `SEM`. If no dimension is available, it falls back to metadata keywords; `Score-MCQ` defaults to `SEM`, and other supported checklist questions default to `IFS`.
