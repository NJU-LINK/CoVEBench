# Subjective MLLM-Checklist Metrics

This directory contains the CoVEBench subjective benchmark metrics. They are implemented separately from `metrics/vq/` and `metrics/vf/` because all four scores share the same MLLM judge, checklist schema, video inputs, and inference cache.

## Metrics

| Dimension | Metric | Column | Range | Direction |
| --- | --- | --- | --- | --- |
| Instruction Compliance | Union Accuracy | `UAS` | 0-100 by default | higher is better |
| Instruction Compliance | Instruction Following Score | `IFS` | 0-100 by default | higher is better |
| Instruction Compliance | Video Realism Score | `VRS` | 0-100 by default | higher is better |
| Video Fidelity | Semantic Consistency | `SEM` | 1-10 by default | higher is better |

`PhysicsLaws` and `AIArtifacts` prompts are not included here because they are not part of the released main benchmark scores.

## Definitions

For task `i`, let `Q_i^ifs` be instruction-following checklist questions, `Q_i^vrs` be realism checklist questions, and `Q_i^sem` be semantic-consistency questions.

Instruction Following Score:

```text
IFS_i = mean_{q in Q_i^ifs} Acc(q)
```

Video Realism Score:

```text
VRS_i = mean_{q in Q_i^vrs} Acc(q)
```

Union Accuracy:

```text
UAS_i = 1 if every q in Q_i^ifs union Q_i^vrs is correct, otherwise 0
```

Semantic Consistency:

```text
SEM_i = mean_{q in Q_i^sem} s_q, where s_q is an integer score in [1, 10]
```

The released model score is the mean over all evaluated tasks. The default CSV stores `UAS`, `IFS`, and `VRS` as percentages and `SEM` in its raw 1-10 scale. Use `--accuracy-scale unit` or `--sem-scale percent` if a different display scale is required.

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

The final user-facing CSV contains one row per task and only the four benchmark columns:

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

The aggregator first uses explicit metadata fields such as `metric`, `dimension`, `category`, `score_type`, `name`, `tag`, or `tags` to map questions to `IFS`, `VRS`, or `SEM`. If no metadata is available, `Score-MCQ` defaults to `SEM`, and other supported checklist questions default to `IFS`.
