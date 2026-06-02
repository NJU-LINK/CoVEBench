#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

BOOL_TRUE = {"yes", "true", "1", "correct"}
BOOL_FALSE = {"no", "false", "0", "incorrect"}


def load_id_list(path: str | None) -> set[str] | None:
    if not path:
        return None
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.add(value)
    return ids


def norm_text(value: Any) -> str:
    return str(value).strip().lower()


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_question(question: dict[str, Any]) -> float | None:
    answer = question.get("model_answer")
    q_type = question.get("type")
    if answer is None:
        return None

    if q_type == "Score-MCQ":
        return as_float(answer)

    expected = (
        question.get("expected_answer")
        or question.get("answer")
        or question.get("label")
        or question.get("ground_truth")
        or question.get("target_answer")
        or question.get("correct_answer")
    )
    if expected is None:
        options = question.get("options")
        if isinstance(options, dict):
            expected = options.get("answer") or options.get("correct") or options.get("target")
    if expected is None:
        return None

    pred = norm_text(answer)
    gold = norm_text(expected)
    if q_type in {"Single-TF", "Dual-TF"}:
        if pred in BOOL_TRUE | BOOL_FALSE and gold in BOOL_TRUE | BOOL_FALSE:
            return 1.0 if pred == gold else 0.0
    return 1.0 if pred == gold else 0.0


def category_text(group: dict[str, Any], question: dict[str, Any]) -> str:
    keys = [
        "metric",
        "dimension",
        "category",
        "score_type",
        "sub_category",
        "group",
        "name",
        "type",
        "tag",
        "tags",
        "question_type",
    ]
    pieces: list[str] = []
    for obj in (group, question):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, list):
                pieces.extend(str(x) for x in value)
            elif value is not None:
                pieces.append(str(value))
    return " ".join(pieces).lower()


def classify_question(group: dict[str, Any], question: dict[str, Any]) -> str | None:
    q_type = question.get("type")
    dimension = norm_text(question.get("dimension", ""))
    if dimension == "execution accuracy":
        return "IFS"
    if dimension == "physical logic":
        return "VRS"
    if dimension == "semantic preservation":
        return "SEM"

    text = category_text(group, question)
    compact = re.sub(r"[^a-z0-9]+", "_", text)

    if q_type == "Score-MCQ":
        return "SEM"

    if any(token in compact for token in ["sem", "semantic", "preservation", "consistency", "unedited", "unchanged"]):
        return "SEM"
    if any(token in compact for token in ["vrs", "realism", "quality", "plausibility", "artifact", "artifacts", "natural", "physics", "logic"]):
        return "VRS"
    if any(token in compact for token in ["ifs", "instruction", "execution", "following", "compliance", "accuracy", "edit_success"]):
        return "IFS"

    if q_type in {"AB-MCQ", "Single-TF", "Dual-TF"}:
        return "IFS"
    return None


def mean(values: list[float]) -> float | None:
    clean = [x for x in values if not math.isnan(x)]
    return None if not clean else sum(clean) / len(clean)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def scale_accuracy(value: float | None, mode: str) -> float | None:
    if value is None:
        return None
    return value * 100.0 if mode == "percent" else value


def scale_sem(value: float | None, mode: str) -> float | None:
    if value is None:
        return None
    return value * 10.0 if mode == "percent" else value


def aggregate_task(item: dict[str, Any], accuracy_scale: str, sem_scale: str) -> dict[str, str]:
    ifs_scores: list[float] = []
    vrs_scores: list[float] = []
    sem_scores: list[float] = []

    for group in item.get("evaluation_groups", []):
        for question in group.get("questions", []):
            metric = classify_question(group, question)
            score = score_question(question)
            if metric is None or score is None:
                continue
            if metric == "IFS":
                ifs_scores.append(score)
            elif metric == "VRS":
                vrs_scores.append(score)
            elif metric == "SEM":
                sem_scores.append(score)

    ifs = mean(ifs_scores)
    vrs = mean(vrs_scores)
    sem = mean(sem_scores)
    uas: float | None
    if ifs_scores and vrs_scores:
        uas = 1.0 if all(score == 1.0 for score in [*ifs_scores, *vrs_scores]) else 0.0
    else:
        uas = None

    return {
        "task_id": str(item.get("id", "")),
        "UAS": fmt(scale_accuracy(uas, accuracy_scale)),
        "IFS": fmt(scale_accuracy(ifs, accuracy_scale)),
        "VRS": fmt(scale_accuracy(vrs, accuracy_scale)),
        "SEM": fmt(scale_sem(sem, sem_scale)),
        "ifs_count": str(len(ifs_scores)),
        "vrs_count": str(len(vrs_scores)),
        "sem_count": str(len(sem_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluated CoVEBench checklist JSON into UAS/IFS/VRS/SEM CSV.")
    parser.add_argument("--input", required=True, help="checklist_evaluated.json")
    parser.add_argument("--output", required=True, help="CSV with task_id,UAS,IFS,VRS,SEM")
    parser.add_argument("--id-list")
    parser.add_argument("--accuracy-scale", choices=["percent", "unit"], default="percent")
    parser.add_argument("--sem-scale", choices=["raw", "percent"], default="raw")
    parser.add_argument("--debug-output", default="", help="Optional CSV including question counts for auditing.")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        checklist = json.load(f)
    id_filter = load_id_list(args.id_list)

    rows = []
    debug_rows = []
    for item in checklist:
        if id_filter is not None and str(item.get("id")) not in id_filter:
            continue
        row = aggregate_task(item, args.accuracy_scale, args.sem_scale)
        rows.append({key: row[key] for key in ["task_id", "UAS", "IFS", "VRS", "SEM"]})
        debug_rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "UAS", "IFS", "VRS", "SEM"])
        writer.writeheader()
        writer.writerows(rows)

    if args.debug_output:
        debug_output = Path(args.debug_output)
        debug_output.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id", "UAS", "IFS", "VRS", "SEM", "ifs_count", "vrs_count", "sem_count"])
            writer.writeheader()
            writer.writerows(debug_rows)

    print(f"Wrote {output} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
