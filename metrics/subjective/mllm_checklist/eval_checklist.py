#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

DEFAULT_SUBJECTIVE_MODEL = "Qwen/Qwen3.5-122B-A10B"

ARRAY_OUTPUT_TYPES = {"AB-MCQ", "Single-TF", "Dual-TF", "Score-MCQ"}
PROMPT_FILES = {
    "AB-MCQ": "AB-MCQ.txt",
    "Single-TF": "Single-TF.txt",
    "Dual-TF": "Dual-TF.txt",
    "Score-MCQ": "Score-MCQ.txt",
}


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


def get_absolute_path(base_path: Path, video_path: str | None) -> str:
    if not video_path:
        return ""
    path = Path(video_path)
    if path.is_absolute():
        return str(path.resolve())
    return str((base_path / path).resolve())


def common_path(paths: list[str]) -> str:
    existing = [str(Path(path).resolve()) for path in paths if path and Path(path).exists()]
    if not existing:
        return str(Path.cwd())
    try:
        return os.path.commonpath(existing)
    except ValueError:
        return "/"


def load_processed_records(record_file: Path) -> set[str]:
    processed: set[str] = set()
    if record_file.exists():
        with open(record_file, encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if value:
                    processed.add(value)
    return processed


def save_processed_record(record_file: Path, video_id: Any, question_id: Any) -> None:
    with open(record_file, "a", encoding="utf-8") as f:
        f.write(f"{video_id}_{question_id}\n")
        f.flush()


def load_prompts(prompt_dir: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for question_type, filename in PROMPT_FILES.items():
        path = prompt_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"System prompt file not found: {path}")
        prompts[question_type] = path.read_text(encoding="utf-8").strip()
    return prompts


def extract_json_from_text(text: str, question_type: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    match_md = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match_md:
        return match_md.group(1).strip()
    match_arr = re.search(r"(\[.*\])", text, re.DOTALL)
    if match_arr:
        return match_arr.group(0).strip()
    match_obj = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_obj:
        return match_obj.group(0).strip()
    return text.strip()


def extract_answer_from_dict(item: dict[str, Any]) -> Any:
    for key in ["final_answer", "final_score", "answer", "score"]:
        if key in item and item[key] is not None:
            return item[key]
    return None


def validate_llm_output(json_data: Any, expected_ids: list[Any]) -> tuple[bool, str]:
    if not isinstance(json_data, list):
        return False, "Output is not a JSON array."

    result_ids = []
    for item in json_data:
        if not isinstance(item, dict):
            return False, "Item in array is not an object."
        q_id = str(item.get("id", ""))
        if not q_id:
            return False, "Missing id in one answer."
        if "reasoning" not in item:
            return False, f"Missing reasoning for ID {q_id}."
        if extract_answer_from_dict(item) is None:
            return False, f"Missing final answer/score for ID {q_id}."
        result_ids.append(q_id)

    expected_str_ids = [str(x) for x in expected_ids]
    missing = [x for x in expected_str_ids if x not in result_ids]
    if missing:
        return False, f"Missing answers for IDs: {missing}"
    return True, "Valid"


def atomic_save_json(data: Any, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def build_task_messages(task: dict[str, Any]) -> list[dict[str, Any]]:
    q_type = task["q_type"]
    system_prompt = task["prompt_text"]
    content_list: list[dict[str, Any]] = []

    if q_type in {"Dual-TF", "Score-MCQ"}:
        content_list.extend(
            [
                {"type": "text", "text": "Video A (Original Source):\n"},
                {"type": "video_url", "video_url": {"url": f"file://{task['video_a_abs']}"}},
                {"type": "text", "text": "\nVideo B (Edited/Generated):\n"},
                {"type": "video_url", "video_url": {"url": f"file://{task['video_b_abs']}"}},
            ]
        )
    else:
        content_list.extend(
            [
                {"type": "text", "text": "Video for evaluation:\n"},
                {"type": "video_url", "video_url": {"url": f"file://{task['video_b_abs']}"}},
            ]
        )

    json_qs = json.dumps(task["llm_input_qs"], indent=2, ensure_ascii=False)
    instruction = task.get("editing_instruction") or ""
    if instruction:
        user_text = f"\nEditing Instruction:\n{instruction}\n\n"
    else:
        user_text = "\n"
    user_text += (
        f"Here are the questions you need to evaluate based on the video(s):\n{json_qs}\n\n"
        "CRITICAL: Output STRICTLY as a valid JSON array. "
        "Do NOT include any markdown formatting or explanations outside the JSON."
    )
    content_list.append({"type": "text", "text": user_text})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content_list}]


def process_single_task_with_retry(llm: Any, sampling_params: Any, task: dict[str, Any], max_retries: int) -> tuple[bool, Any, str | None]:
    messages = [build_task_messages(task)]
    expected_ids = [q["id"] for q in task["llm_input_qs"]]
    for attempt in range(1, max_retries + 1):
        error_msg = ""
        try:
            outputs = llm.chat(messages=messages, sampling_params=sampling_params)
            raw_text = outputs[0].outputs[0].text
            parsed_json = json.loads(extract_json_from_text(raw_text, task["q_type"]))
            valid, message = validate_llm_output(parsed_json, expected_ids)
            if valid:
                return True, parsed_json, None
            error_msg = f"Validation failed: {message}"
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
        print(
            f"    [Retry {attempt}/{max_retries}] Failed for video {task['video_id']} "
            f"type {task['q_type']}: {error_msg}",
            flush=True,
        )
    return False, None, error_msg


def build_pending_tasks(
    checklist: list[dict[str, Any]],
    prompts: dict[str, str],
    processed_records: set[str],
    base_path: Path,
    id_filter: set[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    pending_tasks: list[dict[str, Any]] = []
    media_paths: list[str] = []

    for video_item in checklist:
        video_id = video_item.get("id")
        if id_filter is not None and str(video_id) not in id_filter:
            continue

        video_a_abs = get_absolute_path(base_path, video_item.get("videoA_path"))
        video_b_abs = get_absolute_path(base_path, video_item.get("videoB_path"))
        video_a_valid = Path(video_a_abs).is_file() if video_a_abs else False
        video_b_valid = Path(video_b_abs).is_file() if video_b_abs else False
        if video_a_valid:
            media_paths.append(video_a_abs)
        if video_b_valid:
            media_paths.append(video_b_abs)

        type_to_questions = {key: [] for key in ARRAY_OUTPUT_TYPES}
        for group in video_item.get("evaluation_groups", []):
            for question in group.get("questions", []):
                question_type = question.get("type")
                question_id = question.get("id")
                if question_type not in type_to_questions:
                    continue
                if f"{video_id}_{question_id}" in processed_records:
                    continue
                if question_type in {"Dual-TF", "Score-MCQ"} and not (video_a_valid and video_b_valid):
                    print(f"WARNING: missing source or edited video for task {video_id}, type {question_type}; skipped.", flush=True)
                    continue
                if question_type not in {"Dual-TF", "Score-MCQ"} and not video_b_valid:
                    print(f"WARNING: missing edited video for task {video_id}, type {question_type}; skipped.", flush=True)
                    continue
                type_to_questions[question_type].append(question)

        for question_type, original_refs in type_to_questions.items():
            if not original_refs:
                continue
            llm_input_questions = []
            for question in original_refs:
                clean_question = {"id": question["id"], "question": question["question"]}
                if "options" in question:
                    clean_question["options"] = question["options"]
                llm_input_questions.append(clean_question)
            pending_tasks.append(
                {
                    "video_id": video_id,
                    "video_a_abs": video_a_abs,
                    "video_b_abs": video_b_abs,
                    "q_type": question_type,
                    "prompt_text": prompts[question_type],
                    "editing_instruction": video_item.get("editing_instruction", ""),
                    "original_qs_refs": original_refs,
                    "llm_input_qs": llm_input_questions,
                }
            )

    return pending_tasks, media_paths


def write_answers(task: dict[str, Any], parsed_json: list[dict[str, Any]], record_file: Path, processed_records: set[str]) -> None:
    for answer_item in parsed_json:
        answer_id = str(answer_item["id"])
        for original_question in task["original_qs_refs"]:
            if str(original_question["id"]) == answer_id:
                original_question["model_reasoning"] = answer_item.get("reasoning")
                original_question["model_answer"] = extract_answer_from_dict(answer_item)
                save_processed_record(record_file, task["video_id"], answer_id)
                processed_records.add(f"{task['video_id']}_{answer_id}")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CoVEBench MLLM checklist questions with vLLM.")
    parser.add_argument("--base-path", required=True, help="Directory containing checklist.json and evaluation outputs.")
    parser.add_argument(
        "--model-path",
        required=True,
        help=f"Path or HF id of the Qwen video MLLM served through vLLM. Released judge: {DEFAULT_SUBJECTIVE_MODEL}.",
    )
    parser.add_argument("--prompt-dir", default="", help="Prompt directory. Defaults to ./prompts next to this script.")
    parser.add_argument("--input-json", default="checklist.json")
    parser.add_argument("--output-json", default="checklist_evaluated.json")
    parser.add_argument("--processed-record", default="processed_q_ids.txt")
    parser.add_argument("--failed-log", default="failed_evaluation_log.txt")
    parser.add_argument("--id-list")
    parser.add_argument("--allowed-media-dir", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit-mm-video", type=int, default=2)
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    prompt_dir = Path(args.prompt_dir).resolve() if args.prompt_dir else Path(__file__).resolve().parent / "prompts"
    input_json = base_path / args.input_json
    output_json = base_path / args.output_json
    record_file = base_path / args.processed_record
    failed_log = base_path / args.failed_log
    target_json = output_json if output_json.exists() else input_json
    if not target_json.exists():
        raise SystemExit(f"Input checklist not found: {target_json}")

    with open(target_json, encoding="utf-8") as f:
        checklist = json.load(f)

    prompts = load_prompts(prompt_dir)
    processed_records = load_processed_records(record_file)
    id_filter = load_id_list(args.id_list)
    pending_tasks, media_paths = build_pending_tasks(checklist, prompts, processed_records, base_path, id_filter)
    print(f"Task parsing complete. Built {len(pending_tasks)} categorized task batches.", flush=True)
    if not pending_tasks:
        print("All questions have been evaluated. No inference needed.", flush=True)
        atomic_save_json(checklist, output_json)
        return

    import torch
    from vllm import LLM, SamplingParams

    allowed_media_dir = args.allowed_media_dir or common_path(media_paths)
    print(f"Initializing vLLM model TP={args.tensor_parallel_size}, allowed_media_dir={allowed_media_dir}", flush=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"video": args.limit_mm_video},
        allowed_local_media_path=allowed_media_dir,
    )
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16384)

    try:
        total_batches = (len(pending_tasks) + args.batch_size - 1) // args.batch_size
        for batch_start in range(0, len(pending_tasks), args.batch_size):
            batch_tasks = pending_tasks[batch_start : batch_start + args.batch_size]
            print(f"Starting inference batch {batch_start // args.batch_size + 1}/{total_batches} tasks={len(batch_tasks)}", flush=True)
            batch_messages = [build_task_messages(task) for task in batch_tasks]
            failed_tasks: list[dict[str, Any]] = []

            try:
                outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)
                for output, task in zip(outputs, batch_tasks):
                    try:
                        raw_text = output.outputs[0].text
                        parsed_json = json.loads(extract_json_from_text(raw_text, task["q_type"]))
                        expected_ids = [question["id"] for question in task["llm_input_qs"]]
                        valid, error_msg = validate_llm_output(parsed_json, expected_ids)
                        if valid:
                            write_answers(task, parsed_json, record_file, processed_records)
                        else:
                            print(f"  Validation failed task={task['video_id']} type={task['q_type']}: {error_msg}", flush=True)
                            failed_tasks.append(task)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  Parsing error task={task['video_id']} type={task['q_type']}: {exc}", flush=True)
                        failed_tasks.append(task)
            except Exception as exc:  # noqa: BLE001
                print(f"  vLLM batch execution error: {exc}", flush=True)
                failed_tasks = batch_tasks

            for task in failed_tasks:
                success, parsed_json, error = process_single_task_with_retry(llm, sampling_params, task, args.max_retries)
                if success:
                    write_answers(task, parsed_json, record_file, processed_records)
                    print(f"    Retry succeeded task={task['video_id']} type={task['q_type']}", flush=True)
                else:
                    print(f"    Task permanently failed task={task['video_id']} type={task['q_type']} error={error}", flush=True)
                    with open(failed_log, "a", encoding="utf-8") as f:
                        f.write(f"Video:{task['video_id']} | Type:{task['q_type']} | Err:{error}\n")

            atomic_save_json(checklist, output_json)
            print(f"  Progress saved to {output_json}", flush=True)
    finally:
        if "llm" in locals():
            del llm
        torch.cuda.empty_cache()
        print(f"Evaluation finished. Final results saved to {output_json}", flush=True)


if __name__ == "__main__":
    main()
