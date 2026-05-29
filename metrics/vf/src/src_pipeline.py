#!/usr/bin/env python3
"""
Semantic Preservation Mask Pipeline

Input : checklist.json  +  data/<video>.mp4
Output: outputs/{item_id}/
          frames/                    0001.jpg ...                       raw frames (SAM2-friendly numeric names)
          masks/                     mask_0000.png ...                  white = preserved-entity region
          masked_frames/             frame_0000.png ...                 RGB = preserved region, BLACK = everything else
          gdino_boxes_preview.jpg    one green box per grounded entity, red caption when no match
          metadata.json

Pipeline logic
  Stage 1A   skip if editing_instruction contains camera or style change
             (fast path: pre-annotated `category` field)
  Stage 1B   from evaluation_groups, take questions where dimension == "Semantic Preservation".
             LLM extracts the concrete physical entities those questions ask to preserve, as
             grounding-friendly NOUN PHRASES (e.g. "asian woman holding a microphone on the right").
             Abstract preservation targets (camera, lighting, etc.) are dropped.
  Stage 2A   ffmpeg @ FRAME_FPS → numeric-named JPEGs
  Stage 2B   Grounding DINO grounds each entity description directly to a single bbox.
             Probes the first 3 frames per entity to recover from occlusions and keeps the
             highest-scoring match. Descriptive phrases like "asian woman" let DINO target
             the correct subject without an extra VLM verification step.
  Stage 2C   SAM2 propagates each grounded box through the video
  Stage 2D   union all per-frame masks
  Save       per-frame mask PNG + masked_frame (RGB only inside preserved region)

Semantics
  mask=True  pixel that belongs to a preserved entity from a Semantic Preservation question
  mask=False everything else (the edited subject, abstract background, anything outside SP scope)
  masked_frame[~mask] = 0     pixels outside the preservation scope are blacked out
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from openai import OpenAI
except ImportError:  # OpenAI is only needed when --allow-llm generates missing source masks.
    OpenAI = None  # type: ignore[assignment]

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
CHECKLIST   = SCRIPT_DIR / "checklist.json"
# data/ lives as a sibling of cut/ on the server (parent/data, parent/cut).
# Override with CUT_DATA_DIR env var if you need a different layout.
DATA_DIR    = Path(os.environ.get("CUT_DATA_DIR", SCRIPT_DIR.parent / "data"))
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://app.ppapi.ai/v1")
LLM_MODEL    = os.environ.get("LLM_MODEL",    "gemini-3-flash-preview")
LLM_API_KEY  = os.environ.get("LLM_API_KEY")   # required — must come from env, never hard-code

FRAME_FPS              = 2

# Grounding DINO (HuggingFace transformers)
GDINO_MODEL_ID         = os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-base")
GDINO_BOX_THRESHOLD    = 0.30   # raw bbox confidence
GDINO_TEXT_THRESHOLD   = 0.25   # text-token alignment
GDINO_MIN_SCORE        = 0.25   # final acceptance — below this we treat the entity as not found
GDINO_PROBE_FRAMES     = 3      # frames to probe per entity (recover from occlusions)
GDINO_MAX_BOX_AREA     = 0.55   # reject single-entity boxes covering > 55% of the frame
                                # (almost always means the prompt pulled adjacent objects in)

# SAM2
SAM2_CFG               = "sam2_hiera_l"   # actual filename: sam2_hiera_l.yaml
SAM2_CKPT              = str(SCRIPT_DIR / "checkpoints" / "sam2_hiera_large.pt")

MAX_LLM_RETRY          = 3
LLM_SLEEP              = 0.4   # seconds between LLM calls (rate limiting)
LLM_LOG_PATH           = SCRIPT_DIR / "llm_log.jsonl"   # aggregated raw + parsed Gemini output
LLM_1A_MAX_TOKENS      = int(os.environ.get("LLM_1A_MAX_TOKENS", "2048"))
LLM_1B_MAX_TOKENS      = int(os.environ.get("LLM_1B_MAX_TOKENS", "4096"))
LLM_WORKERS            = int(os.environ.get("LLM_WORKERS", "8"))

# Concurrent writes from the ThreadPool to llm_log.jsonl must not interleave.
_LOG_LOCK = threading.Lock()

TERMINAL_STATUSES = {
    "completed",
    "skipped",
    "video_not_found",
    "no_sp_questions",
    "no_concrete_entities",
    "no_grounded_entities",
    "ready_for_gpu",
}
# ──────────────────────────────────────────────────────────────────────────────


# ── Stage 1A prompt ────────────────────────────────────────────────────────────
_1A_SYS = "You are a precise video editing instruction classifier."

_1A_USR = """\
Your task: decide whether this video editing instruction should be SKIPPED for \
downstream mask-based evaluation.

SKIP (skip=true) if the instruction includes ANY of the following — even when \
combined with other edits:
  • Camera movement: pan, zoom, tilt, orbit, dolly, tracking, crane, handheld \
shake, camera rotation, zoom-in / zoom-out, push-in / pull-out
  • Camera angle or framing change: bird's-eye view, low-angle, close-up reframe, \
switch perspective
  • Visual / cinematic style change: color grading, warm / cool tint, film grain, \
vignette, bokeh change, LUT application, anime style, oil-painting look, any \
aesthetic / photographic style transformation

DO NOT SKIP (skip=false) only if the instruction exclusively involves:
  • Replacing or modifying the background environment (no camera move)
  • Adding, removing, or replacing foreground objects / subjects
  • Changing object position, pose, or motion within the scene
  • Lighting that is an intrinsic part of an environment swap

Example output (format only):
{{"skip":true,"reason":"Instruction includes a camera pan."}}

Editing instruction:
{instruction}

Respond with ONLY the JSON object — no prose, no markdown, no code fences."""

# ── Stage 1B prompt ────────────────────────────────────────────────────────────
_1B_SYS = "You are a precise visual entity extractor for video editing evaluation."

_1B_USR = """\
You are given a list of Semantic Preservation questions for a video editing evaluation. \
Each question asks whether a specific element of the original video (Video A) is \
faithfully preserved in the edited video (Video B).

Your job: from these questions, extract the CONCRETE PHYSICAL ENTITIES (people / animals / \
objects) whose preservation can be verified by looking at a specific spatial region of the frame.

For each kept entity output an object with EXACTLY one key:
  "description" — a SHORT NOUN PHRASE that an open-vocabulary detector (Grounding DINO)
                  can directly ground in a frame.

==================================  STRICT RULES  ==================================
GRAMMAR
  • Lower-case English; no leading article needed; NO trailing period.
  • Max 10 words. NOUN PHRASE only — no verbs and no participles.
  • Allowed disambiguators (these become attributes of the noun, NOT separate
    nouns that pull the box):
        position : "on the left", "on the right", "in the center", "in the back"
        color    : "red dress", "in green blazer", "blonde"
        identity : "asian", "elderly", "young", "tall"
  • DO NOT mention any other object the subject interacts with — neither as a verb
    phrase ("holding a microphone") NOR as a prepositional noun ("with a microphone",
    "in front of a door"). Reason: Grounding DINO is phrase-grounding, so every noun
    in the prompt EXTENDS THE BOX to cover that object's region. Mentioning an
    adjacent object pulls everything between the subject and that object into the box.

SPLITTING
  • If a question is about a subject AND a separate object they hold / wear / touch,
    output them as TWO SEPARATE ENTITIES. Each gets its own minimal phrase.
    Example: "...the woman holding the microphone..." →
      {{"description": "asian woman on the right"}}
      {{"description": "microphone"}}

GOOD examples (will produce a tight, correct box):
  • "asian woman on the right"
  • "blonde woman on the left"
  • "young woman in red dress"
  • "small white dog"
  • "microphone"
  • "wooden door"

BAD examples (will produce a wrong, oversized box):
  • "asian woman holding a microphone"        (verb phrase: pulls in the microphone region)
  • "woman with a red rose"                   (adjacent noun: pulls in the rose region)
  • "young woman in front of a wooden door"   (adjacent noun: pulls the door in)
  • "the woman"                               (no disambiguator — fails when multiple women)
  • "How accurately is the woman preserved?"  (a question, not a phrase)
  • "the warm mood"                           (not physical)

DROP questions whose subject is abstract / non-localisable — do NOT emit them:
  • Camera (angle, framing, movement, perspective, composition, close-up, framing)
  • Lighting / atmosphere / mood / color grading / aesthetic / style / bokeh
  • Background environment as an abstract whole (unless a specific named object is asked about)
If a question mixes a concrete entity with an abstract property, keep only the entity.

==================================  WORKED EXAMPLE  ================================
Original description: "A young woman with a red rose stands in front of a wooden door, \
lit by warm light."
Semantic Preservation questions:
  1. How well is the identity and appearance of the young woman preserved?
  2. How accurately is the warm lighting and close-up framing preserved?
  3. How well is the red rose preserved?
Output:
  {{"preserved_entities":[
    {{"description":"young woman"}},
    {{"description":"red rose"}}
  ]}}
Notes:
  • Question 2 dropped — purely camera + lighting.
  • The rose is its OWN entity; we did NOT write "young woman holding a red rose"
    because that would pull the rose region into the woman's box.

==================================  YOUR TASK  =====================================
Original description (Video A):
{original_description}

Semantic Preservation questions:
{sp_questions}

Respond with ONLY the JSON object — no prose, no markdown, no code fences."""
# ──────────────────────────────────────────────────────────────────────────────


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm_client() -> OpenAI:
    if OpenAI is None:
        raise SystemExit("pip install openai, or run SRC with an existing --source-mask-root cache")
    if not LLM_API_KEY:
        raise SystemExit(
            "LLM_API_KEY env var is not set.\n"
            "  export LLM_API_KEY='sk-...'\n"
            f"  (current LLM_BASE_URL={LLM_BASE_URL!r}, LLM_MODEL={LLM_MODEL!r})"
        )
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


# Errors where retrying makes no sense — the key/permission/request is just wrong.
# We import lazily so a missing openai install doesn't break import order.
try:
    from openai import (
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        BadRequestError,
    )
    _NON_RETRYABLE_LLM_ERRORS: tuple[type[Exception], ...] = (
        AuthenticationError,      # 401  – wrong key
        PermissionDeniedError,    # 403  – key lacks access
        NotFoundError,            # 404  – wrong model name / base_url
        BadRequestError,          # 400  – malformed request, won't fix itself
    )
except ImportError:
    _NON_RETRYABLE_LLM_ERRORS = ()


def _call_llm(client: OpenAI, system: str, user: str, max_tokens: int = 1024) -> str:
    last: Exception | None = None
    for attempt in range(1, MAX_LLM_RETRY + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except _NON_RETRYABLE_LLM_ERRORS as e:
            # Don't burn 3 attempts on a wrong key / wrong model.
            raise RuntimeError(f"LLM non-retryable error ({type(e).__name__}): {e}") from e
        except Exception as e:
            last = e
            if attempt < MAX_LLM_RETRY:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"LLM failed after {MAX_LLM_RETRY} attempts: {last}")


def _log_llm_call(
    item_id: int,
    stage: str,
    system: str,
    user: str,
    raw_response: str,
    parsed: Any,
) -> None:
    """
    Append one record per LLM call to LLM_LOG_PATH (JSON-lines format).
    Logging failures must NEVER break the pipeline.
    """
    record = {
        "ts":           time.strftime("%Y-%m-%dT%H:%M:%S"),
        "item_id":      item_id,
        "stage":        stage,
        "system":       system,
        "user":         user,
        "raw_response": raw_response,
        "parsed":       parsed,
    }
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _LOG_LOCK, LLM_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _llm_call_logged(
    client: OpenAI,
    item_id: int,
    stage: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
) -> tuple[str, dict]:
    """
    Wrapper around `_call_llm` + `_parse_json` that always emits an entry to
    LLM_LOG_PATH — even when JSON parsing fails — so the raw model output is
    never lost. Returns (raw_response, parsed_dict).
    """
    raw = _call_llm(client, system, user, max_tokens=max_tokens)
    try:
        parsed = _parse_json(raw)
    except Exception as e:
        _log_llm_call(item_id, stage, system, user, raw, {"_parse_error": str(e)})
        raise
    _log_llm_call(item_id, stage, system, user, raw, parsed)
    return raw, parsed


def _parse_json(text: str) -> dict:
    """
    Robustly extract a JSON object from LLM output.
    Handles markdown fences, leading/trailing prose, trailing commas, balanced-brace extraction.
    """
    raw = text

    text = re.sub(r"```(?:json|JSON)?\s*", "", text)
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No '{{' found in LLM output. Raw:\n{raw[:500]}")

    depth, in_str, escape = 0, False, False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        recovered = _recover_truncated_preserved_entities(text[start:])
        if recovered is not None:
            return recovered
        recovered = _recover_truncated_skip_decision(text[start:])
        if recovered is not None:
            return recovered
        raise ValueError(f"Unbalanced braces in LLM output. Raw:\n{raw[:500]}")

    block = text[start:end]
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r",(\s*[}\]])", r"\1", block)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON parse failed after all repairs: {e}\n"
            f"Extracted block:\n{block[:1000]}\n"
            f"Raw output:\n{raw[:1000]}"
        )


def _recover_truncated_preserved_entities(block: str) -> dict | None:
    """
    Best-effort recovery for truncated Stage 1B output.
    If response is cut mid-stream, keep only fully closed entity objects:
      {"preserved_entities": [{"description": "..."} ...]}
    """
    m = re.search(r'"preserved_entities"\s*:\s*\[', block)
    if not m:
        return None

    arr = block[m.end():]
    entities: list[dict[str, str]] = []
    depth = 0
    in_str = False
    escape = False
    obj_start = -1

    for i, c in enumerate(arr):
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue

        if c == "{":
            if depth == 0:
                obj_start = i
            depth += 1
            continue
        if c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    obj_text = arr[obj_start:i + 1]
                    try:
                        obj = json.loads(obj_text)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        desc = (obj.get("description") or "").strip()
                        if desc:
                            entities.append({"description": desc})
            continue
        if c == "]" and depth == 0:
            break

    return {"preserved_entities": entities}


def _recover_truncated_skip_decision(block: str) -> dict | None:
    """
    Best-effort recovery for truncated Stage 1A output.
    Stage 1A only needs the boolean `skip`; a clipped reason should not fail the item.
    """
    m = re.search(r'"skip"\s*:\s*(true|false)', block, flags=re.IGNORECASE)
    if not m:
        return None

    skip = m.group(1).lower() == "true"
    reason = "Recovered from truncated LLM JSON; original reason was incomplete."
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', block)
    if reason_match:
        partial_reason = reason_match.group(1).strip()
        if partial_reason:
            reason = partial_reason + " [truncated]"

    return {
        "skip": skip,
        "reason": reason,
        "recovered_from_truncated_json": True,
    }


# ── Stage 1A ──────────────────────────────────────────────────────────────────

def _category_has_camera_or_style(item: dict) -> bool:
    return any(
        c.get("Level 1") in ("Camera", "Style")
        for c in item.get("category", [])
    )


def stage1a_skip(client: OpenAI, item: dict) -> dict:
    item_id = item["id"]

    if _category_has_camera_or_style(item):
        level1s = [c["Level 1"] for c in item.get("category", []) if c.get("Level 1") in ("Camera", "Style")]
        result = {
            "skip": True,
            "reason": f"Category field contains: {', '.join(set(level1s))}",
            "method": "category",
        }
        # Log even fast-path decisions so llm_log.jsonl has every Stage 1A outcome.
        _log_llm_call(item_id, "1A", "(category fast-path)",
                      item.get("editing_instruction", ""), "", result)
        return result

    user_msg = _1A_USR.format(instruction=item["editing_instruction"])
    _, parsed = _llm_call_logged(
        client, item_id, "1A", _1A_SYS, user_msg, max_tokens=LLM_1A_MAX_TOKENS
    )
    parsed["method"] = "llm"
    parsed.setdefault("skip", False)
    return parsed


# ── Stage 1B ──────────────────────────────────────────────────────────────────

def _extract_sp_questions(item: dict) -> list[str]:
    """Pull Semantic Preservation question texts from evaluation_groups."""
    questions: list[str] = []
    for group in item.get("evaluation_groups", []) or []:
        for q in group.get("questions", []) or []:
            if q.get("dimension") == "Semantic Preservation":
                qt = (q.get("question") or "").strip()
                if qt:
                    questions.append(qt)
    return questions


def stage1b_entities(client: OpenAI, item: dict) -> list[dict]:
    """
    Extract preserved_entities from Semantic Preservation questions.
    Returns list of {"description": <noun phrase>}, or [] when no SP questions exist or
    every question targets an abstract / non-localisable property.
    """
    item_id = item["id"]
    sp_questions = _extract_sp_questions(item)
    if not sp_questions:
        return []

    sp_text = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(sp_questions))
    user_msg = _1B_USR.format(
        original_description=item["original_description"],
        sp_questions=sp_text,
    )
    _, result = _llm_call_logged(
        client, item_id, "1B", _1B_SYS, user_msg, max_tokens=LLM_1B_MAX_TOKENS
    )
    entities = result.get("preserved_entities", [])

    cleaned: list[dict] = []
    for e in entities:
        if isinstance(e, str):
            desc = e.strip()
        elif isinstance(e, dict):
            desc = (e.get("description") or "").strip()
        else:
            continue
        if not desc:
            continue
        cleaned.append({"description": desc})
    return cleaned


# ── Stage 2A: frame extraction ────────────────────────────────────────────────

def extract_frames(video_path: Path, frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    # SAM2 requires filenames to be pure integers (it does int(stem) internally).
    out_pattern = str(frames_dir / "%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={FRAME_FPS}",
        "-q:v", "2",
        out_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode(errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed (exit {e.returncode}) on {video_path}\n--- stderr ---\n{stderr}"
        ) from e
    return sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))


# ── Stage 2B: Grounding DINO open-vocab grounding ─────────────────────────────
#
# Grounding DINO accepts free-form noun phrases as text prompts. For each entity
# description we issue ONE call per probe frame and keep the single highest-scoring
# detection. Descriptive phrases like "asian woman holding a microphone" already
# disambiguate the correct subject — no extra verification step is required.

def ground_entities(
    gdino_processor,
    gdino_model,
    frames: list[Path],
    entities: list[dict],
) -> list[dict[str, Any]]:
    """
    Returns one record per entity:
        {"entity_id": int (1-based),
         "description": str,
         "box": [x1,y1,x2,y2] | None,
         "score": float,
         "frame_idx": int | None}     # which probe frame produced the kept box
    """
    import inspect
    import torch

    # transformers renamed `box_threshold` → `threshold` around 4.51.
    # Pick the right kwarg once based on the installed version.
    sig = inspect.signature(gdino_processor.post_process_grounded_object_detection)
    box_kw = "threshold" if "threshold" in sig.parameters else "box_threshold"

    probes     = frames[: min(GDINO_PROBE_FRAMES, len(frames))]
    pil_images = [Image.open(p).convert("RGB") for p in probes]
    device     = next(gdino_model.parameters()).device

    records: list[dict[str, Any]] = []

    for ent_idx, ent in enumerate(entities, start=1):
        # Grounding DINO HF API expects lower-case and a trailing period.
        prompt = ent["description"].strip().lower().rstrip(".") + "."

        best_box:    list[float] | None = None
        best_score:  float = 0.0
        best_frame:  int | None = None

        for frame_i, img in enumerate(pil_images):
            inputs = gdino_processor(
                images=img, text=prompt, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = gdino_model(**inputs)

            postprocessed = gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[img.size[::-1]],   # (H, W)
                text_threshold=GDINO_TEXT_THRESHOLD,
                **{box_kw: GDINO_BOX_THRESHOLD},
            )[0]

            scores = postprocessed["scores"]
            boxes  = postprocessed["boxes"]
            if len(scores) == 0:
                continue

            top = int(scores.argmax().item())
            s   = float(scores[top].item())
            if s > best_score:
                best_score = s
                best_box   = boxes[top].cpu().numpy().tolist()
                best_frame = frame_i

        accepted = best_score >= GDINO_MIN_SCORE
        rec: dict[str, Any] = {
            "entity_id":   ent_idx,
            "description": ent["description"],
            "box":         best_box if accepted else None,
            "score":       round(best_score, 4),
            "frame_idx":   best_frame if accepted else None,
        }

        # Sanity check: a single-entity box that fills most of the frame almost
        # always means the prompt pulled adjacent objects in (e.g. "woman holding
        # a microphone" extends the box to cover the microphone + whatever else
        # is between them). Reject — it would corrupt the SAM2 mask.
        if rec["box"] is not None:
            img_w, img_h = pil_images[best_frame].size
            box_w  = max(0.0, best_box[2] - best_box[0])
            box_h  = max(0.0, best_box[3] - best_box[1])
            ratio  = (box_w * box_h) / float(img_w * img_h or 1)
            rec["box_area_ratio"] = round(ratio, 3)
            if ratio > GDINO_MAX_BOX_AREA:
                rec["box"]             = None
                rec["frame_idx"]       = None
                rec["rejected_reason"] = (
                    f"box covers {ratio:.0%} of frame (> {GDINO_MAX_BOX_AREA:.0%} threshold) — "
                    f"prompt likely pulled adjacent objects into the box"
                )

        records.append(rec)

    return records


# ── Preview overlay (debugging) ───────────────────────────────────────────────

def _draw_entity_boxes(frame_path: Path, records: list[dict[str, Any]]) -> bytes:
    """
    Render the first frame with one box per entity:
      • green   — entity matched (box drawn at its location)
      • red     — entity not matched (caption stacked in upper-left)
    Caption format: "<id>. <description[:48]>  (score=0.42)".
    """
    img = Image.open(frame_path).convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    font: Any = None
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(name, max(18, W // 60))
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    no_match_y = 10  # stack unmatched-entity captions in the upper-left
    for r in records:
        box     = r.get("box")
        matched = box is not None
        color   = (0, 200, 0) if matched else (220, 0, 0)
        label   = f"{r['entity_id']}. {r['description'][:48]}  (score={r['score']:.2f})"
        if not matched:
            label += "  NO MATCH"

        if matched:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(W - 1, x1));  x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1));  y2 = max(0, min(H - 1, y2))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            label_x, label_y = x1, y1
        else:
            label_x, label_y = 10, no_match_y
            no_match_y += 32

        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(label) * 9, 22
        bg_x2, bg_y2 = label_x + tw + 10, label_y + th + 8
        draw.rectangle([label_x, label_y, bg_x2, bg_y2], fill=(255, 255, 255))
        draw.text((label_x + 5, label_y + 3), label, fill=color, font=font)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


# ── Stage 2C: SAM2 propagation ────────────────────────────────────────────────

def sam2_propagate(
    sam2_predictor,
    frames_dir: Path,
    bboxes: list[list[float]],
) -> dict[int, np.ndarray]:
    """
    Propagates SAM2 masks from a flat list of bboxes through the full video.
    Each bbox becomes a separate SAM2 object; masks are unioned per frame.
    Returns {frame_idx: union_bool_mask [H, W]}.
    """
    import torch

    if not bboxes:
        return {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam2_predictor.init_state(video_path=str(frames_dir))
        sam2_predictor.reset_state(state)

        for obj_id, box in enumerate(bboxes, start=1):
            sam2_predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=obj_id,
                box=np.array(box, dtype=np.float32),
            )

        video_masks: dict[int, np.ndarray] = {}
        for frame_idx, obj_ids, mask_logits in sam2_predictor.propagate_in_video(state):
            per_obj = (mask_logits > 0.0).squeeze(1).cpu().numpy()  # [N, H, W]
            video_masks[frame_idx] = per_obj.any(axis=0)            # [H, W]

    return video_masks


# ── Stage 2D: save outputs ────────────────────────────────────────────────────

def save_outputs(
    item_dir: Path,
    frames: list[Path],
    video_masks: dict[int, np.ndarray],
) -> None:
    masks_dir = item_dir / "masks"
    mf_dir    = item_dir / "masked_frames"
    masks_dir.mkdir(exist_ok=True)
    mf_dir.mkdir(exist_ok=True)

    for idx, frame_path in enumerate(frames):
        frame_np = np.array(Image.open(frame_path).convert("RGB"))
        H, W = frame_np.shape[:2]

        mask = video_masks.get(idx, np.zeros((H, W), dtype=bool))

        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
            masks_dir / f"mask_{idx:04d}.png"
        )

        mf = frame_np.copy()
        mf[~mask] = 0
        Image.fromarray(mf).save(mf_dir / f"frame_{idx:04d}.png")


# ── Per-item orchestration ────────────────────────────────────────────────────

def _local_path(videoA_path: str) -> Path | None:
    p = DATA_DIR / Path(videoA_path).name
    return p if p.exists() else None


def run_stage1(
    item: dict,
    client: OpenAI,
) -> dict[str, Any]:
    item_id  = item["id"]
    item_dir = OUTPUTS_DIR / str(item_id)
    item_dir.mkdir(parents=True, exist_ok=True)
    meta_path = item_dir / "metadata.json"

    if meta_path.exists():
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached.get("status") in TERMINAL_STATUSES:
            return cached

    meta: dict[str, Any] = {
        "id":          item_id,
        "videoA_path": item["videoA_path"],
    }

    # ── resolve local video ───────────────────────────────────────────────────
    local = _local_path(item["videoA_path"])
    if local is None:
        meta.update(status="video_not_found", local_path=None)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta
    meta["local_path"] = str(local)

    # ── Stage 1A ──────────────────────────────────────────────────────────────
    skip_result = stage1a_skip(client, item)
    meta["stage1a"] = skip_result
    if skip_result["skip"]:
        meta["status"] = "skipped"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta
    time.sleep(LLM_SLEEP)

    # ── Stage 1B ──────────────────────────────────────────────────────────────
    sp_questions = _extract_sp_questions(item)
    meta["sp_questions"]       = sp_questions
    meta["sp_questions_count"] = len(sp_questions)

    if not sp_questions:
        meta.update(status="no_sp_questions",
                    error="evaluation_groups has no Semantic Preservation questions")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    entities = stage1b_entities(client, item)
    meta["preserved_entities"] = entities
    time.sleep(LLM_SLEEP)

    if not entities:
        meta.update(status="no_concrete_entities",
                    error="all SP questions are abstract (camera / lighting / etc.)")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    meta["status"] = "ready_for_gpu"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def run_stage2(
    item: dict,
    meta: dict[str, Any],
    gdino_processor,
    gdino_model,
    sam2_predictor,
) -> dict[str, Any]:
    item_id = item["id"]
    item_dir = OUTPUTS_DIR / str(item_id)
    item_dir.mkdir(parents=True, exist_ok=True)
    meta_path = item_dir / "metadata.json"

    local = Path(meta["local_path"])
    entities = meta.get("preserved_entities", [])

    # ── Stage 2A ──────────────────────────────────────────────────────────────
    frames_dir = item_dir / "frames"
    frames     = extract_frames(local, frames_dir)
    meta["num_frames"] = len(frames)
    meta["fps"]        = FRAME_FPS

    if not frames:
        meta.update(status="error", error="ffmpeg produced 0 frames")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    # ── Stage 2B : Grounding DINO ─────────────────────────────────────────────
    records       = ground_entities(gdino_processor, gdino_model, frames, entities)
    matched       = [r for r in records if r["box"] is not None]
    meta["gdino_records"]    = records
    meta["matched_entities"] = len(matched)

    # Always save preview so a human can inspect what DINO saw
    preview_bytes = _draw_entity_boxes(frames[0], records)
    (item_dir / "gdino_boxes_preview.jpg").write_bytes(preview_bytes)

    if not matched:
        meta.update(
            status="no_grounded_entities",
            error=f"Grounding DINO found no entity at score ≥ {GDINO_MIN_SCORE:.2f}",
        )
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    matched_boxes = [r["box"] for r in matched]

    # ── Stage 2C/2D : SAM2 propagation + union ────────────────────────────────
    video_masks = sam2_propagate(sam2_predictor, frames_dir, matched_boxes)
    save_outputs(item_dir, frames, video_masks)

    meta["status"] = "completed"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def process_item(
    item: dict,
    client: OpenAI,
    gdino_processor,
    gdino_model,
    sam2_predictor,
) -> dict[str, Any]:
    meta = run_stage1(item, client)
    if meta.get("status") != "ready_for_gpu":
        return meta
    return run_stage2(item, meta, gdino_processor, gdino_model, sam2_predictor)


# ── Model loading ─────────────────────────────────────────────────────────────

def _ensure_hf_reachable() -> None:
    """
    If HF_ENDPOINT is unset, probe huggingface.co. When it's unreachable
    (typical on China-region cloud servers, errno 99 / EADDRNOTAVAIL),
    silently fall back to hf-mirror.com so transformers can resolve weights.
    Must run before any `from transformers import ...`.
    """
    if os.environ.get("HF_ENDPOINT"):
        return
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
    except OSError:
        print("[!] huggingface.co unreachable — switching HF_ENDPOINT to https://hf-mirror.com",
              flush=True)
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_models():
    _ensure_hf_reachable()

    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        raise SystemExit("pip install 'transformers>=4.40'")

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        raise SystemExit("pip install 'git+https://github.com/facebookresearch/sam2.git'")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. This pipeline requires a GPU.")
    device = "cuda"

    print(f"Loading Grounding DINO ({GDINO_MODEL_ID}) ...", flush=True)
    gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    gdino_model     = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(device)
    gdino_model.eval()

    print("Loading SAM2 ...", flush=True)
    ckpt = Path(SAM2_CKPT)
    if not ckpt.exists():
        raise SystemExit(
            f"SAM2 checkpoint not found: {ckpt}\n"
            "Download from https://github.com/facebookresearch/sam2#model-checkpoints"
        )
    cfg_name = _resolve_sam2_config()
    print(f"  using config: {cfg_name}", flush=True)
    sam2 = build_sam2_video_predictor(cfg_name, str(ckpt), device=device)

    return gdino_processor, gdino_model, sam2


def _resolve_sam2_config() -> str:
    """
    Hydra searches configs in pkg://sam2 (package root).
    Depending on the installed version the yaml may live at:
      sam2/sam2_hiera_large.yaml          → hydra name: "sam2_hiera_large"
      sam2/configs/sam2_hiera_large.yaml  → hydra name: "configs/sam2_hiera_large"
    Also handles sam2.1 naming (PyPI package ships SAM2.1 configs).
    """
    import sam2 as _sam2_pkg

    sam2_dir = os.path.dirname(_sam2_pkg.__file__)

    candidates = [
        ("sam2_hiera_l",                         "sam2_hiera_l.yaml"),
        ("sam2_hiera_large",                     "sam2_hiera_large.yaml"),
        ("configs/sam2/sam2_hiera_l",            os.path.join("configs", "sam2", "sam2_hiera_l.yaml")),
        ("configs/sam2/sam2_hiera_large",        os.path.join("configs", "sam2", "sam2_hiera_large.yaml")),
        ("configs/sam2.1/sam2.1_hiera_l",        os.path.join("configs", "sam2.1", "sam2.1_hiera_l.yaml")),
        ("configs/sam2.1/sam2.1_hiera_large",    os.path.join("configs", "sam2.1", "sam2.1_hiera_large.yaml")),
    ]
    for hydra_name, rel_path in candidates:
        if os.path.exists(os.path.join(sam2_dir, rel_path)):
            return hydra_name

    found_yamls: list[str] = []
    for root, _, files in os.walk(sam2_dir):
        for f in files:
            if f.endswith(".yaml"):
                found_yamls.append(os.path.relpath(os.path.join(root, f), sam2_dir))

    raise SystemExit(
        "SAM2 config not found in package.\n"
        + ("YAML files found:\n" + "\n".join(f"  {y}" for y in found_yamls)
           if found_yamls else "  No YAML files found at all — SAM2 install may be broken.")
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Preservation Mask Pipeline")
    parser.add_argument("--ids",   nargs="+", type=int, help="Run only specific item IDs")
    parser.add_argument("--limit", type=int,            help="Process at most N items")
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=LLM_WORKERS,
        help="Parallel workers for Stage 1 LLM calls",
    )
    args = parser.parse_args()

    items: list[dict] = json.loads(CHECKLIST.read_text(encoding="utf-8"))
    if args.ids:
        items = [it for it in items if it["id"] in set(args.ids)]
    if args.limit:
        items = items[: args.limit]

    OUTPUTS_DIR.mkdir(exist_ok=True)
    gdino_processor, gdino_model, sam2 = load_models()

    counts: dict[str, int] = {}
    # Stage 1: LLM-heavy, run in parallel.
    stage1_results: dict[int, dict[str, Any]] = {}
    max_workers = max(1, args.llm_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_item = {
            ex.submit(run_stage1, item, _llm_client()): item
            for item in items
        }
        done = 0
        for fut in as_completed(fut_to_item):
            item = fut_to_item[fut]
            item_id = item["id"]
            done += 1
            label = f"[{done}/{len(items)}] id={item_id}"
            try:
                meta = fut.result()
                status = meta.get("status", "error")
                stage1_results[item_id] = meta
            except Exception as exc:
                status = "error"
                stage1_results[item_id] = {
                    "id": item_id,
                    "videoA_path": item["videoA_path"],
                    "status": "error",
                    "error": str(exc),
                }
                print(f"{label}  EXCEPTION: {exc}")
            print(f"{label}  {status}", flush=True)

    # Stage 2: GPU-heavy, keep serial to avoid OOM and contention.
    for idx, item in enumerate(items, 1):
        item_id = item["id"]
        label = f"[GPU {idx}/{len(items)}] id={item_id}"
        stage1_meta = stage1_results[item_id]
        status = stage1_meta.get("status", "error")
        if status != "ready_for_gpu":
            counts[status] = counts.get(status, 0) + 1
            print(f"{label}  {status}", flush=True)
            continue
        print(label, end="  ", flush=True)
        try:
            meta = run_stage2(item, stage1_meta, gdino_processor, gdino_model, sam2)
            status = meta.get("status", "error")
        except Exception as exc:
            status = "error"
            print(f"EXCEPTION: {exc}")
        counts[status] = counts.get(status, 0) + 1
        print(status)

    print("\n── Summary ──────────────────────")
    total = sum(counts.values())
    for k, v in sorted(counts.items()):
        print(f"  {k:<22} {v:>4}  ({v/total*100:.0f}%)")
    print(f"  {'total':<22} {total:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
