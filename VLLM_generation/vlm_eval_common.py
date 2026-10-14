#!/usr/bin/env python3
"""Shared logic for the VID-DRIFT video-LLM evaluations over downsample.csv.

Each model script (*_generation.py) only defines an adapter class with
`__init__(args)` and `answer(video_path, prompt) -> str`. Manifest filtering,
clean/degraded/restored path resolution, prompts, scoring, resume and summaries
live here, so results from different models are directly comparable. Use one
--out per model.

Variants: every non-clean condition (e.g. downscale_4) is evaluated as
  - degraded : the raw downsample   downsamples/{ds}/{cond}/file   (`degraded_path`)
  - restored : the restorer output  restored/{ds}/{cond}/{UPSCALER}/file
(select with --variants). Clean rows are evaluated once. Each output row carries
`variant` and `upscaler` columns. Row keys for clean/restored rows are unchanged
from the restored-only version, so existing results resume; only the degraded
rows are new work.

Report across models (MAcc per condition x variant, plus restoration transitions):
    python vid_drift_common.py results/qwen.csv results/llava.csv [--save transitions.csv]
  RIE     = clean correct, degraded correct,   restored incorrect  (restoration-induced error)
  RIC     = clean correct, degraded incorrect, restored correct    (restoration-induced correction)
  RIE_any / RIC_any = the same degraded->restored flips regardless of the clean outcome
  p_mcnemar = exact two-sided McNemar test on RIE_any vs RIC_any (degraded vs restored)

--protocol vstat reproduces the VSTAT lmms-eval task
(github.com/vision-x-nyu/vstat, lmms_eval/tasks/vstat/{vstat.yaml,utils.py}):
  - prompt: "Watch the full video carefully before answering.\\n\\nQuestion: {q}"
    + "Please answer with the letter (A, B, C, or D)." for MCQ, or the per-task
    numeric instruction (needs a `source_task` column), + VSTAT's post-prompt
  - MCQ options are expected inside the question text ("(A) ... (B) ..."), as in
    the VSTAT json; if a row has none there but has `candidates`, they are appended
  - MCQ: last standalone A-D letter in the upper-cased output; numeric: last
    integer; open-ended: exact lower-cased match
  - numeric score = MRA over thresholds .50:.95:.05 with rel_err <= 1 - t
  - score column = MCQ 0/1 or numeric MRA; ALL_Score_avg = mean over ALL questions
    (what the paper reports as "Avg."), MCQ_ACC (= MAcc) / Numeric_MRA per question type
  - greedy decoding, max_new_tokens 16384
--protocol generic keeps the original qwen_generation.py prompt/scoring.

Short videos (fps_* / fpsabs_* conditions can leave fewer frames than --nframes, even 1):
  - qwen_vl_utils adapters: run() calls patch_qwen_vl_utils(), which caps nframes to what each
    video has (any adapter, any entry format) and stops the torchvision fallback hiding errors
  - frame-based adapters: sample_frames() already returns all frames when the video is short;
    pass min_frames= / multiple_of= if the model needs a minimum or even frame count
  - --dry-run reports, per condition/variant, how many eval videos are shorter than --nframes
"""
from __future__ import annotations

import os
import sys
import ctypes

# Must happen before anything imports qwen_vl_utils (its backend choice is lru_cached).
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")

# Env workaround from the original script: preload the env's libstdc++ so torchcodec loads.
_LIBSTDCXX = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
if os.path.isfile(_LIBSTDCXX):
    ctypes.CDLL(_LIBSTDCXX, mode=ctypes.RTLD_GLOBAL)

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ---------------- easily changeable defaults ----------------
DATASETS = ["MVBench"]                                        # which `dataset` values to keep
# CONDITIONS = ["clean", "downscale_4", "downscale_12", "downscale_16", "jpeg_40", "jpeg_10", "jpeg_5", "h265_28", "h265_45", "h265_51", "streamproxy_x4_crf35", "streamproxy_x8_crf45"]  # method_amount combos ("clean" = reference)
CONDITIONS = ["clean", "fps_quarter", "fps_eighth", "fpsabs_1fps", "fpsabs_0_5fps"]
VARIANTS = ["degraded", "restored"]                            # evaluated for every non-clean condition
UPSCALER = "EDEN"                                           # restored/{ds}/{cond}/{UPSCALER}/file
DOWNSAMPLES_DIR = "downsamples"                             # replaced by RESTORED_DIR for eval
RESTORED_DIR = "restored"
MRA_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]  # generic protocol
NUMERIC_REL_TOL = 0.10                                      # generic protocol
# -------------------------------------------------------------

OUT_FIELDS = ["key", "dataset", "condition", "variant", "upscaler", "eval_video", "video_ok",
              "question", "type", "gold", "raw_output", "pred", "correct", "mra", "parse_ok",
              "model", "item_id", "score"]
_NONE_UPSCALERS = ("none", "degraded", "")
_VARIANT_RANK = {"clean": 0, "degraded": 1, "restored": 2}


# ---------------- path resolution ----------------

def condition_of(row: dict) -> str:
    m = (row.get("method") or "").strip()
    a = (row.get("amount") or "").strip()
    return m if m == "clean" or not a else f"{m}_{a}"


def resolve_eval_path(row: dict, upscaler: str) -> str:
    if condition_of(row) == "clean":
        return (row.get("video_path") or "").strip()
    dpath = (row.get("degraded_path") or "").strip()
    if not dpath or upscaler.lower() in _NONE_UPSCALERS:
        return dpath
    p = Path(dpath.replace(f"/{DOWNSAMPLES_DIR}/", f"/{RESTORED_DIR}/"))
    return str(p.parent / upscaler / p.name)


def eval_units(row: dict, variants: list[str], upscaler: str) -> list[tuple[str, str, str]]:
    """(variant, upscaler_label, eval_path) for every video to evaluate for this manifest row:
    one 'clean' unit for clean rows, one unit per requested variant otherwise."""
    if condition_of(row) == "clean":
        return [("clean", "", resolve_eval_path(row, upscaler))]
    units = []
    for v in variants:
        if v == "degraded":
            units.append(("degraded", "", resolve_eval_path(row, "none")))
        elif v == "restored" and upscaler.lower() not in _NONE_UPSCALERS:
            units.append(("restored", upscaler, resolve_eval_path(row, upscaler)))
    return units


def infer_variant(r: dict) -> str:
    """Variant of an output row; inferred from eval_video for files written before the column existed."""
    if r.get("variant"):
        return r["variant"]
    if r.get("condition") == "clean":
        return "clean"
    return "restored" if f"/{RESTORED_DIR}/" in (r.get("eval_video") or "") else "degraded"


def infer_upscaler(r: dict) -> str:
    if r.get("upscaler"):
        return r["upscaler"]
    return Path(r.get("eval_video") or "").parent.name if infer_variant(r) == "restored" else ""


def _migrate_out(path: str) -> None:
    """Rewrite an existing results CSV to the current OUT_FIELDS (adds variant/upscaler) so
    appended rows line up with the header. The original is kept as <path>.bak."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if header == OUT_FIELDS:
            return
        rows = list(reader)
    for r in rows:
        r["variant"] = infer_variant(r)
        r["upscaler"] = infer_upscaler(r)
    shutil.copy2(path, path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    print(f"Migrated {path} to the variant-aware schema ({len(rows)} rows; backup at {path}.bak)")


# ---------------- shared parsing helpers ----------------

_LETTER_RE = re.compile(r"\b([A-H])\b")
_LEADING_RE = re.compile(r"^\s*(?:answer\s*[:\-]?\s*)?\(?([A-H])\)?[\.\):]?", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_EMBEDDED_OPT_RE = re.compile(r"\(([A-H])\)")
_MCQ_TYPES = {"mcq", "multiple_choice", "multiple-choice", "mc", "choice"}


def parse_candidates(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return [str(v) for v in val] if isinstance(val, list) else []
    except json.JSONDecodeError:
        return [s.strip() for s in raw.split("|") if s.strip()]


def gold_letter(answer: str, options: list[str]) -> str | None:
    """Gold may be a letter ('B') or the option text; normalize to a letter."""
    a = (answer or "").strip()
    if len(a) == 1 and a.upper() in string.ascii_uppercase[:len(options) or 8]:
        return a.upper()
    low = [o.strip().lower() for o in options]
    if a.lower() in low:
        return string.ascii_uppercase[low.index(a.lower())]
    # option text may embed the letter: "B. cat"
    m = _LEADING_RE.match(a)
    if m:
        return m.group(1).upper()
    return None


# ---------------- VSTAT protocol (lmms_eval/tasks/vstat) ----------------

VSTAT_PRE_PROMPT = ""
VSTAT_MCQ_POST_PROMPT = "Do not response anything other than a single letter from the given choices!"
VSTAT_NA_POST_PROMPT = "Do not response anything other than a single number!"
VSTAT_INTEGER_PATTERN = re.compile(r"-?\d+")
VSTAT_MCQ_LETTER_PATTERN = re.compile(r"\b([A-D])\b")
VSTAT_NUMERICAL_INSTRUCTIONS = {
    "book": "Return only a single integer (positive, negative, or 0).\nAnswer:",
    "packing_order_green": "Return only the final count as a single integer.\nAnswer:",
    "packing_order_blue": "Return only the final count as a single integer.\nAnswer:",
    "packing_order_yellow": "Return only the final count as a single integer.\nAnswer:",
    "packing_order_chopsticks": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_diamond": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_heart": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_club": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_spade": "Return only the final count as a single integer.\nAnswer:",
}


def _vstat_mcq_instruction(n_choices: int) -> str:
    letters = "ABCD"[:n_choices]
    if len(letters) < 2:
        return ""
    return f"Please answer with the letter ({', '.join(letters[:-1])}, or {letters[-1]})."


def _vstat_kind(row: dict) -> str:
    """VSTAT: MCQ iff the item has choices; otherwise numeric iff the answer is an int."""
    t = (row.get("type") or "").strip().lower()
    if parse_candidates(row.get("candidates", "")) or t in _MCQ_TYPES:
        return "mcq"
    try:
        int(str(row.get("answer", "")).strip())
        return "numeric"
    except ValueError:
        return "open"


def _vstat_question(row: dict) -> tuple[str, int, bool]:
    """Question text, number of choices, and whether options had to be appended."""
    q = (row.get("question") or "").strip()
    choices = parse_candidates(row.get("candidates", ""))
    embedded = sorted(set(_EMBEDDED_OPT_RE.findall(q)))
    if embedded:
        return q, len(choices) or len(embedded), False
    if choices:  # VSTAT json embeds "(A) ... (B) ..." in the question; rebuild it if stripped
        opts = "\n".join(f"({string.ascii_uppercase[i]}) {c}" for i, c in enumerate(choices))
        return f"{q}\n{opts}", len(choices), True
    return q, 4, False


def _vstat_prompt(row: dict) -> str:
    kind = _vstat_kind(row)
    q, n_choices, _ = _vstat_question(row) if kind == "mcq" else ((row.get("question") or "").strip(), 0, False)
    body = f"Watch the full video carefully before answering.\n\nQuestion: {q}"
    if kind == "mcq":
        instruction = _vstat_mcq_instruction(n_choices)
        if instruction:
            body += f"\n\n{instruction}"
        return f"{VSTAT_PRE_PROMPT}{body}\n\n{VSTAT_MCQ_POST_PROMPT}"
    instruction = VSTAT_NUMERICAL_INSTRUCTIONS.get(str(row.get("source_task", "")).strip())
    if instruction:
        body += f"\n\n{instruction}"
    return f"{VSTAT_PRE_PROMPT}{body}\n\n{VSTAT_NA_POST_PROMPT}"


def _abs_dist_norm(pred, target):
    if target == 0:
        return 0.0 if pred == target else float("inf")
    return abs(pred - target) / abs(target)


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05) -> float:
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = _abs_dist_norm(pred, target) <= 1 - conf_intervs
    return float(accuracy.mean())


def _vstat_numeric_mra(pred, target) -> float:
    if pred is None:
        return 0.0
    try:
        return mean_relative_accuracy(pred, target, 0.5, 0.95, 0.05)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _vstat_score(row: dict, raw_output: str) -> dict:
    kind = _vstat_kind(row)
    prediction = str(raw_output).strip()
    answer = str(row.get("answer", "")).strip()
    if kind == "mcq":
        gold = gold_letter(answer, parse_candidates(row.get("candidates", "")))
        matches = VSTAT_MCQ_LETTER_PATTERN.findall(prediction.upper())
        pred = matches[-1] if matches else None
        correct = pred is not None and pred == gold
        return {"gold": gold, "pred": pred, "parse_ok": pred is not None, "correct": correct,
                "mra": "", "score": float(correct)}
    if kind == "numeric":
        target = int(answer)
        ints = [int(m) for m in VSTAT_INTEGER_PATTERN.findall(prediction)]
        pred = ints[-1] if ints else None
        mra = _vstat_numeric_mra(pred, target)
        return {"gold": target, "pred": pred, "parse_ok": pred is not None,
                "correct": pred is not None and pred == target, "mra": round(mra, 4), "score": mra}
    pred = prediction.lower()
    correct = pred == answer.lower()
    return {"gold": answer, "pred": pred, "parse_ok": bool(pred), "correct": correct,
            "mra": "", "score": float(correct)}


# ---------------- generic protocol (original qwen_generation.py) ----------------

def extract_choice(text: str, options: list[str]) -> str | None:
    if not text:
        return None
    t = text.strip()
    m = _LEADING_RE.match(t)
    if m:
        return m.group(1).upper()
    m = _LETTER_RE.search(t)
    if m:
        return m.group(1).upper()
    low = t.lower()
    hits = [i for i, o in enumerate(options) if o and o.lower() in low]
    return string.ascii_uppercase[hits[0]] if len(hits) == 1 else None


def extract_number(text: str) -> float | None:
    if not text:
        return None
    m = _NUM_RE.search(text.replace(",", ""))
    try:
        return float(m.group()) if m else None
    except ValueError:
        return None


def _norm(s: str) -> str:
    s = s.lower().strip().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s)


def _generic_kind(row: dict) -> str:
    t = (row.get("type") or "mcq").strip().lower()
    return t if t in ("mcq", "numeric") else "open"


def _generic_score(row: dict, raw_output: str) -> dict:
    qtype = _generic_kind(row)
    answer = (row.get("answer") or "").strip()
    if qtype == "mcq":
        options = parse_candidates(row.get("candidates", ""))
        gold = gold_letter(answer, options)
        pred = extract_choice(raw_output, options)
        correct = pred is not None and gold is not None and pred == gold
        return {"gold": gold, "pred": pred, "parse_ok": pred is not None,
                "correct": correct, "mra": "", "score": float(correct)}
    if qtype == "numeric":
        try:
            gold_v = float(answer.replace(",", ""))
        except ValueError:
            return {"gold": answer, "pred": None, "parse_ok": False, "correct": False, "mra": 0.0, "score": 0.0}
        pred = extract_number(raw_output)
        if pred is None:
            return {"gold": gold_v, "pred": None, "parse_ok": False, "correct": False, "mra": 0.0, "score": 0.0}
        rel = abs(pred - gold_v) / max(abs(gold_v), 1e-9)
        mra = sum(1.0 for t in MRA_THRESHOLDS if rel < 1.0 - t) / len(MRA_THRESHOLDS)
        return {"gold": gold_v, "pred": pred, "parse_ok": True,
                "correct": rel <= NUMERIC_REL_TOL, "mra": round(mra, 3), "score": mra}
    # open-ended: normalized containment
    p, g = _norm(raw_output), _norm(answer)
    correct = bool(g) and (g == p or g in p or p in g)
    return {"gold": answer, "pred": raw_output.strip(), "parse_ok": bool(p),
            "correct": correct, "mra": "", "score": float(correct)}


def _generic_prompt(row: dict) -> str:
    q = (row.get("question") or "").strip()
    qtype = _generic_kind(row)
    if qtype == "mcq":
        options = parse_candidates(row.get("candidates", ""))
        lettered = "\n".join(f"{string.ascii_uppercase[i]}. {o}" for i, o in enumerate(options))
        return f"{q}\n{lettered}\nAnswer with the single letter of the correct option only."
    if qtype == "numeric":
        return f"{q}\nAnswer with a single number only."
    return f"{q}\nAnswer concisely."


# ---------------- protocol dispatch ----------------

def kind_of(row: dict, protocol: str = "vstat") -> str:
    return _vstat_kind(row) if protocol == "vstat" else _generic_kind(row)


def build_prompt(row: dict, protocol: str = "vstat") -> str:
    return _vstat_prompt(row) if protocol == "vstat" else _generic_prompt(row)


def score_row(row: dict, raw_output: str, protocol: str = "vstat") -> dict:
    return _vstat_score(row, raw_output) if protocol == "vstat" else _generic_score(row, raw_output)


def row_key(row: dict, model: str = "", upscaler: str = "", nframes: int = 0, protocol: str = "vstat") -> str:
    """Pass the restorer name for restored rows and "none" for degraded rows (clean ignores it),
    which keeps keys identical to earlier runs made with --upscaler X / --upscaler none."""
    cond = condition_of(row)
    payload = "|".join([row.get("dataset", ""), cond, row.get("video_path", ""),
                        row.get("question", ""), model,
                        "" if cond == "clean" else upscaler, str(nframes), protocol])
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def item_id(row: dict) -> str:
    """Condition-independent id for one QA item (pairs conditions and models)."""
    payload = "|".join([row.get("dataset", ""), row.get("video_path", ""), row.get("question", "")])
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


# ---------------- shared frame sampling / loading helpers ----------------

QWEN_FRAME_FACTOR = 2  # qwen_vl_utils FRAME_FACTOR: nframes must be even and <= the video's frame count


def video_info(video_path: str) -> tuple[int, float]:
    """(frame count, average fps) from torchcodec metadata, i.e. what qwen_vl_utils'
    torchcodec reader sees. Frame-dropped videos (fps_* / fpsabs_*) can have very few frames."""
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(video_path)
    total = int(dec.metadata.num_frames or len(dec))
    fps = float(dec.metadata.average_fps or 0.0)
    return total, fps


def fit_nframes(video_path: str, nframes: int, factor: int = QWEN_FRAME_FACTOR) -> int:
    """Frame budget qwen_vl_utils accepts for this video: `nframes` rounded the way
    qwen_vl_utils rounds it, capped at the largest multiple of `factor` <= the video's
    frame count. Identical to the old behavior whenever the video is long enough (so clean
    and restored rows are unaffected). Returns 0 if the video has fewer than `factor` frames."""
    total, _ = video_info(video_path)
    want = max(factor, round(nframes / factor) * factor)
    return min(want, (total // factor) * factor)


def _decode_all(dec):
    """Decode frames one by one until the decoder runs out. Used when the container
    metadata overstates the frame count (seen on re-muxed, frame-dropped mp4s)."""
    out = []
    i = 0
    while True:
        try:
            out.append(dec.get_frame_at(i).data)
        except (IndexError, RuntimeError, ValueError):
            break
        i += 1
    if not out:
        raise RuntimeError("no decodable frames")
    import torch
    return torch.stack(out)


def sample_frames(video_path: str, nframes: int, max_side: int = 0, min_frames: int = 1,
                  multiple_of: int = 1):
    """Uniformly sample `nframes` over the WHOLE video with the same index rule as
    qwen_vl_utils (linspace(0, total-1, n).round()). Returns uint8 numpy (T, H, W, C).
    If the video has fewer than `nframes` frames, returns all of them.

    For models with frame-count constraints, `min_frames` / `multiple_of` pad the result by
    repeating the last frame (as qwen_vl_utils does for frame lists). Neither changes anything
    for videos with at least `nframes` frames when `nframes` already satisfies them."""
    import torch
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(video_path)
    total = dec.metadata.num_frames or len(dec)
    n = max(1, min(nframes, total))
    idx = torch.linspace(0, total - 1, n).round().long().tolist()
    try:
        frames = dec.get_frames_at(indices=idx).data  # (T, C, H, W) uint8
    except (IndexError, RuntimeError, ValueError):
        allf = _decode_all(dec)
        n = max(1, min(nframes, len(allf)))
        frames = allf[torch.linspace(0, len(allf) - 1, n).round().long()]
    target = max(len(frames), min_frames)
    if multiple_of > 1:
        target = -(-target // multiple_of) * multiple_of
    if target > len(frames):
        frames = torch.cat([frames, frames[-1:].expand(target - len(frames), -1, -1, -1)])
    h, w = frames.shape[-2:]
    if max_side and max(h, w) > max_side:
        import torch.nn.functional as F
        s = max_side / max(h, w)
        frames = F.interpolate(frames.float(), size=(max(1, round(h * s)), max(1, round(w * s))),
                               mode="bicubic", antialias=True, align_corners=False)
        frames = frames.clamp(0, 255).to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous().numpy()


def to_pil(frames) -> list:
    from PIL import Image
    return [Image.fromarray(f) for f in frames]


_QWEN_PATCHED = False


def patch_qwen_vl_utils() -> None:
    """Make qwen_vl_utils work on frame-dropped videos, for EVERY adapter that uses it,
    however the adapter builds its {"type": "video", "video": path, "nframes": N} entry.
    Called by run() before the adapter loads; no-op if qwen_vl_utils is not installed.

    1. smart_nframes: when the video has fewer frames than the requested `nframes`, use the
       largest even count it has instead of raising "nframes should in interval [2, T]".
       A 1-frame video gets 2 (its only frame, twice). Unchanged for videos long enough, so
       clean/restored inputs and row keys are identical to before.
    2. The torchvision fallback (torchvision.io.read_video, removed in recent torchvision)
       re-raises the original torchcodec error instead of hiding it."""
    global _QWEN_PATCHED
    if _QWEN_PATCHED:
        return
    try:
        from qwen_vl_utils import vision_process as vp
    except ImportError:
        return
    _QWEN_PATCHED = True
    factor = getattr(vp, "FRAME_FACTOR", QWEN_FRAME_FACTOR)
    orig_smart_nframes = vp.smart_nframes

    def smart_nframes(ele, total_frames, video_fps):
        cap = max(factor, (int(total_frames) // factor) * factor)
        if "nframes" in ele:
            want = max(factor, round(ele["nframes"] / factor) * factor)
            return min(want, cap)
        try:
            return orig_smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        except ValueError:  # fps-based sampling on a very short video
            return cap

    vp.smart_nframes = smart_nframes  # the readers look it up as a module global

    try:
        import torchvision.io as tvio
        has_read_video = hasattr(tvio, "read_video")
    except ImportError:
        has_read_video = False
    if not has_read_video and "torchvision" in getattr(vp, "VIDEO_READER_BACKENDS", {}):
        def _no_torchvision(ele):
            err = sys.exc_info()[1]  # called inside qwen_vl_utils' except block: the torchcodec error
            raise RuntimeError(f"torchcodec read failed (torchvision fallback unavailable): "
                               f"{type(err).__name__}: {err}")
        vp.VIDEO_READER_BACKENDS["torchvision"] = _no_torchvision
    print("Patched qwen_vl_utils: nframes capped to each video's frame count"
          + ("" if has_read_video else "; torchvision fallback re-raises the torchcodec error"))


def qwen_video_entry(video_path: str, nframes: int, **extra) -> dict:
    """{"type": "video", "video": path, "nframes": nframes, **extra} for qwen_vl_utils, with
    patch_qwen_vl_utils() applied, so short videos are handled. Adapters that build the
    entry themselves are covered too, since run() applies the same patch."""
    patch_qwen_vl_utils()
    return {"type": "video", "video": video_path, "nframes": nframes, **extra}


def dtype_kwarg(torch_dtype) -> dict:
    """`dtype=` on transformers>=4.56, `torch_dtype=` before."""
    import transformers
    from packaging.version import Version
    new = Version(transformers.__version__.split("+")[0]) >= Version("4.56")
    return {"dtype": torch_dtype} if new else {"torch_dtype": torch_dtype}


def move_inputs(inputs, device):
    """Move a processor output (BatchFeature or dict) to `device`, leaving non-tensors alone."""
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(inputs).items()}


# ---------------- main loop ----------------

def build_parser(default_model: str, default_out: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../DS_csv/downsample.csv")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--protocol", default="generic", choices=["vstat", "generic"],
                    help="vstat = VSTAT lmms-eval prompts/scoring; generic = original script's")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS,
                    help="method_amount combos, e.g. clean downscale_4 jpeg_40")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=["degraded", "restored"],
                    help="for each non-clean condition: the raw downsample under downsamples/ "
                         "and/or the --upscaler output under restored/")
    ap.add_argument("--upscaler", default=UPSCALER,
                    help='restorer folder name under restored/; "none" = degraded variant only')
    ap.add_argument("--nframes", type=int, default=16,
                    help="frame budget, sampled uniformly over the WHOLE video (not per second)")
    ap.add_argument("--max-side", type=int, default=0,
                    help="frame-based adapters: resize so longest side <= this (0 = native)")
    ap.add_argument("--max-new-tokens", type=int, default=16384, help="VSTAT uses 16384")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (0 = all)")
    ap.add_argument("--show", type=int, default=3,
                    help="print prompt / raw output / gold for the first N evaluated rows")
    ap.add_argument("--dry-run", action="store_true", help="resolve paths and count only, no model")
    ap.add_argument("--all-items", action="store_true",
                    help="also run items whose eval video is missing for some condition/variant "
                         "(default: only items complete across all --conditions x --variants)")
    return ap


def _manifest_checks(rows: list[dict], columns: list[str], protocol: str) -> None:
    kinds = Counter(kind_of(r, protocol) for r in rows)
    print(f"Question kinds ({protocol} protocol): {dict(kinds)}")
    if protocol == "vstat":
        other = sorted({r.get("dataset", "") for r in rows} - {"vstat"})
        if other:
            print(f"NOTE: VSTAT prompts/scoring also applied to datasets {other}; "
                  f"use --protocol generic for those if that is not intended.")
        mcq = [r for r in rows if _vstat_kind(r) == "mcq"]
        appended = sum(1 for r in mcq if _vstat_question(r)[2])
        no_opts = sum(1 for r in mcq if not _EMBEDDED_OPT_RE.search(r.get("question") or "")
                      and not parse_candidates(r.get("candidates", "")))
        if appended:
            print(f"NOTE: {appended} MCQ rows had no '(A) ...' options in the question; "
                  f"appended them from `candidates`.")
        if no_opts:
            print(f"WARNING: {no_opts} MCQ rows have no options in the question or `candidates`.")
        n_num = sum(1 for r in rows if _vstat_kind(r) == "numeric")
        if n_num and "source_task" not in columns:
            print(f"NOTE: no `source_task` column, so the per-task numeric instruction VSTAT adds "
                  f"for book / packing_order_* / showing_card_count_* is omitted ({n_num} numeric rows).")
    else:
        unknown = {t: n for t, n in kinds.items() if t not in ("mcq", "numeric")}
        if unknown:
            print(f"WARNING: these types use the open-ended branch: {unknown}")
        no_opts = sum(1 for r in rows if _generic_kind(r) == "mcq"
                      and not parse_candidates(r.get("candidates", "")))
        if no_opts:
            print(f"WARNING: {no_opts} MCQ rows have no parseable `candidates`; their prompt lists "
                  f"NO options. Manifest columns: {columns}")


def run(adapter_cls, default_model: str, default_out: str, add_args=None) -> None:
    ap = build_parser(default_model, default_out)
    if add_args:
        add_args(ap)
    args = ap.parse_args()

    want_ds = {d.lower() for d in args.datasets}
    want_cond = set(args.conditions)
    variants = list(dict.fromkeys(args.variants))
    if args.upscaler.lower() in _NONE_UPSCALERS and "restored" in variants:
        print(f"NOTE: --upscaler {args.upscaler!r} means no restorer; evaluating the degraded variant only.")
        variants = [v for v in variants if v != "restored"]
    want_units = {("clean", "clean")} if "clean" in want_cond else set()
    want_units |= {(c, v) for c in want_cond - {"clean"} for v in variants}
    if "clean" not in want_cond:
        print("NOTE: 'clean' is not in --conditions; restoration transitions need clean results "
              "for the same items (from this or an earlier run in the same --out).")

    # load manifest and filter
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for row in reader:
            if row.get("dataset", "").lower() not in want_ds:
                continue
            if condition_of(row) not in want_cond:
                continue
            rows.append(row)
    print(f"Manifest rows after filter (datasets={sorted(want_ds)}, "
          f"conditions={sorted(want_cond)}): {len(rows)}")
    _manifest_checks(rows, columns, args.protocol)

    # manifests without explicit clean rows: build one per item from its `video_path`
    # (same row_key as a real clean row, so earlier clean results still resume)
    if "clean" in want_cond and not any(condition_of(r) == "clean" for r in rows):
        synth = {}
        for r in rows:
            synth.setdefault(item_id(r), {**r, "method": "clean", "amount": "", "degraded_path": ""})
        rows += list(synth.values())
        print(f"NOTE: manifest has no clean rows; built {len(synth)} from each item's video_path.")

    # resume: skip rows already scored successfully (errors / missing videos are retried)
    done = set()
    if os.path.isfile(args.out):
        _migrate_out(args.out)
        with open(args.out, newline="", encoding="utf-8") as f:
            done = {r["key"] for r in csv.DictReader(f)
                    if r.get("video_ok") == "True" and r.get("raw_output")
                    and not r["raw_output"].startswith("__ERROR__")}
        print(f"Resume: {len(done)} rows already scored in {args.out}")

    # resolve every (row, variant) eval video once
    resolved = []
    for row in rows:
        for variant, up, ev in eval_units(row, variants, args.upscaler):
            resolved.append((row, variant, up, ev, bool(ev) and os.path.isfile(ev)))

    # diagnostics: folders checked per condition/variant, and how many videos exist in each
    by_unit = defaultdict(Counter)
    example_missing = {}
    for row, variant, _, ev, ok in resolved:
        u = f"{condition_of(row)}/{variant}"
        folder = str(Path(ev).parent) if ev else "<empty path>"
        by_unit[(u, folder)][ok] += 1
        if not ok:
            example_missing.setdefault(u, ev or "<empty path>")
    print("Folders checked (unit | folder | found / missing):")
    for (u, folder), c in sorted(by_unit.items()):
        print(f"  {u:<24} {folder}  {c[True]} / {c[False]}")
    for u, ev in sorted(example_missing.items()):
        print(f"  e.g. missing {u}: {ev}")

    # complete-items filter: keep only items whose eval video exists for EVERY requested
    # condition x variant, so every evaluated item is paired across all of them
    if not args.all_items:
        have = defaultdict(set)
        for row, variant, _, _, ok in resolved:
            if ok:
                have[item_id(row)].add((condition_of(row), variant))
        all_ids = {item_id(t[0]) for t in resolved}
        complete = {i for i in all_ids if want_units <= have.get(i, set())}
        missing = Counter(f"{c}/{v}" for i in all_ids - complete for c, v in want_units - have.get(i, set()))
        print(f"Complete items (eval video exists for all {len(want_units)} condition/variant units): "
              f"{len(complete)} / {len(all_ids)}")
        if missing:
            print(f"  skipped {len(all_ids) - len(complete)} incomplete items; missing videos per unit: "
                  f"{dict(missing.most_common())}")
        if not complete:
            print("WARNING: no item has videos for every condition/variant; check --conditions / "
                  "--variants / --upscaler, or pass --all-items.")
        resolved = [t for t in resolved if item_id(t[0]) in complete]

    todo = []
    n_missing = 0
    for row, variant, up, ev, ok in resolved:
        key = row_key(row, args.model, up if variant == "restored" else "none", args.nframes, args.protocol)
        if key in done:
            continue
        if not ok:
            n_missing += 1
        todo.append((key, row, variant, up, ev, ok))
    # item-major order: every condition/variant of an item runs back to back, so --limit
    # yields complete paired items instead of e.g. only the first condition's rows
    first_seen: dict[str, int] = {}
    for t in todo:
        first_seen.setdefault(item_id(t[1]), len(first_seen))
    cond_rank = {c: i for i, c in enumerate(args.conditions)}
    todo.sort(key=lambda t: (first_seen[item_id(t[1])],
                             cond_rank.get(condition_of(t[1]), len(cond_rank)), _VARIANT_RANK[t[2]]))
    per_unit = Counter(f"{condition_of(t[1])}/{t[2]}" for t in todo)
    print(f"To evaluate: {len(todo)} (missing eval videos: {n_missing}) {dict(per_unit)}")

    if args.dry_run:
        try:  # frame counts per unit: frame-dropped videos can be shorter than --nframes
            short, tiny, n_seen = Counter(), Counter(), Counter()
            for _, row, variant, _, ev, ok in todo:
                if not ok:
                    continue
                u = f"{condition_of(row)}/{variant}"
                total, _ = video_info(ev)
                n_seen[u] += 1
                short[u] += total < args.nframes
                tiny[u] += total < QWEN_FRAME_FACTOR
            print(f"Frame counts (videos with < --nframes={args.nframes} frames | < {QWEN_FRAME_FACTOR} frames):")
            for u in sorted(n_seen):
                print(f"  {u:<24} {short[u]} / {n_seen[u]}  |  {tiny[u]}")
        except Exception as e:
            print(f"(frame-count check skipped: {type(e).__name__}: {e})")
        for key, row, variant, up, ev, ok in todo[:15]:
            print(f"  [{condition_of(row):>14}/{variant:<8}] ok={ok} {ev}")
        if todo:
            print("\nExample prompt:\n" + build_prompt(todo[0][1], args.protocol))
        return

    if not any(t[5] for t in todo):
        summarize(args.out) if os.path.isfile(args.out) else None
        return

    patch_qwen_vl_utils()  # short (frame-dropped) videos, for any adapter using qwen_vl_utils
    print(f"Loading {args.model} ...", flush=True)
    model = adapter_cls(args)

    new_file = not os.path.isfile(args.out)
    n_done = n_shown = 0
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for key, row, variant, up, ev, ok in todo:
            cond = condition_of(row)
            base = {"key": key, "dataset": row["dataset"], "condition": cond, "variant": variant,
                    "upscaler": up, "eval_video": ev, "video_ok": ok, "question": row["question"],
                    "type": kind_of(row, args.protocol), "model": args.model, "item_id": item_id(row)}
            if not ok:
                w.writerow({**base, "gold": row.get("answer", ""), "raw_output": "", "pred": "",
                            "correct": "", "mra": "", "parse_ok": "", "score": ""})
                f.flush()
                continue
            prompt = build_prompt(row, args.protocol)
            try:
                raw = model.answer(ev, prompt)
            except Exception as e:  # keep batch alive; record the failure (retried on rerun)
                raw = f"__ERROR__ {type(e).__name__}: {e}"
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if raw.startswith("__ERROR__"):
                sc = {"gold": row.get("answer", ""), "pred": "", "correct": "", "mra": "",
                      "parse_ok": "", "score": ""}
            else:
                sc = score_row(row, raw, args.protocol)
            w.writerow({**base, **sc, "raw_output": raw})
            f.flush()
            if n_shown < args.show:
                n_shown += 1
                print(f"\n--- [{cond}/{variant}] {ev}\nPROMPT:\n{prompt}\nRAW: {raw!r}\n"
                      f"GOLD: {row.get('answer', '')!r} -> gold={sc['gold']!r} pred={sc['pred']!r} "
                      f"score={sc['score']}", flush=True)
            n_done += 1
            if n_done % 10 == 0:
                print(f"  ... {n_done}/{len(todo)}", flush=True)
            if args.limit and n_done >= args.limit:
                print(f"--limit {args.limit} reached, stopping.")
                break

    summarize(args.out)


# ---------------- summaries / cross-model report ----------------

def load_scored(path: str):
    """Successfully scored rows of one results CSV as a DataFrame with numeric `score`,
    `variant`, `upscaler`, `unit` (clean | degraded | restored:<upscaler>) and `run` (file stem)."""
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("variant", "upscaler", "model", "score"):
        if col not in df.columns:  # files from before these columns existed
            df[col] = ""
    recs = df.to_dict("records")
    df["variant"] = [infer_variant(r) for r in recs]
    df["upscaler"] = [infer_upscaler(r) for r in recs]
    df = df[(df["video_ok"] == "True") & (df["raw_output"] != "")
            & ~df["raw_output"].str.startswith("__ERROR__")].copy()
    mra = pd.to_numeric(df["mra"], errors="coerce")
    acc = (df["correct"] == "True").astype(float)
    df["score_f"] = pd.to_numeric(df["score"], errors="coerce").fillna(mra.where(mra.notna(), acc))
    df["is_numeric"] = df["type"].eq("numeric") | mra.notna()
    df["unit"] = [f"restored:{u}" if v == "restored" else v for v, u in zip(df["variant"], df["upscaler"])]
    df["run"] = Path(path).stem
    df.loc[df["model"] == "", "model"] = Path(path).stem
    return df


def vstat_metrics(df) -> dict:
    """VSTAT aggregates (x100): ALL_Score_avg over all questions, MCQ_ACC (= MAcc) over
    non-numeric, Numeric_MRA over numeric questions."""
    out = {"ALL_Score_avg": 100 * df["score_f"].mean(), "n": len(df)}
    non_num, num = df[~df["is_numeric"]], df[df["is_numeric"]]
    out["MCQ_ACC"] = 100 * non_num["score_f"].mean() if len(non_num) else float("nan")
    out["Numeric_MRA"] = 100 * num["score_f"].mean() if len(num) else float("nan")
    return out


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial) p-value on the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def restoration_transitions(df):
    """Per (run, model, condition, upscaler): clean/degraded/restored correctness patterns over
    MCQ (non-numeric) items scored in all three units. Uses the binary `correct` column, so a
    parse failure counts as incorrect."""
    import pandas as pd
    df = df[~df["is_numeric"]].assign(c=df["correct"].eq("True").astype(int))
    out = []
    for (run_, model), gm in df.groupby(["run", "model"], sort=False):
        C = gm[gm["variant"] == "clean"].set_index("item_id")["c"]
        for cond, gc in gm[gm["variant"] != "clean"].groupby("condition", sort=False):
            D = gc[gc["variant"] == "degraded"].set_index("item_id")["c"]
            for up, gr in gc[gc["variant"] == "restored"].groupby("upscaler", sort=False):
                R = gr.set_index("item_id")["c"]
                t = pd.concat({"C": C, "D": D, "R": R}, axis=1, join="inner")
                if t.empty:
                    continue
                rie_any = int(((t.D == 1) & (t.R == 0)).sum())
                ric_any = int(((t.D == 0) & (t.R == 1)).sum())
                out.append({
                    "run": run_, "model": model, "condition": cond, "upscaler": up, "n_paired": len(t),
                    "MAcc_clean": 100 * t.C.mean(), "MAcc_degraded": 100 * t.D.mean(),
                    "MAcc_restored": 100 * t.R.mean(),
                    "RIE": int(((t.C == 1) & (t.D == 1) & (t.R == 0)).sum()),
                    "RIC": int(((t.C == 1) & (t.D == 0) & (t.R == 1)).sum()),
                    "RIE_any": rie_any, "RIC_any": ric_any, "net_any": ric_any - rie_any,
                    "p_mcnemar": _mcnemar_exact(rie_any, ric_any),
                })
    tr = pd.DataFrame(out)
    if len(tr):
        rank = {c: i for i, c in enumerate(CONDITIONS)}
        tr = tr.sort_values(["run", "model", "upscaler", "condition"],
                            key=lambda s: s.map(rank).fillna(len(rank)) if s.name == "condition" else s,
                            kind="stable")
    return tr


def summarize(paths, save: str = "") -> None:
    """Per-model MAcc tables and restoration transitions for one or more results CSVs."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; skipping summary")
        return
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames = [load_scored(str(p)) for p in paths if os.path.isfile(p)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        print(f"\nNo scored rows in {', '.join(map(str, paths))} yet.")
        return
    has_items = "item_id" in df.columns and df["item_id"].fillna("").ne("").all()
    if has_items:
        df = df.drop_duplicates(["run", "model", "item_id", "condition", "variant", "upscaler"], keep="last")

    cond_order = [c for c in CONDITIONS if c in set(df["condition"])] + \
                 sorted(set(df["condition"]) - set(CONDITIONS))

    def unit_order(g):
        keys = g.groupby(["condition", "unit"]).groups.keys()
        return sorted(keys, key=lambda k: (cond_order.index(k[0]),
                                           _VARIANT_RANK.get(k[1].split(":")[0], 3), k[1]))

    def table_for(g, keys):
        t = pd.DataFrame({k: vstat_metrics(g[(g["condition"] == k[0]) & (g["unit"] == k[1])])
                          for k in keys}).T.rename(columns={"MCQ_ACC": "MAcc"})
        t.index.names = ["condition", "variant"]
        return t[["n", "MAcc", "Numeric_MRA", "ALL_Score_avg"]].dropna(axis=1, how="all").round(1)

    for (run_, model), gm in df.groupby(["run", "model"], sort=False):
        keys = unit_order(gm)
        print(f"\n=== {model} [{run_}]: all scored rows (x100; MAcc = accuracy over MCQ items) ===")
        print(table_for(gm, keys).to_string())
        if not has_items:
            continue
        per_item = gm.groupby("item_id").size()
        paired = gm[gm["item_id"].isin(per_item[per_item == len(keys)].index)]
        n_items = paired["item_id"].nunique()
        print(f"\n=== {model} [{run_}]: paired, {n_items} items present in all {len(keys)} "
              f"condition/variant units (x100) ===")
        if n_items:
            print(table_for(paired, keys).drop(columns="n").to_string())
            same = paired.groupby("item_id")["raw_output"].nunique().eq(1).mean()
            print(f"items with identical raw_output in every unit: {same:.1%}")
        print(f"parse_ok rate: {(gm['parse_ok'] == 'True').mean():.1%}")
        print("Most common raw outputs (first 40 chars):")
        print(gm["raw_output"].str.slice(0, 40).value_counts().head(5).to_string())

    if not has_items:
        return
    tr = restoration_transitions(df)
    if tr.empty:
        print("\nNo restoration transitions: need clean, degraded and restored results for the same items.")
        return
    print("\n=== Restoration transitions (MCQ items with clean, degraded and restored all scored) ===")
    print("RIE = clean ok, degraded ok, restored wrong | RIC = clean ok, degraded wrong, restored ok")
    print("*_any = same degraded->restored flip regardless of clean | p = exact McNemar, degraded vs restored")
    print(tr.round({"MAcc_clean": 1, "MAcc_degraded": 1, "MAcc_restored": 1, "p_mcnemar": 4})
          .to_string(index=False))
    if save:
        tr.to_csv(save, index=False)
        print(f"Saved transitions to {save}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Summarize VID-DRIFT results CSVs (one per model).")
    _ap.add_argument("results", nargs="+", help="results CSVs written by the *_generation.py scripts")
    _ap.add_argument("--save", default="", help="also write the transition table to this CSV")
    _a = _ap.parse_args()
    summarize(_a.results, save=_a.save)