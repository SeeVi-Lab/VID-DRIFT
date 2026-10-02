#!/usr/bin/env python3
"""Build one CSV per dataset (VSI-Bench, MLVU, MVBench, vstat) for VID-DRIFT.

Each row describes one QA item:
    dataset, video_path, video_found, question, answer, candidates, type,
    question_type, source_subset

Design: the on-disk video layout differs per dataset and often nests (e.g. MVBench
star/Charades_v1_480). Rather than hard-code fragile prefixes, we index every video
file/dir under each dataset once (basename -> path) and resolve each annotation's
`video` field by basename. This matches "look through json/ and search down the file".

Usage:
    python build_dataset_csvs.py --root ./data/ --out ./exp_inputs
    python build_dataset_csvs.py --root ./data/ --only vsibench mlvu

Only the standard library is required (pandas is used *only* if you ask for the parquet
fallback and test.jsonl is missing).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".gif", ".m4v"}
CSV_FIELDS = ["dataset", "video_path", "video_found", "question", "answer",
              "candidates", "type", "question_type", "source_subset"]


# ----------------------------- generic helpers -----------------------------

def index_videos(root: Path):
    """Walk `root`; return (by_basename, by_stem, by_dirname). Later wins on dupes are
    recorded but the first is kept; dup counts are reported by the caller if needed."""
    by_name: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    by_dir: dict[str, str] = {}
    dupes = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # skip version-control noise
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for d in dirnames:
            by_dir.setdefault(d, os.path.join(dirpath, d))
        for fn in filenames:
            if Path(fn).suffix.lower() in VIDEO_EXTS:
                full = os.path.join(dirpath, fn)
                if fn in by_name:
                    dupes += 1
                by_name.setdefault(fn, full)
                by_stem.setdefault(Path(fn).stem, full)
    return by_name, by_stem, by_dir, dupes


def resolve_video(name: str, base: Path, by_name, by_stem, by_dir):
    """Return (path, found). Try: direct join -> basename -> stem -> dir (frame folders)."""
    if not name:
        return "", False
    name = str(name)
    direct = base / name
    if direct.exists():
        return str(direct), True
    bn = os.path.basename(name.rstrip("/"))
    if bn in by_name:
        return by_name[bn], True
    stem = Path(bn).stem
    if stem in by_stem:
        return by_stem[stem], True
    if bn in by_dir:                 # frame-folder tasks (e.g. MVBench episodic/tvqa)
        return by_dir[bn], True
    if stem in by_dir:
        return by_dir[stem], True
    return str(base / name), False   # best guess, flagged not found


def first_key(entry: dict, aliases, default=None):
    for k in aliases:
        if k in entry and entry[k] not in (None, ""):
            return entry[k]
    return default


def dump_candidates(cands):
    if not cands:
        return ""
    if isinstance(cands, dict):               # {"A": "...", "B": "..."} -> list
        cands = [f"{k}. {v}" for k, v in cands.items()]
    return json.dumps(list(cands), ensure_ascii=False)


def looks_numeric(x) -> bool:
    try:
        float(str(x).replace(",", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_CONTAINER_KEYS = ("data", "questions", "annotations", "qa", "qa_list", "items", "videos", "test")


def to_record_list(obj) -> list[dict]:
    """Normalize any JSON shape into a flat list[dict] of QA records.

    Handles: list[dict]; {"data": [...]}; {id: {...}} (id injected as `_key`);
    {video: [ {...}, ... ]} (key injected as `_key`). Non-dict records are dropped.
    """
    # unwrap a single common container level, e.g. {"data": [...]}, {"questions": {...}}
    if isinstance(obj, dict):
        for k in _CONTAINER_KEYS:
            if k in obj and isinstance(obj[k], (list, dict)):
                obj = obj[k]
                break

    recs: list = []
    if isinstance(obj, list):
        recs = obj
    elif isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, dict):
                recs.append({"_key": key, **val})
            elif isinstance(val, list):
                for it in val:
                    if isinstance(it, dict):
                        recs.append({"_key": key, **it})
    return [r for r in recs if isinstance(r, dict)]


def report(name, rows):
    found = sum(1 for r in rows if r["video_found"])
    types = defaultdict(int)
    for r in rows:
        types[r["type"]] += 1
    print(f"  [{name}] {len(rows)} items | videos resolved {found}/{len(rows)} "
          f"| types { {k: v for k, v in types.items()} }")


def write_csv(out_dir: Path, name: str, rows: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"  -> wrote {path}")
    return path


# ----------------------------- per-dataset builders -----------------------------

def build_vsibench(root: Path):
    base = root / "VSI-Bench"
    jsonl = base / "test.jsonl"
    rows = []
    by_name, by_stem, by_dir, _ = index_videos(base)

    entries = []
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            entries = [json.loads(ln) for ln in f if ln.strip()]
    else:  # parquet fallback
        try:
            import pandas as pd
            pq = base / "test_debiased.parquet"
            if not pq.exists():
                pq = base / "test_pruned.parquet"
            entries = pd.read_parquet(pq).to_dict("records")
        except Exception as e:
            print(f"  [vsibench] no test.jsonl and parquet fallback failed: {e}", file=sys.stderr)
            return rows
    if entries:
        print(f"  [vsibench] sample keys: {sorted(entries[0].keys())}", file=sys.stderr)

    for e in entries:
        if e.get("pruned") is True:
            continue
        sub = first_key(e, ["dataset", "source"], "")
        scene = first_key(e, ["scene_name", "scene", "video"], "")
        options = first_key(e, ["options", "candidates", "choices"], None)
        gt = first_key(e, ["ground_truth", "answer", "gt"], "")
        qtype = first_key(e, ["question_type", "type"], "")
        # VSI-Bench: MCQ iff options present, else numeric.
        qkind = "mcq" if options else "numeric"
        vid_name = f"{scene}.mp4" if scene and not str(scene).endswith(tuple(VIDEO_EXTS)) else scene
        vpath = base / str(sub) / vid_name if sub else base / vid_name
        if vpath.exists():
            resolved, found = str(vpath), True
        else:
            resolved, found = resolve_video(vid_name, base, by_name, by_stem, by_dir)
        rows.append({"dataset": "VSI-Bench", "video_path": resolved, "video_found": found,
                     "question": first_key(e, ["question"], ""), "answer": gt,
                     "candidates": dump_candidates(options), "type": qkind,
                     "question_type": qtype, "source_subset": sub})
    return rows


def build_mlvu(root: Path):
    # directory is named "MVLU", inner package "MLVU"
    base = None
    for cand in [root / "MVLU" / "MLVU", root / "MLVU" / "MLVU", root / "MLVU", root / "MVLU"]:
        if (cand / "json").is_dir():
            base = cand
            break
    rows = []
    if base is None:
        print("  [mlvu] json/ dir not found under MVLU/MLVU", file=sys.stderr)
        return rows
    json_dir, video_dir = base / "json", base / "video"
    by_name, by_stem, by_dir, _ = index_videos(video_dir if video_dir.exists() else base)

    printed = False
    for jf in sorted(json_dir.glob("*.json")):
        task = jf.stem                                  # e.g. "4_count"
        qtype = task.split("_", 1)[1] if "_" in task and task.split("_", 1)[0].isdigit() else task
        task_video_dir = video_dir / task
        data = to_record_list(load_json(jf))
        if data and not printed:
            print(f"  [mlvu] sample keys ({jf.name}): {sorted(data[0].keys())}", file=sys.stderr)
            printed = True
        for e in data:
            cands = first_key(e, ["candidates", "options", "choices"], None)
            ans = first_key(e, ["answer", "ground_truth", "gt"], "")
            vid = first_key(e, ["video", "video_name", "video_path", "vid"], "")
            qkind = "mcq" if cands else "openended"
            base_try = task_video_dir if task_video_dir.exists() else video_dir
            resolved, found = resolve_video(vid, base_try, by_name, by_stem, by_dir)
            rows.append({"dataset": "MLVU", "video_path": resolved, "video_found": found,
                         "question": first_key(e, ["question"], ""), "answer": ans,
                         "candidates": dump_candidates(cands), "type": qkind,
                         "question_type": first_key(e, ["question_type"], qtype),
                         "source_subset": task})
    return rows


def build_mvbench(root: Path):
    base = root / "MVBench"
    json_dir, video_base = base / "json", base / "video"
    rows = []
    if not json_dir.is_dir():
        print("  [mvbench] json/ dir not found", file=sys.stderr)
        return rows
    by_name, by_stem, by_dir, _ = index_videos(video_base if video_base.exists() else base)

    printed = False
    for jf in sorted(json_dir.glob("*.json")):
        qtype = jf.stem                                 # e.g. "action_sequence"
        data = to_record_list(load_json(jf))
        if data and not printed:
            print(f"  [mvbench] sample keys ({jf.name}): {sorted(data[0].keys())}", file=sys.stderr)
            printed = True
        for e in data:
            cands = first_key(e, ["candidates", "options", "choices"], None)
            ans = first_key(e, ["answer", "ground_truth", "gt"], "")
            vid = first_key(e, ["video", "video_name", "video_path", "vid"], "")
            resolved, found = resolve_video(vid, video_base, by_name, by_stem, by_dir)
            rows.append({"dataset": "MVBench", "video_path": resolved, "video_found": found,
                         "question": first_key(e, ["question"], ""), "answer": ans,
                         "candidates": dump_candidates(cands), "type": "mcq",
                         "question_type": qtype, "source_subset": ""})
    return rows


def build_vstat(root: Path):
    base = root / "vstat"
    rows = []
    ann = None
    for cand in ["vstat_qa_clean.json", "vstat_qa.json", "agentic_test_qa_list.json"]:
        if (base / cand).exists():
            ann = base / cand
            break
    if ann is None:
        print("  [vstat] no QA json found", file=sys.stderr)
        return rows
    video_root = base / "videos"
    by_name, by_stem, by_dir, _ = index_videos(video_root if video_root.exists() else base)

    raw = load_json(ann)
    if isinstance(raw, dict):
        print(f"  [vstat] {ann.name} top-level dict keys: {list(raw.keys())[:12]}", file=sys.stderr)
    data = to_record_list(raw)
    if data:
        print(f"  [vstat] using {ann.name}; {len(data)} records; sample keys: {sorted(data[0].keys())}",
              file=sys.stderr)
    else:
        print(f"  [vstat] could NOT normalize {ann.name}; top-level type={type(raw).__name__}. "
              f"Paste its structure so I can adjust.", file=sys.stderr)
        return rows

    for e in data:
        cands = first_key(e, ["candidates", "options", "choices"], None)
        ans = first_key(e, ["answer", "ground_truth", "gt", "label"], "")
        vid = first_key(e, ["video", "video_path", "video_id", "path", "video_name", "clip", "_key"], "")
        src = first_key(e, ["source", "video_source", "split", "category_source"], "")
        qtype = first_key(e, ["question_type", "type", "state_element", "task"], "")
        # classify format
        if cands:
            qkind = "mcq"
        elif looks_numeric(ans):
            qkind = "numeric"
        else:
            qkind = "openended"
        resolved, found = resolve_video(vid, video_root, by_name, by_stem, by_dir)
        if not src:  # infer subset from the resolved path if possible
            low = resolved.lower()
            src = ("synthetic" if "synthetic" in low else
                   "self_recorded" if "self_recorded" in low else
                   "youtube" if not found else "")
        rows.append({"dataset": "vstat", "video_path": resolved if found else "",
                     "video_found": found, "question": first_key(e, ["question"], ""),
                     "answer": ans, "candidates": dump_candidates(cands), "type": qkind,
                     "question_type": qtype, "source_subset": src})
    return rows


BUILDERS = {"vsibench": build_vsibench, "mlvu": build_mlvu,
            "mvbench": build_mvbench, "vstat": build_vstat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/")
    ap.add_argument("--out", default="./exp_inputs")
    ap.add_argument("--only", nargs="*", choices=list(BUILDERS), default=list(BUILDERS))
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    print(f"ROOT={root}  OUT={out}")
    for name in args.only:
        print(f"[{name}] building ...")
        rows = BUILDERS[name](root)
        if rows:
            report(name, rows)
            write_csv(out, name, rows)
        else:
            print(f"  [{name}] no rows produced (check paths/keys above)", file=sys.stderr)


if __name__ == "__main__":
    main()
