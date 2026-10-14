#!/usr/bin/env python3
"""Percent of clean videos that would have < N frames after each frame-rate downsampling.

    python count_short_videos.py results/qwen3vl_results.csv [--n 32] [--save per_video.csv]

Uses unique `eval_video` paths from rows with condition == clean.
  fps_quarter   : frames / 4  < N
  fps_eighth    : frames / 8  < N
  fpsabs_1fps   : seconds     < N
  fpsabs_0_5fps : seconds / 2 < N
"""
import argparse
import csv
import os

from torchcodec.decoders import VideoDecoder

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--save", default="", help="optional per-video CSV (path, frames, seconds, fps)")
args = ap.parse_args()

with open(args.csv, newline="", encoding="utf-8") as f:
    paths = sorted({r["eval_video"] for r in csv.DictReader(f)
                    if r.get("condition") == "clean" and r.get("eval_video")})
print(f"{len(paths)} unique clean videos in {args.csv}")

info, failed = [], []
for i, p in enumerate(paths, 1):
    try:
        md = VideoDecoder(p).metadata  # default seek_mode="exact" scans the file: exact counts
        frames = md.num_frames
        secs = md.duration_seconds
        fps = md.average_fps
        if secs is None and frames and fps:
            secs = frames / fps
        info.append((p, int(frames), float(secs), float(fps or 0)))
    except Exception as e:
        failed.append((p, f"{type(e).__name__}: {e}"))
    if i % 200 == 0:
        print(f"  ... {i}/{len(paths)}", flush=True)

if failed:
    print(f"Could not read {len(failed)} videos, e.g. {failed[0][0]}: {failed[0][1]}")
if not info:
    raise SystemExit("no readable videos")

N = args.n
rules = {
    "fps_quarter   (frames/4 )": lambda fr, s: fr / 4 < N,
    "fps_eighth    (frames/8 )": lambda fr, s: fr / 8 < N,
    "fpsabs_1fps   (seconds  )": lambda fr, s: s < N,
    "fpsabs_0_5fps (seconds/2)": lambda fr, s: s / 2 < N,
}
fr = sorted(x[1] for x in info)
sec = sorted(x[2] for x in info)
mid = len(info) // 2
print(f"\nClean videos read: {len(info)} | frames min/median/max {fr[0]}/{fr[mid]}/{fr[-1]} "
      f"| seconds min/median/max {sec[0]:.1f}/{sec[mid]:.1f}/{sec[-1]:.1f}")
print(f"Videos with < {N} frames after downsampling:")
flags = [[rule(f_, s) for rule in rules.values()] for _, f_, s, _ in info]
for j, name in enumerate(rules):
    k = sum(fl[j] for fl in flags)
    print(f"  {name}: {k:>6} / {len(info)}  ({100 * k / len(info):5.1f}%)")
tot, den = sum(map(sum, flags)), len(flags) * len(rules)
any_, all_ = sum(any(fl) for fl in flags), sum(all(fl) for fl in flags)
print(f"  {'TOTAL (all video x condition pairs)':<26}: {tot:>6} / {den}  ({100 * tot / den:5.1f}%)")
print(f"  {'videos short in ANY condition':<26}: {any_:>6} / {len(info)}  ({100 * any_ / len(info):5.1f}%)")
print(f"  {'videos short in ALL conditions':<26}: {all_:>6} / {len(info)}  ({100 * all_ / len(info):5.1f}%)")

if args.save:
    with open(args.save, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eval_video", "frames", "seconds", "fps"])
        w.writerows(info)
    print(f"Saved per-video counts to {args.save}")