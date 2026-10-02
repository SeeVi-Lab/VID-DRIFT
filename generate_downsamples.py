#!/usr/bin/env python3
"""Apply the VID-DRIFT degradation matrix to every unique source video referenced in
the CSVs under `original_csv/`, and emit a manifest (`downsample.csv`).

For each unique `video_path` that is a real file and **< max-minutes long**, produce all
degraded variants and save them as:

    {out_root}/{dataset}/{method}_{amount}/{VIDEO_NAME}.mp4

Methods (extended from table.tsv):
    downscale_2/_4/_8/_12/_16     jpeg_40/_20/_10/_5/_2
    h264_28/_35/_45/_51           h265_28/_35/_45/_51
    fps_half/_quarter/_eighth     fpsabs_1fps/_0_5fps (absolute, extreme)
    streamproxy_x4_crf35/_x8_crf45                                      (25 variants)

Manifest (`downsample.csv`): one row per (QA item x degradation variant). Carries the
input CSV columns (dataset, video_path, video_found, question, answer, candidates, type,
question_type, source_subset) plus: method, amount, degraded_path, degraded_exists,
orig_duration_sec, orig_width, orig_height, orig_fps. Only videos that passed the
duration filter appear (skipped/long videos have no variants).

Outputs are normalized to .mp4 (all target codecs incl. MJPEG are MP4-compatible), so it
works for any source container. Existing outputs are skipped (resumable). Requires
system `ffmpeg` + `ffprobe`.

Usage:
    python apply_degradations.py --csv-dir original_csv \
        --out-root /data/jpk322/VID-DRIFT/downsamples \
        --max-minutes 20 --workers 8 --manifest downsample.csv
    python apply_degradations.py --manifest-only        # rebuild downsample.csv only
    python apply_degradations.py --dry-run              # plan, no encoding
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm isn't installed
    def tqdm(iterable=None, total=None, desc=None, **_):
        if desc:
            print(f"[{desc}] (install tqdm for a progress bar)", file=sys.stderr)
        return iterable if iterable is not None else []

# ---- method matrix: (method, amount) ----  (25 variants)
METHODS: list[tuple[str, str]] = (
    [("downscale", str(f)) for f in (2, 4, 8, 12, 16)]        # spatial, incl. extreme
    + [("jpeg", str(q)) for q in (40, 20, 10, 5, 2)]          # per-frame quality, incl. extreme
    + [("h264", str(c)) for c in (28, 35, 45, 51)]            # H.264 CRF (51 = max)
    + [("h265", str(c)) for c in (28, 35, 45, 51)]            # H.265 CRF (51 = max)
    + [("fps", "half"), ("fps", "quarter"), ("fps", "eighth")]  # temporal, fraction of native
    + [("fpsabs", "1fps"), ("fpsabs", "0_5fps")]             # temporal, absolute (extreme)
    + [("streamproxy", "x4_crf35"), ("streamproxy", "x8_crf45")]  # combined proxies
)
FPS_RATIO = {"half": 0.5, "quarter": 0.25, "eighth": 0.125}
FPS_ABS = {"1fps": 1.0, "0_5fps": 0.5}

INPUT_COLS = ["dataset", "video_path", "video_found", "question", "answer",
              "candidates", "type", "question_type", "source_subset"]
MANIFEST_EXTRA = ["method", "amount", "degraded_path", "degraded_exists",
                  "orig_duration_sec", "orig_width", "orig_height", "orig_fps"]


def jpeg_qv(quality: int) -> int:
    """Map JPEG quality (0-100, higher=better) to ffmpeg mjpeg -q:v (2-31, lower=better)."""
    quality = max(1, min(100, quality))
    return int(round(2 + (31 - 2) * (100 - quality) / 99))


# ---- ffprobe ----

def probe_meta(path: str) -> dict | None:
    """One ffprobe call -> {duration, fps, width, height} or None if unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,avg_frame_rate:format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=60)
        info = json.loads(out.stdout or "{}")
        st = (info.get("streams") or [{}])[0]
        dur = info.get("format", {}).get("duration")
        num, _, den = (st.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den) if den and float(den) else None
        return {"duration": float(dur) if dur is not None else None, "fps": fps,
                "width": st.get("width"), "height": st.get("height")}
    except (ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError):
        return None


# ---- ffmpeg command builder ----

def scale_expr(f: int) -> str:
    """Area-downscale by `f`, keeping dims even and >= 2px (guards extreme factors)."""
    return (f"scale=w=max(trunc(iw/{f}/2)*2\\,2):h=max(trunc(ih/{f}/2)*2\\,2):flags=area")


def build_cmd(method: str, amount: str, src: str, dst: str, native_fps: float | None) -> list[str] | None:
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", src]
    tail = ["-c:a", "copy", "-pix_fmt", "yuv420p", dst]
    if method == "downscale":
        return base + ["-vf", scale_expr(int(amount)), "-c:v", "libx264", "-crf", "12"] + tail
    if method == "jpeg":
        return base + ["-c:v", "mjpeg", "-q:v", str(jpeg_qv(int(amount))),
                       "-pix_fmt", "yuvj420p", "-c:a", "copy", dst]
    if method == "h264":
        return base + ["-c:v", "libx264", "-crf", amount] + tail
    if method == "h265":
        return base + ["-c:v", "libx265", "-crf", amount] + tail
    if method == "fps":
        if not native_fps:
            return None
        target = native_fps * FPS_RATIO[amount]
        return base + ["-vf", f"fps={target:.6f}", "-c:v", "libx264", "-crf", "12"] + tail
    if method == "fpsabs":
        target = FPS_ABS[amount]
        return base + ["-vf", f"fps={target:.6f}", "-c:v", "libx264", "-crf", "12"] + tail
    if method == "streamproxy":
        # amount like "x4_crf35" / "x8_crf45"
        fac = int(amount.split("_")[0].lstrip("x"))
        crf = amount.split("crf")[-1]
        return base + ["-vf", scale_expr(fac), "-c:v", "libx265", "-crf", crf] + tail
    return None


# ---- CSV ingest ----

def iter_csv_rows(csv_dir: Path):
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        sys.exit(f"No CSVs found in {csv_dir}")
    for cf in files:
        with open(cf, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "video_path" not in (reader.fieldnames or []):
                print(f"  ! {cf.name}: no 'video_path' column, skipping", file=sys.stderr)
                continue
            for row in reader:
                if (row.get("video_path") or "").strip():
                    yield cf.name, row


def infer_dataset(path: str) -> str:
    parts = Path(path).parts
    for key in ("VSI-Bench", "MLVU", "MVLU", "MVBench", "vstat"):
        if key in parts:
            return "MLVU" if key == "MVLU" else key
    return "unknown"


def row_dataset(row: dict) -> str:
    return (row.get("dataset") or "").strip() or infer_dataset(row.get("video_path", ""))


def out_path(out_root: Path, dataset: str, method: str, amount: str, src: str,
             claimed: dict[str, str]) -> Path:
    """downsamples/{dataset}/{method}_{amount}/{VIDEO_NAME}.mp4 with collision guard."""
    d = out_root / dataset / f"{method}_{amount}"
    p = d / (Path(src).stem + ".mp4")
    if claimed.get(str(p), src) != src:          # different source already took this name
        h = hashlib.sha1(src.encode()).hexdigest()[:6]
        p = d / f"{Path(src).stem}_{h}.mp4"
    claimed[str(p)] = src
    return p


# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="original_csv")
    ap.add_argument("--out-root", default="/data/jpk322/VID-DRIFT/downsamples")
    ap.add_argument("--manifest", default="downsample.csv")
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--methods", nargs="*", help="limit to method prefixes, e.g. downscale h265")
    ap.add_argument("--include-clean", action="store_true",
                    help="also add a method=clean row (degraded_path = original video) per QA item")
    ap.add_argument("--manifest-only", action="store_true", help="rebuild manifest, no encoding")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    csv_dir, out_root = Path(args.csv_dir), Path(args.out_root)
    max_sec = args.max_minutes * 60.0
    methods = [(m, a) for (m, a) in METHODS
               if not args.methods or any(m.startswith(p) for p in args.methods)]

    # Collect unique videos (video_path -> dataset).
    uniq: dict[str, str] = {}
    for _, row in iter_csv_rows(csv_dir):
        uniq.setdefault(row["video_path"].strip(), row_dataset(row))
    print(f"Unique video_path entries across CSVs: {len(uniq)}")

    # Pass 1: filter by existence + duration; probe each once.
    eligible: dict[str, tuple[str, dict]] = {}
    n_missing = n_long = 0
    for vp, ds in tqdm(uniq.items(), total=len(uniq), desc="probe videos"):
        if not os.path.isfile(vp):
            n_missing += 1; continue
        meta = probe_meta(vp)
        if not meta or meta["duration"] is None:
            n_missing += 1; continue
        if meta["duration"] > max_sec:
            n_long += 1; continue
        eligible[vp] = (ds, meta)
    print(f"  eligible: {len(eligible)} | skipped not-a-file/unreadable: {n_missing} | "
          f"skipped >={args.max_minutes}min: {n_long}")

    # Pass 2: assign a deterministic output path to every (video, method) variant.
    claimed: dict[str, str] = {}
    variant_path: dict[tuple[str, str, str], str] = {}
    for vp, (ds, _meta) in eligible.items():
        for (m, a) in methods:
            variant_path[(vp, m, a)] = str(out_path(out_root, ds, m, a, vp, claimed))

    # Encode (unless manifest-only / dry-run).
    if not args.manifest_only:
        tasks = [(m, a, vp, variant_path[(vp, m, a)], eligible[vp][1]["fps"])
                 for vp in eligible for (m, a) in methods
                 if not Path(variant_path[(vp, m, a)]).exists()]
        print(f"  planned encodes: {len(tasks)} ({len(eligible)} videos x {len(methods)} methods, minus existing)")
        if args.dry_run:
            for (m, a, src, dst, _) in tasks[:20]:
                print(f"    {m}_{a:<10} {Path(src).name}  ->  {dst}")
            if len(tasks) > 20:
                print(f"    ... (+{len(tasks) - 20} more)")
            return

        def run(task):
            m, a, src, dst, fps = task
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            cmd = build_cmd(m, a, src, dst, fps)
            if cmd is None:
                return (dst, "skip")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                Path(dst).unlink(missing_ok=True)
                return (dst, f"FAIL: {proc.stderr.strip()[-300:]}")
            return (dst, "ok")

        done = fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(tasks), desc="encode", unit="clip"):
                dst, status = fut.result()
                if status == "ok":
                    done += 1
                elif status.startswith("FAIL"):
                    fail += 1
                    print(f"  ! {Path(dst).name}: {status}", file=sys.stderr)
        print(f"Encoding done. encoded={done} failed={fail}")

    # Manifest: one row per (QA item x variant) for eligible videos.
    man_path = Path(args.manifest)
    n_rows = n_excluded = 0
    total_qa = sum(1 for _ in iter_csv_rows(csv_dir))
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INPUT_COLS + MANIFEST_EXTRA)
        w.writeheader()
        for _, row in tqdm(iter_csv_rows(csv_dir), total=total_qa, desc="manifest", unit="row"):
            vp = row["video_path"].strip()
            if vp not in eligible:
                n_excluded += 1; continue
            _ds, meta = eligible[vp]
            base = {c: row.get(c, "") for c in INPUT_COLS}
            meta_cols = {"orig_duration_sec": round(meta["duration"], 3),
                         "orig_width": meta["width"], "orig_height": meta["height"],
                         "orig_fps": round(meta["fps"], 3) if meta["fps"] else ""}
            if args.include_clean:
                w.writerow({**base, "method": "clean", "amount": "", "degraded_path": vp,
                            "degraded_exists": True, **meta_cols})
                n_rows += 1
            for (m, a) in methods:
                dst = variant_path[(vp, m, a)]
                w.writerow({**base, "method": m, "amount": a, "degraded_path": dst,
                            "degraded_exists": Path(dst).exists(), **meta_cols})
                n_rows += 1
    print(f"Manifest -> {man_path}  ({n_rows} rows; {n_excluded} QA rows excluded: "
          f"video long/missing)")


if __name__ == "__main__":
    main()
