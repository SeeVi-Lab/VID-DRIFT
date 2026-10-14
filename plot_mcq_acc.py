"""
Plot MCQ_ACC vs. degradation level, one line per (upscaler, model).

Each input folder = one restoration method (folder name -> legend label via
PRETTY_UPSCALERS / pretty_upscaler()).
Each folder holds CSVs with columns:
  key,dataset,condition,eval_video,video_ok,question,type,gold,raw_output,
  pred,correct,mra,parse_ok,model,item_id,score

Folders whose name starts with a TEMPORAL_PREFIXES entry (e.g. EDEN_mvbench) are
frame-interpolation runs: their conditions are parsed as frame-rate reductions
(parse_fps_condition) and plotted in the "fps" panel. All other folders use the
spatial parser (parse_condition).

Outputs (in --out):
  mcq_acc_downscale.<fmt>     clean, x4, x12, x16
  mcq_acc_h265.<fmt>          clean, CRF 28, CRF 45, CRF 51
  mcq_acc_jpeg.<fmt>          clean, Q40, Q10, Q5
  mcq_acc_streamproxy.<fmt>   clean, x4+CRF35, x8+CRF45
  mcq_acc_fps.<fmt>           clean, 1/4, 1/8, 1 fps, 0.5 fps   (EDEN folders)
  mcq_acc_combined.<fmt>      selected panels in one row (or a grid via --ncols),
                              shared y-axis and legend
  mcq_acc_by_condition.csv    aggregated values (+ n per point)

Same video set across folders: per model, only items (item_id) whose clean answer
exists in EVERY input folder are kept, together with all of their degraded /
restored rows. Clean rows with video_ok == False do not count as existing.

SDS (needs a `variant` column separating degraded from restored rows), per
(folder, model, condition), paired within the folder on item_id, on the anchored
set A = {clean correct}:
  RIE = clean ok, degraded ok, restored wrong
  RIC = clean ok, degraded wrong, restored ok
  r = RIC/|A|, h = RIE/|A|, SDS = RIE/(RIE+RIC)   (NaN if no flips)
  *_any = same flip regardless of clean; p_mcnemar = exact McNemar on *_any.
  condition "ALL" pools all conditions (items repeat, so no p).
  -> sds_by_condition.csv (long), sds_wide.csv (rows = folder x model)
With `variant`, accuracy lines use restored rows only (degraded rows are not
averaged into them).

Only families with data are plotted unless --families is given. An upscaler is
drawn in a panel only if it has at least one non-clean point in that family, so
EDEN does not appear as a lone clean marker in the spatial panels (and vice versa).

Colour -> model, line style + marker -> upscaler.

Usage:
  python plot_mcq_acc_by_condition.py runs/FlashVSR_mvbench runs/DOVE_mvbench_472 \
      runs/EDEN_mvbench --out figs/ --fmt pdf
  # original two-panel combined figure:
  python plot_mcq_acc_by_condition.py ... --families downscale h265
  # 2-column grid instead of a single row:
  python plot_mcq_acc_by_condition.py ... --ncols 2
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ----------------------------------------------------------------------------
# Config: data
# ----------------------------------------------------------------------------
# Order = panel order in the combined figure (row-major). Panel width scales
# with the number of levels, so tick spacing is uniform across panels; in a
# multi-row grid (--ncols < #panels) each column is as wide as its widest family.
FAMILIES = {
    "downscale": ["clean", "4", "12", "16"],
    "h265": ["clean", "28", "45", "51"],
    "jpeg": ["clean", "40", "10", "5"],
    "streamproxy": ["clean", "x4_crf35", "x8_crf45"],
    # Fractions are relative to the source frame rate; "Nfps" are absolute.
    "fps": ["clean", "1/4", "1/8", "1fps", "0.5fps"],
}
XLABELS = {
    "downscale": "Downscale factor",
    "h265": "H.265 CRF",
    "jpeg": "JPEG quality",
    "streamproxy": "Downscale + H.265 CRF",
    "fps": "Frame rate",
}
TITLES = {
    "downscale": "Downscaling",
    "h265": "H.265 compression",
    "jpeg": "JPEG compression",
    "streamproxy": "StreamProxy",
    "fps": "Frame-rate reduction",
}

# Folder-name prefixes (case-insensitive) whose conditions are frame-rate changes.
TEMPORAL_PREFIXES = ("EDEN",)


def _fmt_streamproxy(lv: str) -> str:
    m = re.fullmatch(r"x(\d+)_crf(\d+)", lv)
    return f"×{m[1]}\nCRF {m[2]}" if m else lv


def _fmt_fps(lv: str) -> str:
    return re.sub(r"fps$", " fps", lv)  # "0.5fps" -> "0.5 fps"; "1/4" unchanged


TICK_FMT = {
    "downscale": lambda lv: "Clean" if lv == "clean" else f"×{lv}",
    "h265": lambda lv: "Clean" if lv == "clean" else lv,
    "jpeg": lambda lv: "Clean" if lv == "clean" else lv,
    "streamproxy": lambda lv: "Clean" if lv == "clean" else _fmt_streamproxy(lv),
    "fps": lambda lv: "Clean" if lv == "clean" else _fmt_fps(lv),
}

# Values of `type` that count as multiple-choice (compared lower-cased).
MCQ_TYPES = {"mcq", "multiple_choice", "multiple-choice", "mc"}

# ----------------------------------------------------------------------------
# Config: display names (dict order = legend/colour order)
# ----------------------------------------------------------------------------
PRETTY_MODELS = {
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "OpenGVLab/InternVL3-8B-hf": "InternVL3-8B",
    "allenai/Molmo2-8B": "Molmo2-8B",
    "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct": "LLaVA-OV-2-8B",
}
PRETTY_UPSCALERS = {
    "FlashVSR_mvbench": "FlashVSR",
    "DOVE_mvbench_472": "DOVE",
}


def pretty_model(raw: str) -> str:
    """Explicit map first; otherwise drop org prefix and common suffixes."""
    if raw in PRETTY_MODELS:
        return PRETTY_MODELS[raw]
    name = str(raw).split("/")[-1]
    name = re.sub(r"[-_](Instruct|Chat|hf|HF)$", "", name)
    name = re.sub(r"[-_](Instruct|Chat|hf|HF)$", "", name)  # e.g. -Instruct-hf
    return name.replace("OneVision", "OV")


def pretty_upscaler(raw: str) -> str:
    """Explicit map first; otherwise the part before the first underscore."""
    return PRETTY_UPSCALERS.get(raw, str(raw).split("_")[0])


def is_temporal_folder(name: str) -> bool:
    return str(name).upper().startswith(tuple(p.upper() for p in TEMPORAL_PREFIXES))


# ----------------------------------------------------------------------------
# Config: style
# ----------------------------------------------------------------------------
# Categorical slots in fixed order (CVD-validated for <=4 adjacent series).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# (linestyle, marker) per upscaler, in folder order.
UPSCALER_STYLES = [
    ("-", "o"),
    ((0, (4, 1.6)), "s"),
    ((0, (1, 1.3)), "^"),
    ((0, (5, 1.5, 1, 1.5)), "D"),
    ((0, (2, 1)), "v"),
]

INK = "#1f1f1d"
INK_2 = "#52514e"
INK_3 = "#8a8984"
AXIS = "#bdbcb6"
GRID = "#e9e8e3"
CLEAN_BAND = "#f3f2ee"

# Inches of figure width per unit of width_ratio in the combined figure
# (4.3 in per 4-level panel, matching the previous fixed per-column size).
IN_PER_UNIT = 4.3 / 3.7

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,  # embed TrueType (camera-ready friendly)
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def to_bool(s: pd.Series) -> pd.Series:
    """Robustly coerce True/False/1/0/'true'/'False' to float {0,1}."""
    if s.dtype == bool:
        return s.astype(float)
    return s.astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
    )


def load_folder(folder: Path) -> pd.DataFrame:
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {folder}")
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in csvs], ignore_index=True)
    df["upscaler"] = folder.name
    return df


def parse_condition(cond: str):
    """
    Map a raw (spatial) condition string to (family, level).
    'clean'                          -> ('clean', 'clean')
    'streamproxy_x4_crf35'           -> ('streamproxy', 'x4_crf35')
    'jpeg_40' / 'jpg40'              -> ('jpeg', '40')
    'h265_crf28' / 'h265-28'         -> ('h265', '28')
    'downscale_4' / 'ds4'            -> ('downscale', '4')
    Order matters: StreamProxy contains 'crf' and must be checked before H.265.
    Returns (None, None) if unrecognised.
    """
    c = str(cond).strip().lower()
    if "clean" in c or c in {"orig", "original", "none"}:
        return "clean", "clean"
    if "stream" in c or "proxy" in c or re.search(r"x\d+\D*crf\d+", c):
        nums = re.findall(r"\d+", c)  # last two numbers = scale, CRF
        return ("streamproxy", f"x{nums[-2]}_crf{nums[-1]}") if len(nums) >= 2 else (None, None)
    if "jpeg" in c or "jpg" in c:
        nums = re.findall(r"\d+", c)
        return ("jpeg", nums[-1]) if nums else (None, None)
    if any(t in c for t in ("h265", "hevc", "x265", "crf")):
        rest = re.sub(r"h265|x265|hevc", "", c)
        nums = re.findall(r"\d+", rest)
        return ("h265", nums[-1]) if nums else (None, None)
    if any(t in c for t in ("down", "ds", "scale", "res")):
        nums = re.findall(r"\d+", c)
        return ("downscale", nums[-1]) if nums else (None, None)
    return None, None


def parse_fps_condition(cond: str):
    """
    Map a frame-rate condition string (temporal folders only) to ('fps', level).
    Relative to the source rate -> '1/N':
      'fps_1_4' / 'fps_1/4' / 'fps_div4' / 'stride4' / 'quarter'   -> '1/4'
    Absolute rate -> '<v>fps':
      '1fps' / 'fps_1' / 'fps1'                                    -> '1fps'
      '0.5fps' / '0_5fps' / '0p5fps' / 'fps_0.5' / 'fps_0_5'       -> '0.5fps'
    Fractions are checked first, so 'fps_1_4' is 1/4 rather than 1 fps.
    Returns (None, None) if unrecognised.
    """
    c = str(cond).strip().lower()
    if "clean" in c or c in {"orig", "original", "none"}:
        return "clean", "clean"
    c = re.sub(r"(?<=\d)p(?=\d)", ".", c)  # 0p5 -> 0.5
    if "quarter" in c:
        return "fps", "1/4"
    if "eighth" in c:
        return "fps", "1/8"
    m = (re.search(r"(?<![\d.])1\s*[/:_-]\s*(\d+)(?![\d.]|\s*fps)", c)
         or re.search(r"(?:div|stride|every|skip|subsample)[_-]?(\d+)", c))
    if m:
        return "fps", f"1/{int(m[1])}"
    m = (re.search(r"(\d+(?:[._]\d+)?)\s*[_-]?\s*fps", c)
         or re.search(r"fps\s*[_-]?\s*(\d+(?:[._]\d+)?)", c))
    if m:
        return "fps", f"{float(m[1].replace('_', '.')):g}fps"
    return None, None


# ----------------------------------------------------------------------------
# SDS
# ----------------------------------------------------------------------------
def mcnemar_exact(b, c) -> float:
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    n = int(b + c)
    if n == 0:
        return float("nan")
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
          for i in range(int(min(b, c)) + 1)]
    return min(1.0, 2 * sum(math.exp(x) for x in lp))


SDS_COLS = ["upscaler", "model", "condition", "n_paired", "MAcc_clean", "MAcc_degraded",
            "MAcc_restored", "n_anchored", "RIE", "RIC", "r", "h", "SDS",
            "RIE_any", "RIC_any", "net_any", "SDS_any", "p_mcnemar", "_r"]


def sds_table(df: pd.DataFrame) -> pd.DataFrame:
    """SDS per (upscaler, model, condition); clean / degraded / restored answers
    are paired within each folder on item_id. `_r` is a sort key."""
    k = ["upscaler", "model", "item_id"]
    cl = df["family"] == "clean"
    C = (df[cl].drop_duplicates(k)[k + ["correct"]]
         .rename(columns={"correct": "c_C"}))
    D = (df[~cl & (df["variant"] == "degraded")].drop_duplicates(k + ["condition"])
         [k + ["condition", "correct"]].rename(columns={"correct": "c_D"}))
    R = (df[~cl & (df["variant"] == "restored")].drop_duplicates(k + ["condition"])
         [k + ["condition", "family", "level", "correct"]].rename(columns={"correct": "c_R"}))
    p = R.merge(D, on=k + ["condition"]).merge(C, on=k)
    p[["c_C", "c_D", "c_R"]] = p[["c_C", "c_D", "c_R"]].fillna(0.0)  # unreadable = wrong
    Cb, Db, Rb = (p[c] == 1 for c in ("c_C", "c_D", "c_R"))
    fams = list(FAMILIES)
    p = p.assign(anch=Cb, RIE=Cb & Db & ~Rb, RIC=Cb & ~Db & Rb, RIE_any=Db & ~Rb, RIC_any=~Db & Rb,
                 _r=[fams.index(f) * 100 + (FAMILIES[f].index(lv) if lv in FAMILIES[f] else 99)
                     for f, lv in zip(p["family"], p["level"])])

    parts = []
    for data, with_p in ((p, True), (p.assign(condition="ALL", _r=10**6), False)):
        g = data.groupby(["upscaler", "model", "condition"], as_index=False).agg(
            _r=("_r", "first"), n_paired=("c_C", "size"), MAcc_clean=("c_C", "mean"),
            MAcc_degraded=("c_D", "mean"), MAcc_restored=("c_R", "mean"),
            n_anchored=("anch", "sum"), RIE=("RIE", "sum"), RIC=("RIC", "sum"),
            RIE_any=("RIE_any", "sum"), RIC_any=("RIC_any", "sum"),
        )
        g["p_mcnemar"] = ([mcnemar_exact(b, c) for b, c in zip(g["RIE_any"], g["RIC_any"])]
                          if with_p else float("nan"))
        parts.append(g)
    t = pd.concat(parts, ignore_index=True)
    for c in ("MAcc_clean", "MAcc_degraded", "MAcc_restored"):
        t[c] = (t[c] * 100).round(1)
    t["net_any"] = t["RIC_any"] - t["RIE_any"]
    anch = t["n_anchored"].where(t["n_anchored"] > 0)
    t["r"], t["h"] = (t["RIC"] / anch).round(4), (t["RIE"] / anch).round(4)
    flips, flips_any = t["RIE"] + t["RIC"], t["RIE_any"] + t["RIC_any"]
    t["SDS"] = (t["RIE"] / flips.where(flips > 0)).round(4)
    t["SDS_any"] = (t["RIE_any"] / flips_any.where(flips_any > 0)).round(4)
    t["p_mcnemar"] = t["p_mcnemar"].round(4)
    return t[SDS_COLS]


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def draw_panel(ax, agg, fam, upscalers, models, color, style, title=True):
    levels = FAMILIES[fam]
    x_of = {lv: i for i, lv in enumerate(levels)}
    sub = agg[agg["level"].isin(levels) & agg["family"].isin([fam, "clean"])]

    # Shaded reference column for the clean condition
    ax.axvspan(-0.35, 0.35, color=CLEAN_BAND, zorder=0, lw=0)

    for u in upscalers:
        ls, mk = style[u]
        for m in models:
            d = sub[(sub["upscaler"] == u) & (sub["model"] == m)].copy()
            # Skip methods with no degraded points in this family (clean only)
            if d.empty or (d["level"] == "clean").all():
                continue
            d["x"] = d["level"].map(x_of)
            d = d.sort_values("x")
            ax.plot(d["x"], d["MCQ_ACC"], linestyle=ls, marker=mk, color=color[m],
                    lw=1.9, ms=6, mec="white", mew=1.1, zorder=3,
                    solid_capstyle="round", dash_capstyle="round")
            missing = set(levels) - set(d["level"])
            if missing:
                print(f"[warn] {fam}: {u} / {m} missing levels {sorted(missing)}")

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([TICK_FMT[fam](lv) for lv in levels])
    ax.set_xlim(-0.35, len(levels) - 0.65)
    ax.set_xlabel(XLABELS[fam], labelpad=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=3, width=0.8)
    if title:
        ax.set_title(TITLES[fam], loc="left", fontsize=10, fontweight="semibold",
                     color=INK, pad=10)


def legend_handles(upscalers, models, color, style):
    h_models = [Line2D([], [], color=color[m], lw=2.2, solid_capstyle="round", label=m)
                for m in models]
    h_ups = [Line2D([], [], color=INK_2, linestyle=style[u][0], marker=style[u][1],
                    lw=1.6, ms=5.5, mec="white", mew=1.0, label=u)
             for u in upscalers]
    return h_models, h_ups


LEGEND_KW = dict(frameon=False, handlelength=2.6, columnspacing=1.1,
                 handletextpad=0.45, borderaxespad=0, labelcolor=INK)


def add_combined_legend(fig, h_models, h_ups):
    """Single centred row above the figure: models, a gap, then upscalers."""
    spacer = Line2D([], [], color="none", label="")
    handles = h_models + [spacer] + h_ups
    leg = fig.legend(handles=handles, ncol=len(handles), loc="lower center",
                     bbox_to_anchor=(0.5, 0.97), bbox_transform=fig.transFigure,
                     **LEGEND_KW)
    return [leg]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path, help="One folder per restoration method")
    ap.add_argument("--out", type=Path, default=Path("."), help="Output directory")
    ap.add_argument("--fmt", default="pdf", help="pdf / png / svg")
    ap.add_argument("--families", nargs="+", choices=list(FAMILIES), default=None,
                    help="Families to plot, in panel order (default: all with data)")
    ap.add_argument("--ncols", type=int, default=None,
                    help="Columns in the combined grid (default: all panels in one row)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="Keep duplicate (upscaler, model, condition, key) rows")
    ap.add_argument("--no-title", action="store_true", help="Omit panel titles")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([load_folder(f) for f in args.folders], ignore_index=True)

    # --- MCQ rows only -------------------------------------------------------
    is_mcq = df["type"].astype(str).str.strip().str.lower().isin(MCQ_TYPES)
    if not is_mcq.any():
        raise ValueError(
            f"No rows matched MCQ_TYPES={MCQ_TYPES}. "
            f"Observed `type` values: {sorted(df['type'].astype(str).unique())}"
        )
    df = df[is_mcq].copy()
    df["correct"] = to_bool(df["correct"])
    has_variant = "variant" in df.columns
    if has_variant:
        df["variant"] = df["variant"].astype(str).str.strip().str.lower()

    # --- Dedup (multiple CSVs per folder can overlap) ------------------------
    if not args.no_dedup:
        dkeys = ["upscaler", "model", "condition", "key"] + (["variant"] if has_variant else [])
        n_dup = df.duplicated(dkeys).sum()
        if n_dup:
            print(f"[warn] dropping {n_dup} duplicate rows on {dkeys}")
            df = df.drop_duplicates(dkeys, keep="last")

    # --- Same video set: clean answer must exist in every folder -------------
    # Per model, keep only items with a clean row in every input folder, then
    # keep all of those items' degraded / restored rows.
    if "item_id" not in df.columns:
        raise ValueError("`item_id` column needed to match videos across folders")
    df["item_id"] = df["item_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    folders = list(dict.fromkeys(f.name for f in args.folders))
    is_clean = df["condition"].map(lambda c: parse_condition(c)[0] == "clean")
    video_ok = to_bool(df["video_ok"]).ne(0) if "video_ok" in df.columns else True
    clean = df[is_clean & video_ok]
    n_have = clean.groupby(["model", "item_id"])["upscaler"].nunique()
    common = n_have[n_have == len(folders)].reset_index()[["model", "item_id"]]
    counts = (clean.groupby(["model", "upscaler"])["item_id"].nunique().unstack("upscaler")
              .reindex(columns=folders).fillna(0).astype(int))
    counts["in_all"] = common.groupby("model").size().reindex(counts.index).fillna(0).astype(int)
    print("[items] videos with a clean answer, per folder and in all folders:")
    print(counts.rename(index=pretty_model).to_string())
    for m in sorted(set(df["model"]) - set(common["model"])):
        print(f"[warn] {m}: no video has a clean answer in every folder; model dropped")
    if common.empty:
        raise ValueError(f"No video has a clean answer in all of {folders}")
    n0 = len(df)
    df = df.merge(common, on=["model", "item_id"])
    print(f"[items] kept {len(df)}/{n0} rows")

    # --- Parse condition (temporal folders -> frame-rate parser) -------------
    temporal = df["upscaler"].map(is_temporal_folder)
    parsed = pd.Series(
        [parse_fps_condition(c) if t else parse_condition(c)
         for c, t in zip(df["condition"], temporal)],
        index=df.index,
    )
    df["family"] = parsed.str[0]
    df["level"] = parsed.str[1]
    unk = df.loc[df["family"].isna(), ["upscaler", "condition"]].drop_duplicates()
    for u, g in unk.groupby("upscaler"):
        print(f"[warn] {u}: unrecognised conditions ignored: {sorted(map(str, g['condition']))}")
    df = df[df["family"].notna()]
    for fam in FAMILIES:
        extra_lv = set(df.loc[df["family"] == fam, "level"]) - set(FAMILIES[fam])
        if extra_lv:
            print(f"[warn] {fam}: levels {sorted(extra_lv)} present but not in FAMILIES; not plotted")

    # --- SDS per (folder, model, condition) ----------------------------------
    if has_variant:
        sds = sds_table(df)
        u_rank = {u: i for i, u in enumerate(folders)}
        m_rank = {m: i for i, m in enumerate(PRETTY_MODELS)}
        sds = (sds.assign(_u=sds["upscaler"].map(u_rank),
                          _m=sds["model"].map(lambda m: m_rank.get(m, len(m_rank))))
               .sort_values(["_u", "_m", "model", "_r"]).drop(columns=["_u", "_m"]))
        conds = sds.drop_duplicates("condition").sort_values("_r")["condition"].tolist()
        sds = sds.drop(columns="_r").assign(upscaler=sds["upscaler"].map(pretty_upscaler),
                                            model=sds["model"].map(pretty_model))
        sds.to_csv(args.out / "sds_by_condition.csv", index=False)
        rows = list(dict.fromkeys(zip(sds["upscaler"], sds["model"])))
        (sds.pivot(index=["upscaler", "model"], columns="condition", values="SDS")
         .reindex(index=rows, columns=conds).round(3).to_csv(args.out / "sds_wide.csv"))
        print("\nRIE = clean ok, degraded ok, restored wrong | RIC = clean ok, degraded wrong, restored ok")
        print("SDS = RIE/(RIE+RIC) on clean-correct items | *_any = same flip regardless of clean | "
              "p = exact McNemar, degraded vs restored")
        with pd.option_context("display.width", 250, "display.max_columns", None,
                               "display.max_rows", None):
            print(sds.to_string(index=False))
        # Accuracy lines are the restored answers; degraded rows only feed SDS
        df = df[(df["family"] == "clean") | (df["variant"] != "degraded")]
    else:
        print("[warn] no `variant` column: degraded and restored rows can't be told apart; "
              "SDS skipped")

    # --- Families to plot ----------------------------------------------------
    with_data = [f for f in FAMILIES
                 if df.loc[df["family"] == f, "level"].isin(FAMILIES[f]).any()]
    if args.families:
        fams = list(dict.fromkeys(args.families))
        for f in set(fams) - set(with_data):
            print(f"[warn] --families {f}: no data; panel will be empty")
    else:
        fams = with_data
    if not fams:
        raise ValueError("No plottable families found in the input folders")

    # --- Aggregate: MCQ_ACC per (upscaler, model, family, level) -------------
    # Unparsed predictions are kept and count as incorrect (correct == 0).
    agg = (
        df.groupby(["upscaler", "model", "family", "level"], as_index=False)
        .agg(MCQ_ACC=("correct", "mean"), n=("correct", "size"))
    )
    agg["MCQ_ACC"] *= 100
    agg.to_csv(args.out / "mcq_acc_by_condition.csv", index=False)

    # --- Display names -------------------------------------------------------
    agg["upscaler"] = agg["upscaler"].map(pretty_upscaler)
    agg["model"] = agg["model"].map(pretty_model)

    upscalers = list(dict.fromkeys(pretty_upscaler(f.name) for f in args.folders))
    known = [v for v in PRETTY_MODELS.values() if v in set(agg["model"])]
    models = known + sorted(set(agg["model"]) - set(known))
    if len(models) > len(COLORS):
        print(f"[warn] {len(models)} models > {len(COLORS)} colours; colours will repeat")
    color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}
    style = {u: UPSCALER_STYLES[i % len(UPSCALER_STYLES)] for i, u in enumerate(upscalers)}
    # Methods actually drawn in each family (have >= 1 non-clean point there)
    fam_ups = {
        f: [u for u in upscalers
            if ((agg["family"] == f) & (agg["upscaler"] == u)
                & agg["level"].isin(FAMILIES[f])).any()]
        for f in fams
    }
    drawn = [u for u in upscalers if any(u in fam_ups[f] for f in fams)]
    h_models, h_ups = legend_handles(drawn, models, color, style)
    show_title = not args.no_title

    # --- Individual panels ---------------------------------------------------
    for fam in fams:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        draw_panel(ax, agg, fam, upscalers, models, color, style, title=False)
        ax.set_ylabel("MCQ accuracy (%)")
        # Two stacked rows above the axes: models, then this panel's methods
        _, h_ups_fam = legend_handles(fam_ups[fam] or drawn, models, color, style)
        l1 = ax.legend(handles=h_models, ncol=min(len(h_models), 4), loc="lower left",
                       bbox_to_anchor=(0, 1.11), **{**LEGEND_KW, "handlelength": 1.5,
                                                     "columnspacing": 0.9})
        ax.add_artist(l1)
        l2 = ax.legend(handles=h_ups_fam, ncol=len(h_ups_fam), loc="lower left",
                       bbox_to_anchor=(0, 1.02), **LEGEND_KW)
        extra = [l1, l2]
        if show_title:
            t = ax.text(0, 1.22, TITLES[fam], transform=ax.transAxes, ha="left",
                        va="bottom", fontsize=10.5, fontweight="semibold", color=INK)
            extra.append(t)
        out = args.out / f"mcq_acc_{fam}.{args.fmt}"
        fig.savefig(out, bbox_inches="tight", bbox_extra_artists=extra, pad_inches=0.04)
        plt.close(fig)
        print(f"saved {out}")

    # --- Combined figure (single row by default; grid via --ncols) -----------
    ncols = max(1, min(args.ncols or len(fams), len(fams)))
    nrows = math.ceil(len(fams) / ncols)
    # Column width = widest family placed in that column
    widths = [max(len(FAMILIES[fams[i]]) - 0.3 for i in range(c, len(fams), ncols))
              for c in range(ncols)]
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(IN_PER_UNIT * sum(widths), 3.3 * nrows),
        sharey=True, squeeze=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.08,
                     "hspace": 0.75 if show_title else 0.55},
    )
    for i, ax in enumerate(axes.flat):
        if i >= len(fams):
            ax.set_visible(False)
            continue
        draw_panel(ax, agg, fams[i], upscalers, models, color, style, title=show_title)
        if i % ncols == 0:
            ax.set_ylabel("MCQ accuracy (%)")
        else:
            ax.tick_params(axis="y", labelleft=False)
    legs = add_combined_legend(fig, h_models, h_ups)
    out = args.out / f"mcq_acc_combined.{args.fmt}"
    fig.savefig(out, bbox_inches="tight", bbox_extra_artists=legs, pad_inches=0.04)
    plt.close(fig)
    print(f"saved {out}")
    print(f"saved {args.out / 'mcq_acc_by_condition.csv'}")
    if has_variant:
        print(f"saved {args.out / 'sds_by_condition.csv'}")
        print(f"saved {args.out / 'sds_wide.csv'}")


if __name__ == "__main__":
    main()