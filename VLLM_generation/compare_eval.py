#!/usr/bin/env python3
"""Paired comparison of result CSVs from the *_generation.py scripts.

Keeps only items scored for every (model, condition) in the given files, then
prints VSTAT's metrics (x100) per model x condition:
  ALL_Score_avg  mean over all questions (MCQ 0/1, numeric MRA) -- the paper's "Avg."
  MCQ_ACC        non-numeric questions only
  Numeric_MRA    numeric questions only
plus parse rate and the fraction of items whose output never changes across conditions.

Usage:
    python compare_models.py results_dir/*_results.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from vlm_eval_common import CONDITIONS, load_scored, vstat_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    df = pd.concat([load_scored(p) for p in args.files], ignore_index=True)
    df = df.drop_duplicates(["model", "item_id", "condition"], keep="last").copy()

    n_m, n_c = df["model"].nunique(), df["condition"].nunique()
    cells = df.groupby("item_id").size()
    df = df[df["item_id"].isin(cells[cells == n_m * n_c].index)]
    n_items = df["item_id"].nunique()
    print(f"{n_m} models x {n_c} conditions, {n_items} fully paired items\n")
    if df.empty:
        return

    order = [c for c in CONDITIONS if c in set(df["condition"])] + \
            sorted(set(df["condition"]) - set(CONDITIONS))
    metrics = {(m, c): vstat_metrics(g) for (m, c), g in df.groupby(["model", "condition"])}
    for metric in ("ALL_Score_avg", "MCQ_ACC", "Numeric_MRA"):
        table = pd.Series({k: v[metric] for k, v in metrics.items()}).unstack()
        if table.isna().all().all():
            continue
        print(f"{metric} (x100)")
        print(table[order].round(1).to_string() + "\n")

    stats = df.groupby("model").agg(parse_rate=("parse_ok", lambda s: (s == "True").mean()))
    stats["identical_across_conditions"] = df.groupby(["model", "item_id"])["raw_output"] \
        .nunique().eq(1).groupby("model").mean()
    print(stats.round(3).to_string())


if __name__ == "__main__":
    main()