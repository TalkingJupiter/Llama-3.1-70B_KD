#!/usr/bin/env python3
"""
plot_eval.py — Turn LM Eval Harness JSON into tidy CSVs + readable charts

Usage:
  python plot_eval.py --json path/to/results.json [--out out_dir]

What you get in --out (default: ./eval_plots):
  - task_scores.csv                 # one row per task with a primary metric
  - group_scores.csv                # one row per group (e.g., mmlu_stem)
  - top10_subtasks.csv / bottom10_subtasks.csv
  - core_tasks.png                  # ARC, HellaSwag, BBH, MMLU (if present)
  - mmlu_domain_breakdown.png       # mmlu_* domains (if present)
  - subtasks_top40.png              # top 40 non-aggregate tasks
  - subtasks_bottom40.png           # bottom 40 non-aggregate tasks
  - all_tasks_overview.png          # top 60 tasks overall
"""

import argparse, json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------- Helpers -----------------------
def pick_primary_metric(metrics: Dict[str, Any]) -> Tuple[str, float]:
    """
    Choose a single primary metric for a task/group.
    Preference:
      1) exact_match,get-answer
      2) acc_norm,none
      3) acc,none
      4) first numeric metric
    Returns (metric_name, value)
    """
    numeric_items = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    for pref in ["exact_match,get-answer", "acc_norm,none", "acc,none"]:
        if pref in numeric_items:
            return pref, float(numeric_items[pref])
    for k, v in numeric_items.items():
        return k, float(v)
    return "n/a", float("nan")


def stderr_for(metric_name: str) -> str | None:
    if metric_name == "exact_match,get-answer":
        return "exact_match_stderr,get-answer"
    if metric_name == "acc_norm,none":
        return "acc_norm_stderr,none"
    if metric_name == "acc,none":
        return "acc_stderr,none"
    return None


def build_results_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for task, metrics in results.items():
        metric_name, value = pick_primary_metric(metrics)
        key_stderr = stderr_for(metric_name)
        rows.append({
            "task": task,
            "primary_metric": metric_name,
            "score": value,
            "stderr": float(metrics.get(key_stderr)) if key_stderr and isinstance(metrics.get(key_stderr), (int, float)) else None
        })
    return pd.DataFrame(rows).sort_values("task").reset_index(drop=True)


def build_groups_table(groups: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for grp, metrics in groups.items():
        metric_name, value = pick_primary_metric(metrics)
        key_stderr = stderr_for(metric_name)
        rows.append({
            "group": grp,
            "primary_metric": metric_name,
            "score": value,
            "stderr": float(metrics.get(key_stderr)) if key_stderr and isinstance(metrics.get(key_stderr), (int, float)) else None
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, outfile: Path,
             err_col: str | None = None, rotate: int = 30) -> None:
    """Simple bar plot (no seaborn, one chart per figure, no explicit colors)."""
    if df.empty:
        return
    plt.figure()
    x_vals = df[x_col].tolist()
    y_vals = df[y_col].tolist()
    if err_col and err_col in df.columns and df[err_col].notna().any():
        yerr = df[err_col].fillna(0.0).tolist()
        plt.bar(x_vals, y_vals, yerr=yerr)
    else:
        plt.bar(x_vals, y_vals)
    plt.title(title)
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Score")
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path, help="Path to LM Eval Harness JSON results file.")
    ap.add_argument("--out", type=Path, default=Path("eval_plots"), help="Output directory for CSVs/plots.")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: Dict[str, Dict[str, Any]] = data.get("results", {})
    groups: Dict[str, Dict[str, Any]] = data.get("groups", {})
    # Not strictly required, but available if you want to expand:
    # group_subtasks: Dict[str, List[str]] = data.get("group_subtasks", {})
    # configs: Dict[str, Dict[str, Any]] = data.get("configs", {})

    df_tasks = build_results_table(results)
    df_groups = build_groups_table(groups)

    # Save CSVs
    tasks_csv = out_dir / "task_scores.csv"
    groups_csv = out_dir / "group_scores.csv"
    df_tasks.to_csv(tasks_csv, index=False)
    df_groups.to_csv(groups_csv, index=False)

    # Convenience views
    core_keys = ["arc_challenge", "hellaswag", "bbh", "mmlu"]
    df_core = df_tasks[df_tasks["task"].isin(core_keys)].copy()

    mmlu_domains = ["mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"]
    df_mmlu_domains = df_groups[df_groups["group"].isin(mmlu_domains)].copy()

    # Subtasks only (exclude core aggregates)
    df_subtasks_only = df_tasks[~df_tasks["task"].isin(core_keys)]
    top10 = df_subtasks_only.sort_values("score", ascending=False).head(10).copy()
    bot10 = df_subtasks_only.sort_values("score", ascending=True).head(10).copy()
    top10.to_csv(out_dir / "top10_subtasks.csv", index=False)
    bot10.to_csv(out_dir / "bottom10_subtasks.csv", index=False)

    # Plots
    if not df_core.empty:
        bar_plot(df_core, "task", "score", "Core Tasks — Primary Scores",
                 out_dir / "core_tasks.png", err_col="stderr", rotate=10)

    if not df_mmlu_domains.empty:
        bar_plot(df_mmlu_domains, "group", "score", "MMLU Domain Breakdown",
                 out_dir / "mmlu_domain_breakdown.png", err_col="stderr", rotate=10)

    # Top 40 / Bottom 40 subtasks
    if not df_subtasks_only.empty:
        best40 = df_subtasks_only.sort_values("score", ascending=False).head(40)
        worst40 = df_subtasks_only.sort_values("score", ascending=True).head(40)
        bar_plot(best40, "task", "score", "Top 40 Subtasks — Primary Scores",
                 out_dir / "subtasks_top40.png", rotate=75)
        bar_plot(worst40, "task", "score", "Bottom 40 Subtasks — Primary Scores",
                 out_dir / "subtasks_bottom40.png", rotate=75)

    # Overview (top 60)
    bar_plot(df_tasks.sort_values("score", ascending=False).head(60),
             "task", "score", "Tasks Overview (Top 60)",
             out_dir / "all_tasks_overview.png", rotate=75)

    # Small summary
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("LM Eval Harness Plot Pack\n")
        f.write(f"Source file: {args.json.name}\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("Artifacts:\n")
        f.write(f"- CSV: {tasks_csv}\n")
        f.write(f"- CSV: {groups_csv}\n")
        f.write(f"- CSV: {out_dir / 'top10_subtasks.csv'}\n")
        f.write(f"- CSV: {out_dir / 'bottom10_subtasks.csv'}\n")
        f.write(f"- PNG: {out_dir / 'core_tasks.png'}\n")
        f.write(f"- PNG: {out_dir / 'mmlu_domain_breakdown.png'}\n")
        f.write(f"- PNG: {out_dir / 'subtasks_top40.png'}\n")
        f.write(f"- PNG: {out_dir / 'subtasks_bottom40.png'}\n")
        f.write(f"- PNG: {out_dir / 'all_tasks_overview.png'}\n")

    print(f"[OK] Wrote CSVs and plots to: {out_dir.resolve()}")
    if not df_core.empty:
        print("\nCore tasks:")
        print(df_core[["task", "primary_metric", "score", "stderr"]].to_string(index=False))


if __name__ == "__main__":
    main()
