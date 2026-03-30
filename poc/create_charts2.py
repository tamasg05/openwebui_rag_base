#!/usr/bin/env python
"""
Plot metric sums, means, and medians vs topK, with one line per temperature.

- Input: CSV produced by the evaluation script: compute_relevant_chunks.py (with columns 
  total_relevant_chunks, avg_relevant_chunks, median_relevant_chunks, 
  sum_sum_of_relevance_scores, avg_sum_of_relevance_scores, median_sum_of_relevance_scores, std_sum_of_relevance_scores,
  sum_faithfulness, avg_faithfulness, median_faithfulness, etc.).

- Output: One PNG per (metric, statistic) into the given output directory.
  X-axis: topK
  Y-axis: metric value (starts at 0)
  Lines: different temperatures (e.g. 0.3, 0.6, 0.9, 1.2)
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    col_name: str,
    stat_label: str,
    outdir: Path,
) -> None:
    """
    Plot one chart for a single metric/stat combination.

    X-axis  : topK
    Y-axis  : value of `col_name`
    Lines   : different temperatures
    """

    if col_name not in df.columns:
        print(f"[WARN] Column '{col_name}' not found in CSV. Skipping.")
        return

    # Drop rows where topK or the metric or temperature is NaN
    sub = df.dropna(subset=["topK", col_name, "temperature"])

    if sub.empty:
        print(f"[WARN] No data for column '{col_name}' after dropping NaNs. Skipping.")
        return

    plt.figure()

    for temp in sorted(sub["temperature"].unique()):
        temp_df = sub[sub["temperature"] == temp].sort_values("topK")
        if temp_df.empty:
            continue
        plt.plot(
            temp_df["topK"],
            temp_df[col_name],
            marker="o",
            label=f"T={temp}",
        )

    plt.xlabel("topK")
    pretty_metric = metric_name.replace("_", " ").title()
    plt.ylabel(f"{pretty_metric} ({stat_label})")
    plt.title(f"{pretty_metric} ({stat_label}) vs topK")

    # Y-axis should start at 0
    _, ymax = plt.ylim()
    plt.ylim(bottom=0, top=ymax)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    outdir.mkdir(parents=True, exist_ok=True)
    filename = f"{metric_name}_{stat_label}_vs_topK.png"
    outpath = outdir / filename
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"[OK] Saved plot: {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG charts for metric sums, means, and medians vs topK. "
            "Each chart has a separate line for each temperature."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the metrics CSV file",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        required=True,
        help="Directory where PNG charts will be saved",
    )

    args = parser.parse_args()
    csv_path = Path(args.input)
    outdir = Path(args.outdir)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Map logical metrics to the actual column names in the CSV
    metric_columns = {
        "relevant_chunks": {
            "sum": "total_relevant_chunks",
            "mean": "avg_relevant_chunks",
            "median": "median_relevant_chunks",
        },
        "sum_of_relevance_scores": {
            "sum": "sum_sum_of_relevance_scores",
            "mean": "avg_sum_of_relevance_scores",
            "median": "median_sum_of_relevance_scores",
        },
        "faithfulness": {
            "sum": "sum_faithfulness",
            "mean": "avg_faithfulness",
            "median": "median_faithfulness",
        },
        "factual_recall": {
            "sum": "sum_factual_recall",
            "mean": "avg_factual_recall",
            "median": "median_factual_recall",
        },
        "factual_precision": {
            "sum": "sum_factual_precision",
            "mean": "avg_factual_precision",
            "median": "median_factual_precision",
        },
        "context_recall": {
            "sum": "sum_context_recall",
            "mean": "avg_context_recall",
            "median": "median_context_recall",
        },
    }

    # Create plots for each metric/stat
    for metric_name, cols in metric_columns.items():
        for stat_label, col_name in cols.items():
            plot_metric(
                df=df,
                metric_name=metric_name,
                col_name=col_name,
                stat_label=stat_label,
                outdir=outdir,
            )


if __name__ == "__main__":
    main()