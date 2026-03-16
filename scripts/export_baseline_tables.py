"""Export baseline comparison tables to CSV and Markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "fall_rate",
    "mean_survival_time",
    "joint_jerk_integral",
    "com_jerk_integral",
    "foot_slip_integral_during_contact",
    "torque_rate_integral",
    "switch_success_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export baseline comparison tables.")
    parser.add_argument("--single-exp", type=str, default="week1_single_skill")
    parser.add_argument("--switch-exp", type=str, default="week1_random_switch")
    parser.add_argument("--output-dir", type=str, default="outputs/tables/baseline_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    single = pd.read_csv(Path("outputs/eval") / args.single_exp / "metrics.csv")
    switch = pd.read_csv(Path("outputs/eval") / args.switch_exp / "metrics.csv")
    merged = pd.concat([single, switch], ignore_index=True)

    table = merged[["method", "mode", *METRICS]].sort_values(["method", "mode"])
    table.to_csv(out / "baseline_comparison.csv", index=False)
    (out / "baseline_comparison.md").write_text(table.to_markdown(index=False))

    pivot_frames = []
    for metric in METRICS:
        piv = merged.pivot(index="method", columns="mode", values=metric)
        piv["metric"] = metric
        pivot_frames.append(piv.reset_index())
    all_piv = pd.concat(pivot_frames, ignore_index=True)
    all_piv.to_csv(out / "baseline_metric_pivots.csv", index=False)

    print(f"Saved baseline comparison tables to: {out}")


if __name__ == "__main__":
    main()
