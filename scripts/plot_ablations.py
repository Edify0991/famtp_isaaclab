"""Plot FaMTP ablation bars from evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ABLATION_METHODS = ["famtp_stage1", "famtp_nobridge", "famtp_full"]
METRICS = ["fall_rate", "joint_jerk_integral", "torque_rate_integral", "switch_success_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--switch-exp", type=str, default="week1_random_switch")
    parser.add_argument("--output-dir", type=str, default="outputs/plots/baseline_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path("outputs/eval") / args.switch_exp / "metrics.csv")
    df = df[df["method"].isin(ABLATION_METHODS)]

    for metric in METRICS:
        plt.figure(figsize=(6, 4))
        plt.bar(df["method"], df[metric])
        plt.title(f"Ablation: {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(out / f"ablation_{metric}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
