"""Plot representative switch-window traces and skill-pair comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--switch-exp", type=str, default="week1_random_switch")
    parser.add_argument("--output-dir", type=str, default="outputs/plots/baseline_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path("outputs/eval") / args.switch_exp / "per_switch_metrics.csv")

    # switch-window time-series proxy by ordering switch id
    for method, g in df.groupby("method"):
        g = g.sort_values("switch_id")
        plt.figure(figsize=(7, 4))
        plt.plot(g["switch_id"], g["joint_jerk_integral"], label="joint_jerk_integral")
        plt.plot(g["switch_id"], g["torque_rate_integral"], label="torque_rate_integral")
        plt.title(f"Representative traces: {method}")
        plt.xlabel("switch_id")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"traces_{method}.png", dpi=150)
        plt.close()

    # per-skill-pair comparison
    if "skill_pair" in df.columns:
        piv = df.groupby(["skill_pair", "method"], as_index=False)["joint_jerk_integral"].mean()
        for pair, sub in piv.groupby("skill_pair"):
            plt.figure(figsize=(6, 4))
            plt.bar(sub["method"], sub["joint_jerk_integral"])
            plt.title(f"Per-skill-pair: {pair}")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(out / f"skillpair_{pair.replace('->','_to_')}.png", dpi=150)
            plt.close()


if __name__ == "__main__":
    main()
