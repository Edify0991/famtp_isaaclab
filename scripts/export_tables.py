"""Export publication tables (CSV + LaTeX + optional bootstrap CI)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["fall_rate", "mean_survival_time", "joint_jerk_integral", "com_jerk_integral", "torque_rate_integral", "switch_success_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-exp", type=str, default="week1_single_skill")
    parser.add_argument("--switch-exp", type=str, default="week1_random_switch")
    parser.add_argument("--output-dir", type=str, default="outputs/tables/baseline_comparison")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    return parser.parse_args()


def _bootstrap_ci(values: np.ndarray, samples: int = 1000) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    means = []
    for _ in range(samples):
        res = rng.choice(values, size=values.size, replace=True)
        means.append(np.mean(res))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    single = pd.read_csv(Path("outputs/eval") / args.single_exp / "metrics.csv")
    switch = pd.read_csv(Path("outputs/eval") / args.switch_exp / "metrics.csv")
    df = pd.concat([single, switch], ignore_index=True)

    grouped = df.groupby(["method", "mode"])  # one row per seed/run expected
    rows = []
    for (method, mode), g in grouped:
        row = {"method": method, "mode": mode}
        for metric in METRICS:
            vals = g[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
            if args.bootstrap_samples > 0:
                lo, hi = _bootstrap_ci(vals, args.bootstrap_samples)
                row[f"{metric}_ci95_lo"] = lo
                row[f"{metric}_ci95_hi"] = hi
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out / "results_mean_std.csv", index=False)
    (out / "results_mean_std.tex").write_text(table.to_latex(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
