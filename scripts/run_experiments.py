"""Batch experiment runner for training baseline and FaMTP methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

METHODS = ["ppo_cmd", "fullbody_amp", "partwise_raw", "famtp_stage1", "famtp_nobridge", "famtp_full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training experiments in batch.")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_dir = Path("outputs/experiments/train_runs")
    meta_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for method in args.methods:
        for seed in args.seeds:
            run_name = f"method-{method}_seed-{seed}_iters-{args.max_iterations}"
            cmd = [
                "python",
                "scripts/rsl_rl/train.py",
                "--task",
                "FaMTP-Humanoid-Switch-Direct-v0",
                "--prior-mode",
                method,
                "--seed",
                str(seed),
                "--max-iterations",
                str(args.max_iterations),
                "--headless",
            ]
            records.append({"run_name": run_name, "method": method, "seed": seed, "cmd": cmd})
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    (meta_dir / "train_runs.json").write_text(json.dumps(records, indent=2))
    print(f"Saved metadata to {meta_dir / 'train_runs.json'}")


if __name__ == "__main__":
    main()
