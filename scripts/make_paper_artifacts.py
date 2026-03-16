"""Generate publication artifact bundle from existing evaluation outputs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper plots/tables bundle.")
    parser.add_argument("--single-exp", type=str, default="week1_single_skill")
    parser.add_argument("--switch-exp", type=str, default="week1_random_switch")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    Path("outputs/paper_artifacts").mkdir(parents=True, exist_ok=True)

    run(["python", "scripts/plot_switch_metrics.py", "--single-exp", args.single_exp, "--switch-exp", args.switch_exp], args.dry_run)
    run(["python", "scripts/plot_ablations.py", "--switch-exp", args.switch_exp], args.dry_run)
    run(["python", "scripts/plot_representative_traces.py", "--switch-exp", args.switch_exp], args.dry_run)
    run(["python", "scripts/plot_famtp_stage1_latents.py", "--experiment-name", args.switch_exp], args.dry_run)
    run(["python", "scripts/plot_reward_decomposition.py", "--experiment-name", args.switch_exp], args.dry_run)
    run(["python", "scripts/export_tables.py", "--single-exp", args.single_exp, "--switch-exp", args.switch_exp], args.dry_run)


if __name__ == "__main__":
    main()
