"""Build no-transition week1 motion index by trimming segment boundaries."""
"""
Week-1 experiment plan command hook:
1) build no-transition dataset
2) train ppo_cmd
3) evaluate single-skill
4) evaluate random-switch
5) generate figures/report

Build week-1 no-transition motion dataset index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim boundary windows from week1 motion index.")
    parser.add_argument("--input", type=str, default="datasets/canonical/week1_motion_index.json")
    parser.add_argument("--output", type=str, default="datasets/canonical/week1_no_transition_motion_index.json")
    parser.add_argument("--summary", type=str, default="datasets/reports/week1_filtering_summary.json")
    parser.add_argument("--trim-start", type=float, default=0.25)
    parser.add_argument("--trim-end", type=float, default=0.25)

from famtp_lab.motion.motion_loader import DEFAULT_SKILLS, build_no_transition_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-transition motion index.")
    parser.add_argument("--input", type=str, default="datasets/motion_index.json")
    parser.add_argument("--output", type=str, default="datasets/no_transition_motion_index.json")
    parser.add_argument("--summary", type=str, default="datasets/no_transition_summary.json")
    parser.add_argument("--boundary-window-s", type=float, default=0.25)
    parser.add_argument(
        "--skills",
        nargs="*",
        default=DEFAULT_SKILLS,
        help="Skills to preserve in no-transition dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.input).read_text())

    kept = []
    removed = 0
    for r in rows:
        st = float(r.get("start_time", 0.0)) + args.trim_start
        en = float(r.get("end_time", 0.0)) - args.trim_end
        if en <= st:
            removed += 1
            continue
        x = dict(r)
        x["start_time"] = st
        x["end_time"] = en
        x["duration"] = en - st
        kept.append(x)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, indent=2))

    summary = {
        "input_count": len(rows),
        "output_count": len(kept),
        "removed_count": removed,
        "trim_start": args.trim_start,
        "trim_end": args.trim_end,
    }
    s = Path(args.summary)
    s.parent.mkdir(parents=True, exist_ok=True)
    s.write_text(json.dumps(summary, indent=2))

    print(f"Saved no-transition index: {out}")
#     summary = build_no_transition_dataset(
#         input_index_path=args.input,
#         output_index_path=args.output,
#         summary_path=args.summary,
#         boundary_window_s=args.boundary_window_s,
#         allowed_skills=args.skills,
#     )
#     print("No-transition dataset built.")
#     print(f"clips: {summary['total_clips_before']} -> {summary['total_clips_after']}")


if __name__ == "__main__":
    main()
