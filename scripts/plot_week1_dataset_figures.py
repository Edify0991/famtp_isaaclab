"""Generate week1 dataset figures for scan/mine/filter/view workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt

from famtp_lab.motion.keyframe_export import export_keyframe_strip, export_root_trajectory_plot
from famtp_lab.motion.motion_npz_loader import load_npz_canonical


SKILLS = ["locomotion_runlike", "dribble_like", "stop_shoot_like"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot week1 dataset figures.")
    parser.add_argument("--scan-summary", type=str, default="datasets/reports/g1_amass_schema_summary.json")
    parser.add_argument("--motion-index", type=str, default="datasets/canonical/week1_motion_index.json")
    parser.add_argument("--output-dir", type=str, default="outputs/reports/week1_feasibility/figures")
    return parser.parse_args()


def _bar_from_dict(d: dict, title: str, out: Path) -> None:
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scan = json.loads(Path(args.scan_summary).read_text()) if Path(args.scan_summary).exists() else {}
    index = json.loads(Path(args.motion_index).read_text()) if Path(args.motion_index).exists() else []

    # 1) schema figures
    keysets = {" + ".join(x["keys"]): x["count"] for x in scan.get("keyset_combinations", [])[:10]}
    _bar_from_dict(keysets or {"none": 0}, "Common key combinations", out_dir / "schema_keysets.png")
    _bar_from_dict(scan.get("fps_counts", {}) or {"unknown": 0}, "FPS/file-pattern stats (fps)", out_dir / "schema_fps.png")

    # 2/3) skill duration/count
    dur = {k: 0.0 for k in SKILLS}
    cnt = {k: 0 for k in SKILLS}
    for r in index:
        s = r.get("skill_label", "unknown")
        dur[s] = dur.get(s, 0.0) + float(r.get("duration", 0.0))
        cnt[s] = cnt.get(s, 0) + 1
    _bar_from_dict(dur, "Skill total duration", out_dir / "skill_duration.png")
    _bar_from_dict(cnt, "Skill clip count", out_dir / "skill_count.png")

    # 4) duration histogram
    durations = [float(r.get("duration", 0.0)) for r in index]
    plt.figure(figsize=(7, 4))
    plt.hist(durations, bins=20)
    plt.title("Clip duration distribution")
    plt.xlabel("seconds")
    plt.tight_layout()
    plt.savefig(out_dir / "duration_hist.png", dpi=150)
    plt.close()

    # 5/6) sampled keyframes and root trajectories
    rng = random.Random(0)
    for skill in SKILLS:
        rows = [r for r in index if r.get("skill_label") == skill]
        sample = rows if len(rows) <= 3 else rng.sample(rows, 3)
        for i, row in enumerate(sample):
            path = row.get("matched_g1_file_path")
            if not path or not Path(path).exists():
                continue
            clip = load_npz_canonical(path)
            export_keyframe_strip(clip, out_dir / f"sample_{skill}_{i}_keyframes.png")
            export_root_trajectory_plot(clip, out_dir / f"sample_{skill}_{i}_roottraj.png")

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
