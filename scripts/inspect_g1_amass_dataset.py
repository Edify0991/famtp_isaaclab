"""Inspect local AMASS_Retargeted_for_G1 dataset schemas and produce reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from famtp_lab.motion.schema_inspector import inspect_npz_file, summarize_scan


MOTION_SUFFIXES = {".npz", ".npy", ".pkl", ".pt", ".json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect G1 AMASS retargeted dataset.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--report-dir", type=str, default="datasets/reports")
    return parser.parse_args()


def _scan_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in MOTION_SUFFIXES:
            files.append(path)
    return sorted(files)


def _md_summary(scan: list[dict], summary: dict) -> str:
    lines = ["# G1 AMASS scan summary", ""]
    lines.append(f"- scanned files: **{len(scan)}**")
    lines.append(f"- suffix counts: `{summary.get('suffix_counts', {})}`")
    lines.append(f"- fps counts: `{summary.get('fps_counts', {})}`")
    lines.append("")
    lines.append("## Key frequency (top)")
    for key, count in list(summary.get("key_frequency", {}).items())[:20]:
        lines.append(f"- `{key}`: {count}")
    lines.append("")
    lines.append("## Keyset combinations (top)")
    for row in summary.get("keyset_combinations", [])[:10]:
        lines.append(f"- count={row['count']}, keys={row['keys']}")
    lines.append("")
    lines.append("## Field candidates")
    for name, vals in summary.get("field_candidates", {}).items():
        lines.append(f"- {name}: {vals}")
    lines.append("")
    lines.append("## File name patterns (top)")
    for row in summary.get("file_name_patterns", [])[:15]:
        lines.append(f"- {row['pattern']}: {row['count']}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    files = _scan_files(data_root)
    reports = []
    for path in files:
        entry = {"path": path.as_posix(), "suffix": path.suffix, "num_frames": None, "fps": None, "duration_s": None}
        if path.suffix.lower() == ".npz":
            try:
                entry = inspect_npz_file(path)
            except Exception as exc:
                entry["error"] = str(exc)
        reports.append(entry)

    summary = summarize_scan([r for r in reports if r.get("suffix", "").lower() == ".npz" and "keys" in r])

    (report_dir / "g1_amass_scan.json").write_text(json.dumps(reports, indent=2, default=str))
    (report_dir / "g1_amass_schema_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (report_dir / "g1_amass_scan_summary.md").write_text(_md_summary(reports, summary))

    print(f"Scanned files: {len(reports)}")
    print(f"Saved reports to: {report_dir}")


if __name__ == "__main__":
    main()
