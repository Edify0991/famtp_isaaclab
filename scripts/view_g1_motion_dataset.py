"""Interactive G1 motion dataset viewer with Isaac Lab first, terminal fallback second."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import select
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from famtp_lab.motion.g1_motion_player import G1MotionPlayer
from famtp_lab.motion.keyframe_export import export_keyframe_strip, export_root_trajectory_plot
from famtp_lab.motion.motion_npz_loader import load_npz_canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View G1 retargeted motion clips.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--clip", type=str, default=None)
    parser.add_argument("--motion-index", type=str, default=None)
    parser.add_argument("--skill", type=str, default=None)
    parser.add_argument("--robot-asset-path", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-clips", type=int, default=200)
    parser.add_argument("--output-screenshot-dir", type=str, default="outputs/viewer_screenshots")
    parser.add_argument("--output-keyframe-dir", type=str, default="outputs/viewer_keyframes")
    parser.add_argument("--output-plot-dir", type=str, default="outputs/viewer_plots")
    parser.add_argument("--terminal-only", action="store_true")
    return parser.parse_args()


def _discover_npz(root: Path, max_clips: int) -> list[Path]:
    files = [p for p in root.rglob("*.npz") if p.is_file()]
    files.sort()
    if len(files) > max_clips:
        files = files[:max_clips]
    return files


def _load_from_motion_index(index_path: Path, skill: str | None) -> list[dict]:
    data = json.loads(index_path.read_text())
    entries = data["entries"] if isinstance(data, dict) and "entries" in data else data
    if skill:
        entries = [e for e in entries if e.get("skill_label") == skill]
    clips = []
    for e in entries:
        path = e.get("matched_g1_file_path") or e.get("clip_path") or e.get("source_file")
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        clip = load_npz_canonical(p)
        clip["skill_label"] = e.get("skill_label")
        clip["clip_id"] = e.get("clip_id", p.stem)
        clips.append(clip)
    return clips


def _load_clips(args: argparse.Namespace) -> list[dict]:
    if args.clip:
        clip = load_npz_canonical(args.clip)
        clip["clip_id"] = Path(args.clip).stem
        return [clip]

    if args.motion_index:
        clips = _load_from_motion_index(Path(args.motion_index), args.skill)
        if clips:
            return clips

    if not args.data_root:
        raise ValueError("Need --data-root or --clip or --motion-index.")
    files = _discover_npz(Path(args.data_root), args.max_clips)
    clips = []
    for path in files:
        try:
            clip = load_npz_canonical(path)
            clip["clip_id"] = path.stem
            clips.append(clip)
        except Exception as exc:
            print(f"[WARN] failed to load {path}: {exc}")
    if args.skill:
        clips = [c for c in clips if c.get("skill_label") == args.skill]
    return clips


def _save_current_frame_png(player: G1MotionPlayer, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = player.get_frame_payload()
    name = f"{Path(player.current_clip.get('source_file','clip')).stem}_f{payload['frame_idx']:06d}.png"
    out = out_dir / name
    j = np.asarray(payload["joint_pos"]).reshape(-1)
    if j.size == 0:
        j = np.zeros(1)
    plt.figure(figsize=(8, 2.2))
    plt.plot(j)
    plt.title(f"frame={payload['frame_idx']} / {payload['num_frames']}")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out.as_posix()


def _overlay(player: G1MotionPlayer) -> str:
    clip = player.current_clip
    fps = clip.get("fps", 120)
    frame = player.state.frame_idx
    n = max(1, int(clip.get("num_frames", 1)))
    t = frame / max(float(fps), 1.0)
    dur = n / max(float(fps), 1.0)
    fields = ["joint_pos", "root_pos", "root_quat"]
    skill = clip.get("skill_label", "N/A")
    return (
        f"clip={clip.get('clip_id', Path(clip.get('source_file','')).name)} | frame={frame}/{n} | "
        f"time={t:.2f}/{dur:.2f}s | fps={fps} | speed={player.state.speed:.2f}x | loop={player.state.loop} | "
        f"fields={fields} | skill={skill}"
    )


def _process_command(cmd: str, player: G1MotionPlayer, args: argparse.Namespace) -> bool:
    """Handle keyboard-style commands. Return False to quit."""
    out_ss = Path(args.output_screenshot_dir)
    out_kf = Path(args.output_keyframe_dir)
    out_pl = Path(args.output_plot_dir)

    mapping = {
        "space": player.toggle_play,
        "left": lambda: player.step_frames(-1),
        "right": lambda: player.step_frames(1),
        "shift+left": lambda: player.step_frames(-10),
        "shift+right": lambda: player.step_frames(10),
        "up": player.prev_clip,
        "down": player.next_clip,
        "[": player.speed_down,
        "]": player.speed_up,
        "l": player.toggle_loop,
        "r": player.reset_clip,
        "g": player.random_clip,
        "i": player.toggle_overlay,
    }

    c = cmd.strip().lower()
    if c in {"q", "esc"}:
        return False
    if c == "k":
        print(f"saved screenshot: {_save_current_frame_png(player, out_ss)}")
        return True
    if c == "p":
        clip = player.current_clip
        out = out_kf / f"{Path(clip.get('source_file','clip')).stem}_keyframes.png"
        print(f"saved keyframe strip: {export_keyframe_strip(clip, out)}")
        return True
    if c == "t":
        clip = player.current_clip
        out = out_pl / f"{Path(clip.get('source_file','clip')).stem}_root_traj.png"
        print(f"saved trajectory: {export_root_trajectory_plot(clip, out)}")
        return True

    action = mapping.get(c)
    if action is not None:
        action()
    return True


def _terminal_viewer(player: G1MotionPlayer, args: argparse.Namespace) -> None:
    print("Terminal control mode. Commands: space/left/right/shift+left/shift+right/up/down/[ ]/l/r/g/i/k/p/t/q")
    alive = True
    while alive:
        if player.state.show_overlay:
            print(_overlay(player))
        player.tick()
        cmd = input("cmd> ").strip()
        if cmd:
            alive = _process_command(cmd, player, args)


def _isaaclab_viewer(player: G1MotionPlayer, args: argparse.Namespace) -> None:
    """Main path: Isaac Lab app loop + command polling from terminal.

    Keyboard hotkeys are mirrored as terminal commands when direct key events are
    unavailable in this external script context.
    """
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Isaac Lab unavailable ({exc}); falling back to terminal mode")
        _terminal_viewer(player, args)
        return

    app_launcher = AppLauncher(headless=args.headless)
    sim_app = app_launcher.app
    print("Isaac viewer started. Input commands in terminal (space/left/right/...) to control playback.")

    alive = True
    while alive and sim_app.is_running():
        if player.state.show_overlay:
            print(_overlay(player))

        if select.select([sys.stdin], [], [], 0.01)[0]:
            cmd = sys.stdin.readline().strip()
            if cmd:
                alive = _process_command(cmd, player, args)

        # frame payload is prepared here; exact articulation mapping is dataset-schema dependent.
        # this script prioritizes robust visualization and export-first workflow.
        _ = player.get_frame_payload()
        player.tick()
        time.sleep(0.02)

    sim_app.close()


def main() -> None:
    args = parse_args()
    clips = _load_clips(args)
    if not clips:
        print("No clips available after filtering/loading.")
        return

    print(f"Loaded {len(clips)} clips")
    player = G1MotionPlayer(clips)

    if args.terminal_only:
        _terminal_viewer(player, args)
    else:
        _isaaclab_viewer(player, args)


if __name__ == "__main__":
    main()
