"""G1 motion clip playback state machine and mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass
class PlaybackState:
    """Viewer playback state."""

    clip_idx: int = 0
    frame_idx: int = 0
    playing: bool = True
    loop: bool = True
    speed: float = 1.0
    show_overlay: bool = True


class G1MotionPlayer:
    """Lightweight clip playback manager.

    It provides deterministic frame stepping independent of rendering backend.
    """

    def __init__(self, clips: list[dict]):
        if not clips:
            raise ValueError("No clips provided to G1MotionPlayer.")
        self.clips = clips
        self.state = PlaybackState()

    @property
    def current_clip(self) -> dict:
        return self.clips[self.state.clip_idx]

    def toggle_play(self) -> None:
        self.state.playing = not self.state.playing

    def reset_clip(self) -> None:
        self.state.frame_idx = 0

    def next_clip(self) -> None:
        self.state.clip_idx = (self.state.clip_idx + 1) % len(self.clips)
        self.state.frame_idx = 0

    def prev_clip(self) -> None:
        self.state.clip_idx = (self.state.clip_idx - 1) % len(self.clips)
        self.state.frame_idx = 0

    def random_clip(self) -> None:
        self.state.clip_idx = random.randrange(len(self.clips))
        self.state.frame_idx = 0

    def step_frames(self, n: int) -> None:
        num_frames = max(1, int(self.current_clip.get("num_frames", 1)))
        self.state.frame_idx = int(np.clip(self.state.frame_idx + n, 0, num_frames - 1))

    def speed_up(self) -> None:
        self.state.speed = min(4.0, self.state.speed * 1.25)

    def speed_down(self) -> None:
        self.state.speed = max(0.1, self.state.speed / 1.25)

    def toggle_loop(self) -> None:
        self.state.loop = not self.state.loop

    def toggle_overlay(self) -> None:
        self.state.show_overlay = not self.state.show_overlay

    def tick(self) -> None:
        if not self.state.playing:
            return
        clip = self.current_clip
        num_frames = max(1, int(clip.get("num_frames", 1)))
        step = max(1, int(round(self.state.speed)))
        next_frame = self.state.frame_idx + step
        if next_frame >= num_frames:
            if self.state.loop:
                self.state.frame_idx = 0
            else:
                self.state.frame_idx = num_frames - 1
                self.state.playing = False
        else:
            self.state.frame_idx = next_frame

    def get_frame_payload(self) -> dict:
        """Return current frame payload for rendering and export."""
        clip = self.current_clip
        i = self.state.frame_idx

        def frame(arr, default):
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and len(arr) > i:
                return arr[i]
            return default

        return {
            "root_pos": frame(clip.get("root_pos"), np.zeros(3)),
            "root_quat": frame(clip.get("root_quat"), np.array([0.0, 0.0, 0.0, 1.0])),
            "joint_pos": frame(clip.get("joint_pos"), np.zeros(0)),
            "frame_idx": i,
            "num_frames": clip.get("num_frames", 0),
        }
