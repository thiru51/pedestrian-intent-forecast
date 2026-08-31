"""Cut annotated tracks into fixed-length observation windows with a look-ahead
label.

This is where the "1-2 seconds early" part of the project is actually defined,
so it is worth being precise about it.

An observation window is T consecutive annotated frames ending at frame t_end.
The label answers: at frame t_end + lead, is this pedestrian crossing the ego
vehicle's path? lead defaults to 45 frames, which is 1.5 s at the 30 fps both
datasets were shot at.

Two rules keep the task from being accidentally trivial:

  1. The window must end at or before the pedestrian's crossing_point. JAAD and
     PIE both annotate the frame where the pedestrian steps into the road; any
     window that runs past it is no longer a forecast, it is an observation.
     jaad_data.py's own crossing-sequence builder truncates at exactly the same
     frame, so this matches the reference pipeline.
  2. The target frame must be inside the annotated track. No extrapolating a
     label past the end of the annotation.

Tracks with no behaviour columns (JAAD's "ped" and "people" labels) are skipped
entirely — they have boxes and occlusion but no cross/action tags, so there is
nothing to supervise against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import FPS
from .schema import PedTrack, VideoAnnotation
from ..features.context import CONTEXT_DIM, ContextEncoder
from ..features.gesture import GESTURE_GROUPS, IGNORE_INDEX, apply_phone_override
from ..features.gesture import encode_track as encode_gesture_track
from ..features.pose import POSE_DIM

DEFAULT_WINDOW = 16
DEFAULT_LEAD = 45          # frames, 1.5 s
DEFAULT_STRIDE = 4
DEFAULT_HORIZON = 15       # frames of future trajectory to regress, 0.5 s

# "crossing-irrelevant" (PIE) means the person crossed the road but not in front
# of us. For a planner that is not a crossing event, so it is a negative.
_POSITIVE_CROSS = {"crossing"}


@dataclass
class Window:
    """One training sample, before it becomes tensors."""

    video_id: str
    ped_id: str
    dataset: str
    frames: np.ndarray        # (T,) int64, the observed frame numbers
    pose: np.ndarray          # (T, POSE_DIM) float32
    context: np.ndarray       # (T, CONTEXT_DIM) float32
    gesture: np.ndarray       # (T, 4) int64 class ids, IGNORE_INDEX where absent
    motion: np.ndarray        # (T, 4) float32 normalised box, for the traj head
    future: np.ndarray        # (H, 2) float32 future centre deltas
    future_mask: np.ndarray   # (H,) float32, 1 where the future frame is real
    intent: int               # 1 if crossing at t_end + lead
    lead: int                 # frames of look-ahead actually used
    intention_prob: float     # PIE's human intention rating, -1 when absent

    @property
    def t_end(self) -> int:
        return int(self.frames[-1])


def _normalised_motion(track: PedTrack, idx: np.ndarray, w: float, h: float) -> np.ndarray:
    out = np.zeros((len(idx), 4), dtype=np.float32)
    for k, i in enumerate(idx):
        x1, y1, x2, y2 = track.bbox[int(i)]
        out[k] = (
            (x1 + x2) / 2.0 / w,
            (y1 + y2) / 2.0 / h,
            (x2 - x1) / w,
            (y2 - y1) / h,
        )
    return out


def _future_deltas(
    track: PedTrack, end_idx: int, horizon: int, w: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Future centre positions relative to the centre at t_end, so the head
    predicts displacement and not absolute image position."""
    future = np.zeros((horizon, 2), dtype=np.float32)
    mask = np.zeros(horizon, dtype=np.float32)
    x1, y1, x2, y2 = track.bbox[end_idx]
    base = np.array([(x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h], dtype=np.float32)
    for k in range(horizon):
        j = end_idx + k + 1
        if j >= len(track.bbox):
            break
        fx1, fy1, fx2, fy2 = track.bbox[j]
        future[k] = (
            (fx1 + fx2) / 2.0 / w - base[0],
            (fy1 + fy2) / 2.0 / h - base[1],
        )
        mask[k] = 1.0
    return future, mask


def _crossing_at(track: PedTrack, idx: int) -> int | None:
    if idx >= len(track.cross):
        return None
    value = (track.cross[idx] or "").strip().lower()
    if not value:
        return None
    return 1 if value in _POSITIVE_CROSS else 0


def track_windows(
    video: VideoAnnotation,
    track: PedTrack,
    context: np.ndarray,
    gesture: np.ndarray,
    window: int = DEFAULT_WINDOW,
    lead: int = DEFAULT_LEAD,
    stride: int = DEFAULT_STRIDE,
    horizon: int = DEFAULT_HORIZON,
    pose: np.ndarray | None = None,
) -> list[Window]:
    """Windows for one track. `pose` is optional: without the video files there
    are no pose vectors, so it is filled with zeros and the pose branch simply
    sees a constant. Nothing downstream changes shape."""
    frames = np.asarray(track.frames, dtype=np.int64)
    n = len(frames)
    if n < window + 1:
        return []

    w = float(video.width or 1920)
    h = float(video.height or 1080)

    # Frame -> position in the track, so the look-ahead lookup is O(1). Tracks
    # can have gaps where the pedestrian was tagged "outside".
    pos = {int(f): i for i, f in enumerate(frames)}

    crossing_point = track.crossing_point
    intention_prob = float(track.attributes.get("intention_prob", -1.0))

    out: list[Window] = []
    for end in range(window - 1, n, stride):
        t_end = int(frames[end])
        if crossing_point is not None and t_end > crossing_point:
            break  # past the event: no longer a forecast

        target_frame = t_end + lead
        target_idx = pos.get(target_frame)
        if target_idx is None:
            continue
        label = _crossing_at(track, target_idx)
        if label is None:
            continue

        idx = np.arange(end - window + 1, end + 1)
        # A window that is not contiguous in frame numbers spans an occlusion
        # gap; the velocity features would be lies, so drop it.
        if int(frames[idx[-1]] - frames[idx[0]]) != window - 1:
            continue

        future, mask = _future_deltas(track, end, horizon, w, h)
        out.append(
            Window(
                video_id=video.video_id,
                ped_id=track.ped_id,
                dataset=video.dataset,
                frames=frames[idx],
                pose=(
                    pose[idx]
                    if pose is not None
                    else np.zeros((window, POSE_DIM), dtype=np.float32)
                ),
                context=context[idx],
                gesture=gesture[idx],
                motion=_normalised_motion(track, idx, w, h),
                future=future,
                future_mask=mask,
                intent=int(label),
                lead=lead,
                intention_prob=intention_prob,
            )
        )
    return out


def video_windows(
    video: VideoAnnotation,
    window: int = DEFAULT_WINDOW,
    lead: int = DEFAULT_LEAD,
    stride: int = DEFAULT_STRIDE,
    horizon: int = DEFAULT_HORIZON,
    pose_by_ped: dict[str, np.ndarray] | None = None,
) -> list[Window]:
    encoder = ContextEncoder(video)
    out: list[Window] = []
    for track in video.behavior_tracks():
        context = encoder.encode_track(track)
        gesture = apply_phone_override(track, encode_gesture_track(track))
        pose = (pose_by_ped or {}).get(track.ped_id)
        out.extend(
            track_windows(
                video,
                track,
                context,
                gesture,
                window=window,
                lead=lead,
                stride=stride,
                horizon=horizon,
                pose=pose,
            )
        )
    return out


def summarise(windows: list[Window]) -> dict[str, float]:
    if not windows:
        return {"n": 0, "positive_rate": 0.0, "n_tracks": 0, "lead_seconds": 0.0}
    labels = np.array([w.intent for w in windows], dtype=np.float32)
    supervised = np.array(
        [(w.gesture != IGNORE_INDEX).any(axis=0) for w in windows], dtype=np.float32
    )
    stats = {
        "n": float(len(windows)),
        "positive_rate": float(labels.mean()),
        "n_tracks": float(len({(w.video_id, w.ped_id) for w in windows})),
        "lead_seconds": float(windows[0].lead / FPS),
    }
    for gi, group in enumerate(GESTURE_GROUPS):
        stats[f"labelled_{group}"] = float(supervised[:, gi].mean())
    return stats


FEATURE_DIMS = {
    "pose": POSE_DIM,
    "context": CONTEXT_DIM,
}
