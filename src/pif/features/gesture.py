"""Turn JAAD/PIE per-frame behaviour strings into taxonomy label vectors.

The taxonomy itself (four parallel TITAN-style groups) lives in pif.taxonomy.
This module is only the annotation-side adapter: it reads the raw strings a
track carries and produces integer class ids per group per frame, which are
what the gesture head is supervised against.

Note what is deliberately not here: nothing reads pixels. The gesture head in
the model predicts these labels from pose plus context at inference time, so
the annotation strings are training signal only.
"""

from __future__ import annotations

import numpy as np

from ..data.schema import PedTrack
from ..taxonomy import (
    Atomic,
    Communicative,
    Contextual,
    GROUP_SIZES,
    SUPERVISED_GROUPS,
    Transportive,
    map_atomic,
    map_communicative,
    map_contextual,
)

GESTURE_GROUPS: tuple[str, ...] = ("atomic", "communicative", "contextual", "transportive")
GESTURE_LOGIT_DIM = sum(GROUP_SIZES[g] for g in GESTURE_GROUPS)

# Which slice of the concatenated logit vector belongs to each group.
GROUP_SLICES: dict[str, slice] = {}
_off = 0
for _g in GESTURE_GROUPS:
    GROUP_SLICES[_g] = slice(_off, _off + GROUP_SIZES[_g])
    _off += GROUP_SIZES[_g]
del _off, _g

# Groups JAAD/PIE annotate. Anything outside this is emitted as -1 and the loss
# ignores it, rather than being silently trained towards the default class.
IGNORE_INDEX = -1


def _at(values: list[str], i: int) -> str | None:
    return values[i] if i < len(values) else None


def encode_track(track: PedTrack) -> np.ndarray:
    """(len(track.frames), 4) int64 of class ids in GESTURE_GROUPS order.

    Frames whose group is not supervised for this dataset get IGNORE_INDEX.
    """
    n = len(track.frames)
    out = np.full((n, len(GESTURE_GROUPS)), IGNORE_INDEX, dtype=np.int64)
    if n == 0:
        return out

    supervised = set(SUPERVISED_GROUPS)
    has_behavior = bool(track.action or track.cross)

    for i in range(n):
        action = _at(track.action, i)
        gesture = _at(track.gesture, i)
        look = _at(track.look, i)
        cross = _at(track.cross, i)

        if not has_behavior:
            continue  # box-only track: no behaviour columns at all

        if "atomic" in supervised:
            out[i, 0] = map_atomic(action)
        if "communicative" in supervised:
            out[i, 1] = map_communicative(gesture, look)
        if "contextual" in supervised:
            out[i, 2] = map_contextual(cross, action)
        # Transportive is outside SUPERVISED_GROUPS because it cannot be read
        # off the per-frame behaviour columns, but JAAD's appearance file does
        # carry it for the high-visibility videos, so try there and fall back
        # to ignore.
        out[i, 3] = _transportive_from_appearance(track, track.frames[i])

    return out


def _transportive_from_appearance(track: PedTrack, frame: int) -> int:
    """JAAD's appearance file is the only place either dataset says anything
    about carried objects, and it only covers the high-visibility videos."""
    app = track.appearance.get(frame)
    if not app:
        return IGNORE_INDEX
    if app.get("stroller_cart"):
        return int(Transportive.PUSHING)
    if any(
        app.get(k)
        for k in (
            "bag_hand",
            "bag_elbow",
            "bag_shoulder",
            "bag_left_side",
            "bag_right_side",
            "backpack",
            "object",
            "umbrella",
        )
    ):
        return int(Transportive.CARRYING)
    return int(Transportive.NONE)


def phone_distracted(track: PedTrack, frame: int) -> bool:
    """PHONE_DISTRACTED cannot be read off the behaviour columns; the only
    source is the appearance file's phone flag."""
    return bool(track.appearance.get(frame, {}).get("phone", 0))


def apply_phone_override(track: PedTrack, labels: np.ndarray) -> np.ndarray:
    """Phone use beats gaze but not an explicit hand signal, matching the
    precedence already used in taxonomy.map_communicative."""
    for i, frame in enumerate(track.frames):
        if i >= len(labels):
            break
        if labels[i, 1] in (int(Communicative.NONE), int(Communicative.LOOKING_AT_TRAFFIC)):
            if phone_distracted(track, frame):
                labels[i, 1] = int(Communicative.PHONE_DISTRACTED)
    return labels


def one_hot(labels: np.ndarray) -> np.ndarray:
    """(T, 4) class ids -> (T, GESTURE_LOGIT_DIM) float32.

    Used to feed ground-truth gesture labels into the temporal model during
    teacher-forced debugging and by the synthetic generator; at inference the
    model uses its own gesture head's logits in the same layout.
    """
    t = labels.shape[0]
    out = np.zeros((t, GESTURE_LOGIT_DIM), dtype=np.float32)
    for gi, group in enumerate(GESTURE_GROUPS):
        sl = GROUP_SLICES[group]
        for i in range(t):
            c = int(labels[i, gi])
            if c >= 0:
                out[i, sl.start + c] = 1.0
    return out


CLASS_NAMES: dict[str, tuple[str, ...]] = {
    "atomic": tuple(a.name.lower() for a in Atomic),
    "communicative": tuple(a.name.lower() for a in Communicative),
    "contextual": tuple(a.name.lower() for a in Contextual),
    "transportive": tuple(a.name.lower() for a in Transportive),
}
