"""Gesture/action label set, modelled on the TITAN hierarchy.

TITAN (Malla, Dariush & Choi, CVPR 2020) splits pedestrian labels into five
parallel groups rather than one flat class list, because a person can be
walking AND on the phone AND carrying a bag at the same time. We keep four of
those groups; TITAN's "complex contextual" group is dropped because nothing in
JAAD or PIE annotates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Atomic(IntEnum):
    STANDING = 0
    WALKING = 1
    RUNNING = 2


class Communicative(IntEnum):
    NONE = 0
    LOOKING_AT_TRAFFIC = 1
    WAVING_THROUGH = 2
    REQUESTING_RIGHT_OF_WAY = 3
    PHONE_DISTRACTED = 4


class Contextual(IntEnum):
    WALKING_ALONG = 0
    WAITING_TO_CROSS = 1
    HESITATING = 2
    CROSSING = 3


class Transportive(IntEnum):
    NONE = 0
    CARRYING = 1
    PUSHING = 2
    PULLING = 3


GROUPS: dict[str, type[IntEnum]] = {
    "atomic": Atomic,
    "communicative": Communicative,
    "contextual": Contextual,
    "transportive": Transportive,
}

GROUP_SIZES: dict[str, int] = {name: len(enum) for name, enum in GROUPS.items()}

# Total width of the concatenated per-group logit vector the temporal model sees.
GESTURE_DIM: int = sum(GROUP_SIZES.values())

# Groups that JAAD/PIE actually annotate. "transportive" is TITAN-only: we keep
# the head so the taxonomy stays faithful, but it must be masked out of the loss
# when training on JAAD/PIE or it will just learn the constant NONE.
SUPERVISED_GROUPS: tuple[str, ...] = ("atomic", "communicative", "contextual")


@dataclass(frozen=True)
class GestureLabel:
    atomic: int = int(Atomic.STANDING)
    communicative: int = int(Communicative.NONE)
    contextual: int = int(Contextual.WALKING_ALONG)
    transportive: int = int(Transportive.NONE)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.atomic, self.communicative, self.contextual, self.transportive)


# JAAD/PIE record hand gestures as free-ish strings; these are the values that
# actually occur in the released XML. Anything unrecognised falls back to NONE
# rather than raising, because the annotation files are not perfectly uniform.
_JAAD_GESTURE_TO_COMMUNICATIVE = {
    "greet": Communicative.REQUESTING_RIGHT_OF_WAY,
    "hand_ack": Communicative.REQUESTING_RIGHT_OF_WAY,
    "ack": Communicative.REQUESTING_RIGHT_OF_WAY,
    "yield": Communicative.WAVING_THROUGH,
    "hand_yield": Communicative.WAVING_THROUGH,
    "rightofway": Communicative.REQUESTING_RIGHT_OF_WAY,
    "hand_rightofway": Communicative.REQUESTING_RIGHT_OF_WAY,
    "slow_down": Communicative.WAVING_THROUGH,
    "speed_up": Communicative.WAVING_THROUGH,
}

_ACTION_TO_ATOMIC = {
    "standing": Atomic.STANDING,
    "stopped": Atomic.STANDING,
    "walking": Atomic.WALKING,
    "moving": Atomic.WALKING,
    "running": Atomic.RUNNING,
    "fast": Atomic.RUNNING,
}


def map_atomic(action: str | None) -> int:
    if not action:
        return int(Atomic.STANDING)
    return int(_ACTION_TO_ATOMIC.get(action.strip().lower(), Atomic.STANDING))


def map_communicative(gesture: str | None, looking: str | None) -> int:
    """Gesture wins over gaze: an explicit hand signal is a stronger cue than
    where the head is pointing, and the two are annotated independently."""
    key = (gesture or "").strip().lower()
    if key in _JAAD_GESTURE_TO_COMMUNICATIVE:
        return int(_JAAD_GESTURE_TO_COMMUNICATIVE[key])
    look = (looking or "").strip().lower()
    if look == "looking":
        return int(Communicative.LOOKING_AT_TRAFFIC)
    return int(Communicative.NONE)


def map_contextual(cross: str | None, action: str | None) -> int:
    """PHONE_DISTRACTED is deliberately not derivable here — neither JAAD nor
    PIE annotates phone use, so it can only come from TITAN or manual labels."""
    c = (cross or "").strip().lower()
    a = (action or "").strip().lower()
    if c == "crossing":
        return int(Contextual.CROSSING)
    if c == "not-crossing" and a in ("standing", "stopped"):
        return int(Contextual.WAITING_TO_CROSS)
    return int(Contextual.WALKING_ALONG)
