"""Per-frame feature extraction: pose, scene context, gesture attributes."""

from __future__ import annotations

from .context import CONTEXT_DIM, CONTEXT_FEATURES, ContextEncoder
from .gesture import GESTURE_GROUPS, GESTURE_LOGIT_DIM, encode_track as encode_gesture
from .pose import COCO_JOINTS, POSE_DIM, PoseExtractor, PoseFrame, encode_pose

__all__ = [
    "CONTEXT_DIM",
    "CONTEXT_FEATURES",
    "COCO_JOINTS",
    "ContextEncoder",
    "GESTURE_GROUPS",
    "GESTURE_LOGIT_DIM",
    "POSE_DIM",
    "PoseExtractor",
    "PoseFrame",
    "encode_gesture",
    "encode_pose",
]
