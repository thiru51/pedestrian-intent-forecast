"""Per-frame body pose features.

The rest of the pipeline never sees a pose estimator. It sees a fixed-width
vector produced by :func:`encode_pose`, built from 17 COCO-style joints in
bounding-box-normalised coordinates plus a handful of derived angles.

Two things get their own joint layout otherwise:

  MediaPipe Pose (BlazePose, 33 landmarks) is what this repo runs by default.
  It is CPU-real-time on one core, which makes it the right thing for getting
  the pipeline honest end to end.

  Two generations of the MediaPipe API are in the wild and PoseExtractor below
  handles both. Up to 0.10.x the model shipped inside the wheel and
  `mp.solutions.pose` loaded it. From 1.0 that module is gone, replaced by
  `mp.tasks.vision.PoseLandmarker`, and the weights are no longer bundled: a
  .task file has to be on disk. See PoseExtractor for where it is looked for.

  HRNet-W32 (Sun et al., CVPR 2019) is what I would swap in for serving. It is
  top-down (needs the detector box we already have from tracking), outputs the
  17 COCO joints directly, and is markedly more accurate on the small, partly
  occluded pedestrians that matter here — but it is a ~28M parameter network
  per person per frame, so it wants the GPU. Because this module normalises
  everything into the same 17-joint COCO order, swapping estimator means
  replacing PoseExtractor and nothing else: encode_pose, the dataset, and the
  model are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# COCO-17 order. This is the interchange format; both MediaPipe and HRNet get
# mapped into it.
COCO_JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
NUM_JOINTS = len(COCO_JOINTS)
J = {name: i for i, name in enumerate(COCO_JOINTS)}

# BlazePose landmark index for each COCO joint. MediaPipe's 33 landmarks are a
# superset of COCO's 17, so this is a pure lookup with no interpolation.
MEDIAPIPE_TO_COCO: tuple[int, ...] = (
    0,   # nose
    2,   # left_eye (centre of the three left-eye landmarks)
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
)

DERIVED_FEATURES: tuple[str, ...] = (
    "torso_lean",       # angle of the hip->shoulder axis away from vertical
    "shoulder_yaw",     # foreshortening of the shoulder line = body turned away
    "hip_yaw",
    "stride_width",     # ankle separation over torso length
    "knee_lift",        # higher when mid-step, flat when standing
    "head_offset",      # nose displaced sideways from torso centre = looking round
    "arm_extension",    # wrists away from the body: raised-hand gestures
    "visible_frac",     # how much of the skeleton the estimator was sure about
)

# 17 joints x (x, y, confidence) plus the derived scalars.
POSE_DIM = NUM_JOINTS * 3 + len(DERIVED_FEATURES)


@dataclass
class PoseFrame:
    """One person, one frame, in COCO-17 order.

    xy is already normalised into the pedestrian's bounding box: (0,0) is the
    box's top-left, (1,1) its bottom-right. Doing it here rather than in the
    model means a pedestrian 20 px tall at the end of the street and one 400 px
    tall at the kerb produce comparable vectors.
    """

    xy: np.ndarray          # (17, 2) float32
    confidence: np.ndarray  # (17,) float32

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float32).reshape(NUM_JOINTS, 2)
        self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(NUM_JOINTS)


def _angle_from_vertical(dx: float, dy: float) -> float:
    # Image y grows downward, so "up" is -y. Returns radians / pi, in [-1, 1].
    return math.atan2(dx, -dy) / math.pi


def derived(frame: PoseFrame) -> np.ndarray:
    xy, c = frame.xy, frame.confidence
    ls, rs = xy[J["left_shoulder"]], xy[J["right_shoulder"]]
    lh, rh = xy[J["left_hip"]], xy[J["right_hip"]]
    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0

    torso = shoulder_mid - hip_mid
    torso_len = float(np.linalg.norm(torso)) + 1e-6

    lean = _angle_from_vertical(float(torso[0]), float(torso[1]))
    # A body facing the camera shows its full shoulder width; turned side-on it
    # collapses towards zero. That is the cheapest orientation cue there is.
    shoulder_yaw = float(abs(ls[0] - rs[0])) / torso_len
    hip_yaw = float(abs(lh[0] - rh[0])) / torso_len

    la, ra = xy[J["left_ankle"]], xy[J["right_ankle"]]
    stride_width = float(abs(la[0] - ra[0])) / torso_len

    lk, rk = xy[J["left_knee"]], xy[J["right_knee"]]
    ankle_y = (la[1] + ra[1]) / 2.0
    knee_lift = float(ankle_y - max(lk[1], rk[1])) / torso_len

    head_offset = float(xy[J["nose"]][0] - shoulder_mid[0]) / torso_len

    lw, rw = xy[J["left_wrist"]], xy[J["right_wrist"]]
    arm_extension = (
        float(abs(lw[0] - ls[0]) + abs(rw[0] - rs[0])) / (2.0 * torso_len)
    )

    return np.array(
        [
            lean,
            shoulder_yaw,
            hip_yaw,
            stride_width,
            knee_lift,
            head_offset,
            arm_extension,
            float(c.mean()),
        ],
        dtype=np.float32,
    )


def encode_pose(frame: PoseFrame) -> np.ndarray:
    """(POSE_DIM,) float32. Joint coordinates are recentred on the hip so that
    where the person is in the box does not leak into the pose branch — that is
    the context branch's job."""
    hip_mid = (frame.xy[J["left_hip"]] + frame.xy[J["right_hip"]]) / 2.0
    centred = frame.xy - hip_mid
    return np.concatenate(
        [centred.reshape(-1), frame.confidence, derived(frame)]
    ).astype(np.float32)


def encode_sequence(frames: list[PoseFrame]) -> np.ndarray:
    """(T, POSE_DIM) float32."""
    if not frames:
        return np.zeros((0, POSE_DIM), dtype=np.float32)
    return np.stack([encode_pose(f) for f in frames]).astype(np.float32)


class PoseExtractor:
    """MediaPipe Pose behind the PoseFrame interface.

    Constructed lazily: importing mediapipe pulls in a large native library and
    nothing in the synthetic-data path or the tests needs it, so a machine
    without it working can still run everything except real video.

    Supports both MediaPipe generations, because which one is installed decides
    whether a weights file is needed:

      mediapipe <= 0.10.x   `mp.solutions.pose`, weights bundled in the wheel
      mediapipe >= 1.0      `mp.tasks.vision.PoseLandmarker`, weights are a
                            separate .task file that has to be on disk

    For the second, the file is looked for at `model_path`, then at
    $PIF_POSE_MODEL, then at data/pose_landmarker.task. The landmark layout is
    the same 33-point BlazePose skeleton either way, so MEDIAPIPE_TO_COCO maps
    both and nothing downstream can tell which ran.
    """

    DEFAULT_MODEL = "data/pose_landmarker.task"

    def __init__(
        self,
        model_complexity: int = 1,
        min_confidence: float = 0.4,
        model_path: str | None = None,
    ):
        self.model_complexity = model_complexity
        self.min_confidence = min_confidence
        self.model_path = model_path
        self._pose = None
        self._api = ""

    def _model_file(self) -> str:
        import os

        for candidate in (
            self.model_path,
            os.environ.get("PIF_POSE_MODEL"),
            self.DEFAULT_MODEL,
        ):
            if candidate and os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "MediaPipe 1.x needs a pose model file and none was found. Looked "
            f"at $PIF_POSE_MODEL and {self.DEFAULT_MODEL}. Fetch one with "
            "scripts/fetch_pose_model.sh, or point --pose-model at an existing "
            "pose_landmarker .task bundle."
        )

    def _ensure(self):
        if self._pose is not None:
            return self._pose
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "mediapipe is needed to run pose extraction on real video. "
                "pip install mediapipe, or use the synthetic path which "
                "does not need it."
            ) from exc

        if hasattr(mp, "solutions"):
            self._api = "solutions"
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
            )
            return self._pose

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._api = "tasks"
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=self._model_file()
                ),
                # IMAGE rather than VIDEO: the crops handed in are separate
                # pedestrians from separate boxes, not one continuous person,
                # so the tracker between frames would be tracking the wrong
                # thing.
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self.min_confidence,
                min_pose_presence_confidence=self.min_confidence,
            )
        )
        return self._pose

    def __call__(self, crop: np.ndarray) -> PoseFrame | None:
        """crop is the RGB pedestrian patch, already cut out of the frame with
        the annotation box. Returns None when nothing was detected."""
        pose = self._ensure()
        if self._api == "solutions":
            result = pose.process(crop)
            landmarks = getattr(result, "pose_landmarks", None)
            if landmarks is None:
                return None
            pts = landmarks.landmark
        else:
            import mediapipe as mp

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)
            )
            result = pose.detect(image)
            if not result.pose_landmarks:
                return None
            pts = result.pose_landmarks[0]

        xy = np.array(
            [[pts[i].x, pts[i].y] for i in MEDIAPIPE_TO_COCO], dtype=np.float32
        )
        conf = np.array(
            [pts[i].visibility for i in MEDIAPIPE_TO_COCO], dtype=np.float32
        )
        return PoseFrame(xy=xy, confidence=conf)

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None


def empty_pose() -> PoseFrame:
    """What to feed the model for a frame where the estimator found nothing.
    Zero confidence everywhere is a real signal, not padding, so the model can
    learn to distrust the joint values on that frame."""
    return PoseFrame(
        xy=np.zeros((NUM_JOINTS, 2), dtype=np.float32),
        confidence=np.zeros(NUM_JOINTS, dtype=np.float32),
    )
