"""SYNTHETIC pedestrian sequences. Not data. Plumbing only.

Read this before you read anything that reports a number from it.

JAAD and PIE video clips are a 3.1 GB and a 74 GB download respectively, behind
a request form, and this repo does not have them. Without frames there is no
pose estimation, and without pose estimation the pose branch of the model has
nothing to eat. So that the whole path — window -> features -> model -> loss ->
metric -> ONNX — can be run, tested and profiled today, this module fabricates
sequences with the right shapes, the right value ranges, and a plausible causal
story.

What the generator does, stated plainly so nobody mistakes its output for a
result:

  A latent state per track (crossing / waiting-but-not-crossing / passing by)
  drives BOTH the pose stream and the context stream. Pose gets weak, noisy
  cues (a bit of torso rotation towards the road, a head turn, a shortened
  stride before stepping off). Context gets stronger ones (kerb distance
  closing, ego vehicle approaching, crosswalk present). That asymmetry is an
  assumption I wrote into the generator because it is the hypothesis the
  project is built on — it is NOT evidence for the hypothesis. A fused model
  beating a pose-only model on this data proves that the fusion path is wired
  up and can carry information, and nothing whatsoever about real pedestrians.

Everything downstream treats these Windows identically to parsed ones, which is
the point: the same code runs on both.
"""

from __future__ import annotations

import numpy as np

from .schema import PedTrack, VideoAnnotation
from .windows import Window, DEFAULT_HORIZON, DEFAULT_LEAD, DEFAULT_WINDOW
from ..features.context import CONTEXT_DIM, CTX
from ..features.gesture import GESTURE_GROUPS, encode_track as encode_gesture_track
from ..features.pose import J, NUM_JOINTS, PoseFrame, encode_sequence
from ..taxonomy import Atomic, Communicative, Contextual, Transportive

SYNTHETIC_DATASET = "synthetic"

# Latent behaviours. CROSSING and WAITING_KERB both stand still at the kerb for
# a while; separating them is exactly the thing pose alone is bad at.
CROSSING = 0
WAITING_KERB = 1   # at the kerb, but waiting for a bus / a friend / the light
PASSING_BY = 2     # walking along the pavement, never intends to cross

STATE_NAMES = ("crossing", "waiting_kerb", "passing_by")

# Canonical standing skeleton in box coordinates: x is signed from the body
# midline in units of half a shoulder width, y runs 0 (head) to 1 (feet).
_CANONICAL = np.array(
    [
        [0.00, 0.06],  # nose
        [-0.06, 0.04], [0.06, 0.04],      # eyes
        [-0.11, 0.05], [0.11, 0.05],      # ears
        [-0.20, 0.18], [0.20, 0.18],      # shoulders
        [-0.26, 0.34], [0.26, 0.34],      # elbows
        [-0.28, 0.49], [0.28, 0.49],      # wrists
        [-0.12, 0.52], [0.12, 0.52],      # hips
        [-0.13, 0.74], [0.13, 0.74],      # knees
        [-0.14, 0.97], [0.14, 0.97],      # ankles
    ],
    dtype=np.float32,
)
assert _CANONICAL.shape == (NUM_JOINTS, 2)


def _skeleton(
    rng: np.random.Generator,
    phase: float,
    gait: float,
    body_yaw: float,
    head_yaw: float,
    lean: float,
    arm_raise: float,
) -> PoseFrame:
    xy = _CANONICAL.copy()

    # Body yaw squashes horizontal extent: side-on, the shoulder line projects
    # to almost nothing. This is the same cue encode_pose reads back out.
    xy[:, 0] *= float(np.cos(body_yaw))

    # Gait: legs and arms swing in antiphase, amplitude scaled by how much the
    # person is actually walking.
    swing = float(np.sin(phase)) * gait
    xy[J["left_knee"], 0] += 0.10 * swing
    xy[J["right_knee"], 0] -= 0.10 * swing
    xy[J["left_ankle"], 0] += 0.18 * swing
    xy[J["right_ankle"], 0] -= 0.18 * swing
    xy[J["left_ankle"], 1] -= 0.05 * abs(swing)
    xy[J["right_ankle"], 1] -= 0.05 * max(-swing, 0.0)
    xy[J["left_wrist"], 0] -= 0.12 * swing
    xy[J["right_wrist"], 0] += 0.12 * swing

    # Forward lean tips the upper body about the hips.
    upper = [
        J["nose"], J["left_eye"], J["right_eye"], J["left_ear"], J["right_ear"],
        J["left_shoulder"], J["right_shoulder"], J["left_elbow"], J["right_elbow"],
        J["left_wrist"], J["right_wrist"],
    ]
    xy[upper, 0] += lean * (0.52 - xy[upper, 1])

    # Head turned to check for traffic.
    for j in (J["nose"], J["left_eye"], J["right_eye"], J["left_ear"], J["right_ear"]):
        xy[j, 0] += head_yaw * 0.10

    if arm_raise > 0:
        xy[J["right_wrist"]] += np.array([0.18 * arm_raise, -0.34 * arm_raise], np.float32)
        xy[J["right_elbow"]] += np.array([0.10 * arm_raise, -0.16 * arm_raise], np.float32)

    xy[:, 0] += 0.5  # move to box coordinates: midline at 0.5
    xy += rng.normal(0.0, 0.006, size=xy.shape).astype(np.float32)

    conf = np.clip(rng.normal(0.85, 0.08, size=NUM_JOINTS), 0.0, 1.0).astype(np.float32)
    # Legs are the first thing to go when a pedestrian is partly behind a
    # parked car, which is the common occlusion in these datasets.
    conf[[J["left_ankle"], J["right_ankle"], J["left_knee"], J["right_knee"]]] *= float(
        rng.uniform(0.55, 1.0)
    )
    return PoseFrame(xy=xy, confidence=conf)


def _track(rng: np.random.Generator, state: int, length: int) -> dict:
    """One synthetic track: pose frames, context rows, gesture ids, boxes."""
    # Frame index at which the crossing actually starts, for CROSSING tracks.
    step_off = length - int(rng.integers(8, 20)) if state == CROSSING else length + 999

    # Where the person is, laterally, as a fraction of image width, and how
    # fast they close on the kerb.
    lateral0 = float(rng.uniform(0.12, 0.40))
    approach = {
        CROSSING: float(rng.uniform(0.0015, 0.0035)),
        WAITING_KERB: float(rng.uniform(-0.0004, 0.0006)),
        PASSING_BY: 0.0,
    }[state]
    if rng.random() < 0.5:
        side = -1.0
    else:
        side = 1.0

    ego_speed0 = float(rng.uniform(2.0, 14.0))
    ego_decel = float(rng.uniform(0.0, 0.06)) if state == CROSSING else float(
        rng.uniform(-0.02, 0.02)
    )
    box_h0 = float(rng.uniform(0.10, 0.45))
    crosswalk = 1.0 if (state == CROSSING and rng.random() < 0.6) else (
        1.0 if rng.random() < 0.15 else 0.0
    )
    signalized = 1.0 if (crosswalk and rng.random() < 0.5) else 0.0
    light = int(rng.integers(0, 4))

    poses: list[PoseFrame] = []
    context = np.zeros((length, CONTEXT_DIM), dtype=np.float32)
    gesture = np.full((length, len(GESTURE_GROUPS)), -1, dtype=np.int64)
    boxes = np.zeros((length, 4), dtype=np.float32)

    phase = float(rng.uniform(0, 2 * np.pi))
    prev_speed = None
    prev_log_h = None
    prev_cx = None

    for t in range(length):
        crossing_now = t >= step_off
        # How close to stepping off, in [0, 1]. Drives the weak pose cues.
        imminence = float(np.clip(1.0 - (step_off - t) / 45.0, 0.0, 1.0)) if state == CROSSING else 0.0

        if state == PASSING_BY:
            gait = float(rng.uniform(0.85, 1.0))
        elif crossing_now:
            gait = 1.0
        elif state == CROSSING:
            gait = float(np.clip(0.8 - 0.7 * imminence, 0.05, 1.0))
        else:
            gait = float(rng.uniform(0.0, 0.12))
        phase += 0.45 * gait + 0.05

        # Pose cues. Deliberately small relative to the 0.006 joint noise plus
        # the natural variation, because a real pose estimator on a 60-pixel
        # pedestrian is not going to resolve a 5-degree torso rotation cleanly.
        body_yaw = float(rng.uniform(0.9, 1.3)) if state == PASSING_BY else float(
            np.clip(rng.normal(0.55 - 0.30 * imminence, 0.22), 0.0, 1.4)
        )
        looks = (state == CROSSING and rng.random() < 0.25 + 0.5 * imminence) or (
            state == WAITING_KERB and rng.random() < 0.12
        )
        head_yaw = side * float(rng.uniform(0.5, 1.0)) if looks else float(
            rng.normal(0.0, 0.18)
        )
        lean = float(np.clip(rng.normal(0.02 + 0.06 * imminence, 0.05), -0.2, 0.35))
        waves = state == CROSSING and imminence > 0.6 and rng.random() < 0.08
        arm_raise = float(rng.uniform(0.5, 1.0)) if waves else 0.0

        poses.append(_skeleton(rng, phase, gait, body_yaw, head_yaw, lean, arm_raise))

        # Context stream.
        lateral = lateral0 - approach * t
        if crossing_now:
            lateral -= 0.010 * (t - step_off)
        lateral = float(np.clip(lateral, -0.05, 0.5))
        cx = 0.5 + side * lateral
        box_h = box_h0 * (1.0 + 0.0016 * t * max(ego_speed0, 1.0) / 8.0)
        box_h = float(np.clip(box_h, 0.03, 0.95))
        cy = float(np.clip(0.62 + 0.15 * box_h, 0.0, 1.0))
        speed = float(np.clip(ego_speed0 - ego_decel * t, 0.0, 22.0))

        half = 0.28  # matches ContextEncoder's corridor at the bottom of frame
        curb = float(np.clip((lateral - half) * 3.2, -4.0, 8.0) / 8.0)
        accel = 0.0 if prev_speed is None else speed - prev_speed
        prev_speed = speed
        log_h = float(np.log(max(box_h, 1e-4)))
        scale_rate = 0.0 if prev_log_h is None else (log_h - prev_log_h) * 30.0
        prev_log_h = log_h
        vx = 0.0 if prev_cx is None else float(np.clip((cx - prev_cx) * 20.0, -2.0, 2.0))
        prev_cx = cx

        row = context[t]
        row[CTX["cx"]] = cx
        row[CTX["cy"]] = cy
        row[CTX["box_h"]] = box_h
        row[CTX["box_ar"]] = float(rng.normal(0.42, 0.05))
        row[CTX["vx"]] = vx
        row[CTX["vy"]] = float(rng.normal(0.0, 0.02))
        row[CTX["scale_rate"]] = float(np.clip(scale_rate, -5.0, 5.0))
        row[CTX["curb_dist"]] = curb
        row[CTX["corridor_inside"]] = 1.0 if lateral <= half else 0.0
        row[CTX["crosswalk_scene"]] = crosswalk
        row[CTX["crosswalk_near"]] = crosswalk * float(rng.uniform(0.3, 1.0))
        row[CTX["ego_speed"]] = speed / 20.0
        row[CTX["ego_accel"]] = float(np.clip(accel, -5.0, 5.0))
        row[CTX["inv_ttc"]] = float(np.clip(box_h * speed / 20.0, 0.0, 1.0))
        row[CTX["occlusion"]] = 0.5 if rng.random() < 0.08 else 0.0
        row[CTX[["tl_none", "tl_red", "tl_yellow", "tl_green"][light]]] = 1.0
        row[CTX["ped_sign"]] = crosswalk
        row[CTX["stop_sign"]] = 1.0 if rng.random() < 0.1 else 0.0
        row[CTX["signalized"]] = signalized
        row[CTX["num_lanes"]] = float(rng.integers(2, 7)) / 6.0

        # Gesture labels, in the same layout the annotation adapter produces.
        gesture[t, 0] = int(Atomic.WALKING if gait > 0.4 else Atomic.STANDING)
        gesture[t, 1] = int(
            Communicative.WAVING_THROUGH
            if waves
            else (Communicative.LOOKING_AT_TRAFFIC if looks else Communicative.NONE)
        )
        if crossing_now:
            gesture[t, 2] = int(Contextual.CROSSING)
        elif state == CROSSING and imminence > 0.4:
            gesture[t, 2] = int(Contextual.HESITATING)
        elif state == WAITING_KERB:
            gesture[t, 2] = int(Contextual.WAITING_TO_CROSS)
        else:
            gesture[t, 2] = int(Contextual.WALKING_ALONG)
        gesture[t, 3] = int(Transportive.NONE)

        bw = box_h * 0.42
        boxes[t] = (cx - bw / 2, cy - box_h / 2, cx + bw / 2, cy + box_h / 2)

    return {
        "pose": encode_sequence(poses),
        "context": context,
        "gesture": gesture,
        "boxes": boxes,
        "step_off": step_off,
        "state": state,
    }


def make_windows(
    n_tracks: int = 400,
    seed: int = 0,
    window: int = DEFAULT_WINDOW,
    lead: int = DEFAULT_LEAD,
    horizon: int = DEFAULT_HORIZON,
    stride: int = 3,
    track_length: int = 140,
    positive_fraction: float = 0.4,
) -> list[Window]:
    """Synthetic Windows, ready for PedestrianWindows.

    The label follows the same rule as the real pipeline: is the pedestrian
    crossing `lead` frames after the end of the observation window.
    """
    rng = np.random.default_rng(seed)
    out: list[Window] = []
    for k in range(n_tracks):
        r = rng.random()
        if r < positive_fraction:
            state = CROSSING
        elif r < positive_fraction + (1 - positive_fraction) * 0.55:
            state = WAITING_KERB
        else:
            state = PASSING_BY
        length = track_length + int(rng.integers(-20, 21))
        tr = _track(rng, state, length)
        step_off = tr["step_off"]

        for end in range(window - 1, length - lead, stride):
            if step_off <= length and end > step_off:
                break  # past the event, same truncation rule as the real loader
            target = end + lead
            if target >= length:
                break
            label = 1 if target >= step_off else 0
            idx = np.arange(end - window + 1, end + 1)

            centres = tr["boxes"][:, :2] + tr["boxes"][:, 2:] / 2.0
            base = centres[end]
            future = np.zeros((horizon, 2), dtype=np.float32)
            mask = np.zeros(horizon, dtype=np.float32)
            for j in range(horizon):
                if end + j + 1 >= length:
                    break
                future[j] = centres[end + j + 1] - base
                mask[j] = 1.0

            motion = np.stack(
                [
                    centres[idx, 0],
                    centres[idx, 1],
                    tr["boxes"][idx, 2] - tr["boxes"][idx, 0],
                    tr["boxes"][idx, 3] - tr["boxes"][idx, 1],
                ],
                axis=1,
            ).astype(np.float32)

            out.append(
                Window(
                    video_id=f"synthetic_{k:04d}",
                    ped_id=f"{k}_{STATE_NAMES[state]}",
                    dataset=SYNTHETIC_DATASET,
                    frames=idx.astype(np.int64),
                    pose=tr["pose"][idx],
                    context=tr["context"][idx],
                    gesture=tr["gesture"][idx],
                    motion=motion,
                    future=future,
                    future_mask=mask,
                    intent=label,
                    lead=lead,
                    intention_prob=-1.0,
                )
            )
    return out


def split_windows(
    windows: list[Window], val_fraction: float = 0.25
) -> tuple[list[Window], list[Window]]:
    """Split by track, never by window. Windows from one track overlap heavily,
    so splitting at window level leaks the answer across the split and every
    number that comes out is inflated."""
    tracks = sorted({w.video_id for w in windows})
    cut = int(len(tracks) * (1.0 - val_fraction))
    train_ids = set(tracks[:cut])
    train = [w for w in windows if w.video_id in train_ids]
    val = [w for w in windows if w.video_id not in train_ids]
    return train, val


def as_video_annotation(n_tracks: int = 4, seed: int = 0) -> VideoAnnotation:
    """A synthetic VideoAnnotation, so the parser-facing code path (schema ->
    ContextEncoder -> gesture adapter -> windows) can be exercised in tests
    without any real XML."""
    rng = np.random.default_rng(seed)
    video = VideoAnnotation(
        video_id="synthetic_video", dataset=SYNTHETIC_DATASET,
        width=1920, height=1080, num_frames=200,
    )
    for k in range(n_tracks):
        state = int(rng.integers(0, 3))
        tr = _track(rng, state, 200)
        ped = PedTrack(ped_id=f"0_0_{k}b")
        for t in range(200):
            x1, y1, x2, y2 = tr["boxes"][t]
            ped.frames.append(t)
            ped.bbox.append((x1 * 1920, y1 * 1080, x2 * 1920, y2 * 1080))
            ped.occlusion.append(0)
            ped.action.append("walking" if tr["gesture"][t, 0] == Atomic.WALKING else "standing")
            ped.look.append("looking" if tr["gesture"][t, 1] == Communicative.LOOKING_AT_TRAFFIC else "not-looking")
            ped.gesture.append("hand_yield" if tr["gesture"][t, 1] == Communicative.WAVING_THROUGH else "__undefined__")
            ped.cross.append("crossing" if t >= tr["step_off"] else "not-crossing")
        ped.attributes["crossing"] = 1 if tr["step_off"] < 200 else 0
        ped.attributes["crossing_point"] = int(tr["step_off"]) if tr["step_off"] < 200 else -1
        ped.attributes["num_lanes"] = 2
        ped.attributes["signalized"] = "n/a"
        video.pedestrians[ped.ped_id] = ped
        video.ego.update({t: {"action": "moving_slow"} for t in range(200)})
        video.scene.update(
            {t: {"ped_crossing": 0, "ped_sign": 0, "stop_sign": 0, "traffic_light": "n/a"}
             for t in range(200)}
        )
    return video


__all__ = [
    "CROSSING",
    "PASSING_BY",
    "STATE_NAMES",
    "SYNTHETIC_DATASET",
    "WAITING_KERB",
    "as_video_annotation",
    "make_windows",
    "split_windows",
]


# Kept out of the public list on purpose: this exists so encode_gesture_track
# stays imported and its signature is checked by the test that round-trips a
# synthetic VideoAnnotation through the real adapter.
_ = encode_gesture_track
