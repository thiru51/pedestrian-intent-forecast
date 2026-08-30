"""Pose and context features.

Both are hand-rolled encodings, so the thing to check is that they actually
encode what their names claim - a pose feature called "torso_lean" that does
not change when the torso leans is worse than no feature at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from pif.data.schema import PedTrack, VideoAnnotation
from pif.features.context import CONTEXT_DIM, CTX, ContextEncoder
from pif.features.pose import (
    COCO_JOINTS,
    DERIVED_FEATURES,
    J,
    MEDIAPIPE_TO_COCO,
    NUM_JOINTS,
    POSE_DIM,
    PoseFrame,
    derived,
    empty_pose,
    encode_pose,
    encode_sequence,
)


def standing() -> PoseFrame:
    xy = np.array(
        [
            [0.50, 0.06],
            [0.44, 0.04], [0.56, 0.04],
            [0.39, 0.05], [0.61, 0.05],
            [0.30, 0.18], [0.70, 0.18],
            [0.24, 0.34], [0.76, 0.34],
            [0.22, 0.49], [0.78, 0.49],
            [0.38, 0.52], [0.62, 0.52],
            [0.37, 0.74], [0.63, 0.74],
            [0.36, 0.97], [0.64, 0.97],
        ],
        dtype=np.float32,
    )
    return PoseFrame(xy=xy, confidence=np.full(NUM_JOINTS, 0.9, dtype=np.float32))


def test_dims_line_up():
    assert len(COCO_JOINTS) == NUM_JOINTS == 17
    assert len(MEDIAPIPE_TO_COCO) == NUM_JOINTS
    assert POSE_DIM == NUM_JOINTS * 3 + len(DERIVED_FEATURES)
    assert encode_pose(standing()).shape == (POSE_DIM,)
    assert encode_sequence([standing()] * 4).shape == (4, POSE_DIM)
    assert encode_sequence([]).shape == (0, POSE_DIM)


def test_encoding_is_hip_centred():
    """Two identical skeletons at different heights in the box must encode to
    the same joint block - where the person sits in the box is context's job."""
    a = standing()
    b = PoseFrame(xy=a.xy + np.array([0.1, -0.2], np.float32), confidence=a.confidence)
    assert np.allclose(encode_pose(a)[: NUM_JOINTS * 2], encode_pose(b)[: NUM_JOINTS * 2],
                       atol=1e-6)


def test_torso_lean_tracks_a_lean():
    upright = standing()
    leaning = standing()
    for j in ("nose", "left_shoulder", "right_shoulder", "left_eye", "right_eye"):
        leaning.xy[J[j], 0] += 0.12
    assert derived(leaning)[0] > derived(upright)[0]


def test_shoulder_yaw_collapses_when_turned_side_on():
    front = standing()
    side = standing()
    mid = float(side.xy[:, 0].mean())
    side.xy[:, 0] = mid + (side.xy[:, 0] - mid) * 0.15  # foreshortened
    assert derived(side)[1] < derived(front)[1]


def test_arm_extension_rises_with_a_raised_hand():
    down = standing()
    raised = standing()
    raised.xy[J["right_wrist"]] = np.array([0.95, 0.20], np.float32)
    assert derived(raised)[6] > derived(down)[6]


def test_empty_pose_is_zero_confidence_not_padding():
    e = empty_pose()
    assert e.confidence.sum() == 0.0
    v = encode_pose(e)
    assert v.shape == (POSE_DIM,)
    assert np.isfinite(v).all()
    assert v[NUM_JOINTS * 2 : NUM_JOINTS * 3].sum() == 0.0


def _video(width=1920, height=1080):
    return VideoAnnotation(
        video_id="v", dataset="test", width=width, height=height, num_frames=60
    )


def _track_at(xs, y2=900.0, box_w=60.0, box_h=180.0):
    t = PedTrack(ped_id="0_1_1b")
    for i, x in enumerate(xs):
        t.frames.append(i)
        t.bbox.append((x, y2 - box_h, x + box_w, y2))
        t.occlusion.append(0)
        t.action.append("walking")
        t.look.append("not-looking")
        t.gesture.append("__undefined__")
        t.cross.append("not-crossing")
    t.attributes["num_lanes"] = 2
    t.attributes["signalized"] = "n/a"
    return t


def test_context_shape_and_finiteness():
    video = _video()
    track = _track_at([300.0 + 10 * i for i in range(20)])
    video.pedestrians[track.ped_id] = track
    video.ego = {i: {"action": "moving_slow"} for i in range(20)}
    ctx = ContextEncoder(video).encode_track(track)
    assert ctx.shape == (20, CONTEXT_DIM)
    assert np.isfinite(ctx).all()


def test_curb_distance_falls_as_the_pedestrian_closes_on_the_ego_corridor():
    """The feature that is supposed to separate "about to step off" from
    "standing about"."""
    video = _video()
    # Walk from the left edge of frame towards the centre of the lane.
    track = _track_at([100.0 + 40.0 * i for i in range(20)])
    video.pedestrians[track.ped_id] = track
    video.ego = {i: {"action": "moving_slow"} for i in range(20)}
    ctx = ContextEncoder(video).encode_track(track)
    curb = ctx[:, CTX["curb_dist"]]
    assert curb[0] > curb[-1]
    assert ctx[-1, CTX["corridor_inside"]] == 1.0
    assert ctx[0, CTX["corridor_inside"]] == 0.0


def test_ego_speed_uses_the_jaad_action_proxy_and_pie_obd():
    video = _video()
    track = _track_at([500.0] * 4)
    video.pedestrians[track.ped_id] = track
    video.ego = {i: {"action": "moving_fast"} for i in range(4)}
    jaad = ContextEncoder(video).encode_track(track)
    assert jaad[0, CTX["ego_speed"]] == pytest.approx(11.0 / 20.0)

    video.ego = {i: {"OBD_speed": 36.0} for i in range(4)}  # km/h -> 10 m/s
    pie = ContextEncoder(video).encode_track(track)
    assert pie[0, CTX["ego_speed"]] == pytest.approx(10.0 / 20.0)


def test_scene_flags_and_traffic_light_one_hot():
    video = _video()
    track = _track_at([500.0] * 3)
    video.pedestrians[track.ped_id] = track
    video.ego = {i: {"action": "stopped"} for i in range(3)}
    video.scene = {
        i: {"ped_crossing": 1, "ped_sign": 1, "stop_sign": 0, "traffic_light": "red"}
        for i in range(3)
    }
    ctx = ContextEncoder(video).encode_track(track)
    assert ctx[0, CTX["crosswalk_scene"]] == 1.0
    assert ctx[0, CTX["ped_sign"]] == 1.0
    assert ctx[0, CTX["stop_sign"]] == 0.0
    lights = [ctx[0, CTX[k]] for k in ("tl_none", "tl_red", "tl_yellow", "tl_green")]
    assert lights == [0.0, 1.0, 0.0, 0.0]


def test_approaching_pedestrian_has_positive_scale_rate():
    video = _video()
    track = PedTrack(ped_id="0_1_1b")
    for i in range(10):
        h = 100.0 + 15.0 * i  # box grows: the ego vehicle is closing
        track.frames.append(i)
        track.bbox.append((800.0, 900.0 - h, 800.0 + h * 0.4, 900.0))
        track.occlusion.append(0)
        track.action.append("standing")
        track.look.append("looking")
        track.gesture.append("__undefined__")
        track.cross.append("not-crossing")
    video.pedestrians[track.ped_id] = track
    video.ego = {i: {"action": "moving_fast"} for i in range(10)}
    ctx = ContextEncoder(video).encode_track(track)
    assert (ctx[1:, CTX["scale_rate"]] > 0).all()
    assert ctx[-1, CTX["inv_ttc"]] > ctx[1, CTX["inv_ttc"]]
