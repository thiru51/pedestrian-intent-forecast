"""Windowing: the rules that make this a forecast rather than an observation.

If any of these break, the model is being handed the answer and every metric
in the repo becomes meaningless, so they are worth being fussy about.
"""

from __future__ import annotations

import numpy as np
import pytest

from pif.data.schema import PedTrack, VideoAnnotation
from pif.data.windows import (
    DEFAULT_LEAD,
    summarise,
    track_windows,
    video_windows,
)
from pif.features.context import CONTEXT_DIM, ContextEncoder
from pif.features.gesture import encode_track
from pif.features.pose import POSE_DIM


def make_track(n: int = 120, crossing_point: int | None = 80, gap_at: int | None = None):
    video = VideoAnnotation(
        video_id="v", dataset="test", width=1920, height=1080, num_frames=n
    )
    t = PedTrack(ped_id="0_1_1b")
    for f in range(n):
        if gap_at is not None and gap_at <= f < gap_at + 5:
            continue  # pedestrian tagged "outside": no box for these frames
        t.frames.append(f)
        x = 300.0 + 4.0 * f
        t.bbox.append((x, 500.0, x + 60.0, 680.0))
        t.occlusion.append(0)
        t.action.append("walking")
        t.look.append("looking")
        t.gesture.append("__undefined__")
        t.cross.append(
            "crossing" if (crossing_point is not None and f >= crossing_point)
            else "not-crossing"
        )
    t.attributes["crossing"] = 1 if crossing_point is not None else 0
    t.attributes["crossing_point"] = -1 if crossing_point is None else crossing_point
    t.attributes["num_lanes"] = 2
    video.pedestrians[t.ped_id] = t
    video.ego = {f: {"action": "moving_slow"} for f in range(n)}
    video.scene = {
        f: {"ped_crossing": 0, "ped_sign": 0, "stop_sign": 0, "traffic_light": "n/a"}
        for f in range(n)
    }
    return video, t


def windows_for(video, track, **kw):
    ctx = ContextEncoder(video).encode_track(track)
    ges = encode_track(track)
    return track_windows(video, track, ctx, ges, **kw)


def test_shapes_and_dtypes():
    video, track = make_track()
    ws = windows_for(video, track, window=16, lead=45, stride=4, horizon=15)
    assert ws
    w = ws[0]
    assert w.pose.shape == (16, POSE_DIM)
    assert w.context.shape == (16, CONTEXT_DIM)
    assert w.gesture.shape == (16, 4)
    assert w.motion.shape == (16, 4)
    assert w.future.shape == (15, 2)
    assert w.future_mask.shape == (15,)
    assert w.pose.dtype == np.float32
    assert w.gesture.dtype == np.int64
    # No video clips, so no pose. Zeros, and everything downstream still works.
    assert not w.pose.any()


def test_never_observes_past_the_crossing_point():
    video, track = make_track(n=140, crossing_point=80)
    ws = windows_for(video, track, window=16, lead=45, stride=1)
    assert ws
    assert max(w.t_end for w in ws) <= 80


def test_label_is_the_state_lead_frames_later():
    video, track = make_track(n=140, crossing_point=80)
    ws = windows_for(video, track, window=16, lead=45, stride=1)
    for w in ws:
        expected = 1 if w.t_end + 45 >= 80 else 0
        assert w.intent == expected, (w.t_end, w.intent)
    # And it is a genuine forecast: at least one window ends before the
    # crossing starts but is labelled positive.
    early = [w for w in ws if w.intent == 1 and w.t_end < 80]
    assert early, "no window predicts a crossing that has not happened yet"


def test_longer_lead_gives_fewer_positives_at_the_same_end_frames():
    video, track = make_track(n=200, crossing_point=120)
    short = windows_for(video, track, window=16, lead=15, stride=1)
    long = windows_for(video, track, window=16, lead=60, stride=1)
    # Looking further ahead means more of the observable windows are already
    # inside the positive zone.
    assert np.mean([w.intent for w in long]) > np.mean([w.intent for w in short])


def test_occlusion_gap_windows_are_dropped():
    video, track = make_track(n=140, crossing_point=None, gap_at=60)
    ws = windows_for(video, track, window=16, lead=45, stride=1)
    for w in ws:
        # Contiguity: a 16-frame window must span exactly 15 frame numbers, or
        # the velocity features are computed across a hole.
        assert int(w.frames[-1] - w.frames[0]) == 15


def test_future_mask_marks_the_end_of_the_track():
    video, track = make_track(n=70, crossing_point=None)
    ws = windows_for(video, track, window=16, lead=5, stride=1, horizon=15)
    assert ws
    last = ws[-1]
    assert last.future_mask.sum() < 15  # ran out of track inside the horizon
    assert (last.future[last.future_mask == 0] == 0).all()


def test_short_track_yields_nothing():
    video, track = make_track(n=10, crossing_point=None)
    assert windows_for(video, track, window=16) == []


def test_video_windows_skips_box_only_tracks():
    video, _ = make_track(n=120, crossing_point=80)
    box_only = PedTrack(ped_id="0_1_9")
    box_only.frames = list(range(120))
    box_only.bbox = [(0.0, 0.0, 50.0, 100.0)] * 120
    box_only.occlusion = [0] * 120
    video.pedestrians[box_only.ped_id] = box_only
    ws = video_windows(video, window=16, lead=45, stride=4)
    assert ws
    assert {w.ped_id for w in ws} == {"0_1_1b"}


def test_summarise_reports_lead_in_seconds():
    video, track = make_track()
    ws = windows_for(video, track, lead=DEFAULT_LEAD)
    stats = summarise(ws)
    assert stats["lead_seconds"] == pytest.approx(DEFAULT_LEAD / 30.0)
    assert stats["n"] == len(ws)
    assert 0.0 <= stats["positive_rate"] <= 1.0
