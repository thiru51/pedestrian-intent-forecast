"""The CARLA recording parser, checked against a committed fixture.

Two things these tests are deliberately built around.

First, no simulator. They run in the pixi environment, which has torch and
mediapipe but cannot install the `carla` package at all (see README, "Two
environments"). So the fixture under tests/fixtures/carla is read from disk and
nothing here talks to a CARLA server.

Second, the fixture is hand-built, not captured. Its numbers were chosen to
exercise the parser: a clean crosser, a walker who stops at the kerb and never
crosses, one who steps out from behind a parked car, and one on a side road the
ego never enters. Nothing in it is a measurement and nothing in it should be
quoted as a result.

The point of the whole file is the last test: a CARLA recording has to flow
through the existing windowing code without that code being changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pif.data.carla import (
    CARLARecordings,
    UNLABELLED_GESTURE_GROUPS,
    crossing_summary,
    lead_frames,
    mask_unlabelled_gestures,
    occlusion_code,
)
from pif.data.schema import OCCLUSION
from pif.data.windows import video_windows
from pif.features.context import CONTEXT_DIM
from pif.features.gesture import GESTURE_GROUPS, IGNORE_INDEX, encode_track
from pif.features.pose import POSE_DIM

FIXTURES = Path(__file__).parent / "fixtures" / "carla"

# Frames the fixture's walkers step into the road on, which is what
# crossing_point has to come out as.
W0_CROSSING_FRAME = 48
W2_CROSSING_FRAME = 44


@pytest.fixture
def db():
    return CARLARecordings(FIXTURES)


@pytest.fixture
def video(db):
    return db.load("ep_fixture_0000")


def test_the_parser_does_not_need_the_simulator():
    """pif.data.carla reads files. It must never pull in the carla package,
    which cannot be installed in this environment in the first place."""
    assert "carla" not in sys.modules


def test_lists_episodes(db):
    assert db.video_ids() == ["ep_fixture_0000", "ep_fixture_0001"]


def test_fills_the_shared_schema(video):
    assert video.dataset == "carla"
    assert (video.width, video.height) == (1280, 720)
    assert video.num_frames == 80
    assert len(video.pedestrians) == 4
    assert len(video.ego) == 80
    assert len(video.scene) == 80
    assert "ep_fixture_0000_v0" in video.traffic


def test_every_per_frame_list_is_the_same_length(video):
    for track in video.pedestrians.values():
        n = len(track.frames)
        assert len(track.bbox) == n
        assert len(track.occlusion) == n
        assert len(track.cross) == n
        assert all(len(b) == 4 for b in track.bbox)


def test_boxes_are_inside_the_image(video):
    for track in video.pedestrians.values():
        for x1, y1, x2, y2 in track.bbox:
            assert 0 <= x1 < x2 <= video.width
            assert 0 <= y1 < y2 <= video.height


def test_crossing_point_is_the_frame_they_step_into_the_road(video):
    crosser = video.pedestrians["ep_fixture_0000_w0"]
    assert crosser.attributes["crossing_point"] == W0_CROSSING_FRAME
    # The property returns None rather than -1, which is what windowing reads.
    assert crosser.crossing_point == W0_CROSSING_FRAME
    # And it is a frame number, so it must line up with the cross strings.
    i = crosser.frames.index(W0_CROSSING_FRAME)
    assert crosser.cross[i] == "crossing"
    assert crosser.cross[i - 1] == "not-crossing"


def test_crossing_label_uses_the_minus_one_zero_one_convention(video):
    labels = {
        ped_id: (t.crossing_label, t.attributes["crossing_point"])
        for ped_id, t in video.pedestrians.items()
    }
    # Walked across in front of the ego vehicle.
    assert labels["ep_fixture_0000_w0"] == (1, W0_CROSSING_FRAME)
    # Stopped at the kerb, relevant to the ego but never entered the road.
    assert labels["ep_fixture_0000_w1"] == (0, -1)
    # Stepped out from behind the parked car.
    assert labels["ep_fixture_0000_w2"] == (1, W2_CROSSING_FRAME)
    # On a driving lane, but not one the ego passes through: not relevant.
    assert labels["ep_fixture_0000_w3"] == (-1, -1)
    assert video.pedestrians["ep_fixture_0000_w1"].crossing_point is None
    assert video.pedestrians["ep_fixture_0000_w3"].crossing_point is None


def test_crossing_summary_ignores_frames_before_the_track_starts():
    """crossing_point is a frame number, not a list position. A track that
    starts at frame 100 must report 130, not 30."""
    frames = list(range(100, 140))
    cross = ["not-crossing"] * 30 + ["crossing"] * 10
    assert crossing_summary(frames, cross, ego_relevant=True) == (1, 130)


def test_crossing_summary_for_an_irrelevant_walker():
    frames = list(range(20))
    cross = ["not-crossing"] * 10 + ["irrelevant"] * 10
    assert crossing_summary(frames, cross, ego_relevant=False) == (-1, -1)
    # Same strings, but flagged relevant: still 0, because "irrelevant" is not
    # the positive string. The two flags say different things.
    assert crossing_summary(frames, cross, ego_relevant=True) == (0, -1)


def test_occlusion_uses_the_existing_codes(video):
    occluded = video.pedestrians["ep_fixture_0000_w2"]
    seen = set(occluded.occlusion)
    assert seen == {OCCLUSION["none"], OCCLUSION["part"], OCCLUSION["full"]}
    assert occluded.occlusion[0] == OCCLUSION["full"]
    assert occluded.occlusion[20] == OCCLUSION["part"]
    assert occluded.occlusion[-1] == OCCLUSION["none"]


def test_occlusion_code_thresholds():
    assert occlusion_code(1.0) == OCCLUSION["none"]
    assert occlusion_code(0.9) == OCCLUSION["none"]
    assert occlusion_code(0.5) == OCCLUSION["part"]
    assert occlusion_code(0.05) == OCCLUSION["part"]
    assert occlusion_code(0.0) == OCCLUSION["full"]


def test_no_gesture_labels_are_invented(video):
    """The one thing CARLA genuinely cannot supply. Every behaviour list that
    would need a human annotator stays empty."""
    for track in video.pedestrians.values():
        assert track.action == []
        assert track.look == []
        assert track.gesture == []
        # But the track is still supervised for crossing, so windowing keeps it.
        assert track.has_behavior


def test_unlabelled_gesture_groups_are_masked_out(video):
    track = video.pedestrians["ep_fixture_0000_w0"]
    labels = encode_track(track)
    assert labels.shape == (len(track.frames), len(GESTURE_GROUPS))

    # Without masking, the shared encoder falls back to a default class for the
    # groups CARLA says nothing about. That default is not a label the
    # simulator produced, and training on it would be training on a guess.
    before = {g: labels[:, i].copy() for i, g in enumerate(GESTURE_GROUPS)}
    assert (before["atomic"] != IGNORE_INDEX).any()

    masked = mask_unlabelled_gestures(labels)
    for i, group in enumerate(GESTURE_GROUPS):
        if group in UNLABELLED_GESTURE_GROUPS:
            assert (masked[:, i] == IGNORE_INDEX).all(), group
        else:
            # contextual survives: it comes from the crossing state, which the
            # simulator knows exactly.
            assert group == "contextual"
            assert (masked[:, i] != IGNORE_INDEX).all()


def test_lead_frames_converts_seconds_at_the_recording_rate(db):
    assert db.fps("ep_fixture_0000") == 20.0
    # 1.5 s of look-ahead is 30 frames at 20 Hz and 45 at the 30 fps of JAAD.
    assert lead_frames(1.5, 20.0) == 30
    assert lead_frames(1.5, 30.0) == 45


def test_frames_file_is_optional(db):
    assert db.frames_path("ep_fixture_0000") is None
    path = db.frames_path("ep_fixture_0001")
    assert path is not None
    with np.load(path) as z:
        frames = z["frames"]
    small = db.load("ep_fixture_0001")
    assert frames.shape == (small.num_frames, small.height, small.width, 3)
    assert frames.dtype == np.uint8


def test_a_recording_from_a_newer_collector_is_rejected(tmp_path):
    (tmp_path / "bad.json").write_text('{"format": "pif-carla-recording", "version": 9}')
    with pytest.raises(ValueError, match="format v9"):
        CARLARecordings(tmp_path).load("bad")


def test_a_file_that_is_not_a_recording_is_rejected(tmp_path):
    (tmp_path / "bad.json").write_text('{"hello": 1}')
    with pytest.raises(ValueError, match="not a pif-carla-recording"):
        CARLARecordings(tmp_path).load("bad")


def test_flows_through_the_existing_windowing_unchanged(db, video):
    """The whole reason the parser fills the shared schema. Nothing in
    pif.data.windows knows CARLA exists.
    """
    lead = lead_frames(1.5, db.fps("ep_fixture_0000"))
    windows = video_windows(video, window=16, lead=lead, stride=4)
    assert windows

    for w in windows:
        assert w.dataset == "carla"
        assert w.pose.shape == (16, POSE_DIM)
        assert w.context.shape == (16, CONTEXT_DIM)
        assert w.gesture.shape == (16, len(GESTURE_GROUPS))
        assert w.motion.shape == (16, 4)
        assert np.isfinite(w.context).all()
        # Contiguous, so the velocity channels are real differences.
        assert int(w.frames[-1] - w.frames[0]) == 15

    # Rule 1 of windowing: never observe past the moment they step out.
    crosser = [w for w in windows if w.ped_id == "ep_fixture_0000_w0"]
    assert crosser
    assert max(w.t_end for w in crosser) <= W0_CROSSING_FRAME

    # And the label is the crossing state `lead` frames after the window ends.
    for w in crosser:
        assert w.intent == (1 if w.t_end + lead >= W0_CROSSING_FRAME else 0)
    assert any(w.intent == 1 and w.t_end < W0_CROSSING_FRAME for w in crosser)

    # The walker who stopped at the kerb is a negative all the way through.
    hesitater = [w for w in windows if w.ped_id == "ep_fixture_0000_w1"]
    assert hesitater
    assert all(w.intent == 0 for w in hesitater)

    # "irrelevant" means on the road but not in our way, so it is a negative
    # too. Same reading PIE uses.
    bystander = [w for w in windows if w.ped_id == "ep_fixture_0000_w3"]
    assert bystander
    assert all(w.intent == 0 for w in bystander)


def test_occlusion_reaches_the_context_features(video):
    """The occluded-crosser case is this project's documented failure mode, so
    the occlusion channel has to actually carry through to the model input."""
    windows = video_windows(video, window=16, lead=30, stride=1)
    occluded = [w for w in windows if w.ped_id == "ep_fixture_0000_w2"]
    assert occluded
    from pif.features.context import CTX

    values = np.concatenate([w.context[:, CTX["occlusion"]] for w in occluded])
    assert values.max() > 0.0


# --------------------------------------------------------------------------
# The collector's camera maths.
#
# scripts/collect_carla.py only runs against a live simulator, but the 3D-to-2D
# projection inside it is ordinary arithmetic and is the easiest thing in the
# whole collector to get wrong. It imports carla softly for exactly this
# reason, so these two tests run here, in the environment that has no carla.
# --------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_carla import intrinsics, project  # noqa: E402


def test_intrinsics_from_field_of_view():
    k = intrinsics(1280, 720, 90.0)
    # At a 90 degree horizontal field of view the focal length is half the
    # image width, because the image edge sits at 45 degrees.
    assert k[0, 0] == pytest.approx(640.0)
    assert k[1, 1] == pytest.approx(640.0)
    assert (k[0, 2], k[1, 2]) == (640.0, 360.0)


def test_projection_uses_the_unreal_axis_convention():
    """CARLA is x forward, y right, z up. A pinhole matrix is x right, y down,
    z forward. Getting that swap wrong mirrors every box in the dataset."""
    k = intrinsics(1280, 720, 90.0)
    identity = np.eye(4)  # camera at the world origin, facing along +x
    points = np.array(
        [
            [10.0, 0.0, 0.0],  # straight ahead
            [10.0, 1.0, 0.0],  # one metre to the right
            [10.0, 0.0, 1.0],  # one metre up
        ]
    )
    pixels, depth = project(points, identity, k)

    assert depth == pytest.approx([10.0, 10.0, 10.0])
    assert pixels[0] == pytest.approx([640.0, 360.0])
    assert pixels[1][0] > pixels[0][0]   # right in the world is right in frame
    assert pixels[2][1] < pixels[0][1]   # up in the world is up in frame


def test_box_overlap_fraction_backs_up_the_ray_casts():
    """The second occlusion measurement in the collector: how much of a
    walker's box is hidden behind a nearer vehicle's box."""
    from scripts.collect_carla import box_overlap_fraction

    walker = (0.0, 0.0, 10.0, 10.0)
    assert box_overlap_fraction(walker, []) == 0.0
    assert box_overlap_fraction(walker, [(20.0, 20.0, 30.0, 30.0)]) == 0.0
    assert box_overlap_fraction(walker, [(5.0, 0.0, 15.0, 10.0)]) == pytest.approx(0.5)
    # Fully behind something: capped at 1, never above it.
    assert box_overlap_fraction(walker, [(-5.0, -5.0, 15.0, 15.0)]) == 1.0


# --------------------------------------------------------------------------
# Wiring into the dataset builder.
# --------------------------------------------------------------------------


def test_build_splits_by_episode_and_masks_gestures():
    """Two things that would each quietly ruin a result.

    A window-level split leaks: consecutive windows share 12 of their 16
    frames, so the same pedestrian would sit on both sides and the score would
    be inflated. And an unmasked gesture column trains three of the four heads
    against a default class CARLA never produced.
    """
    from pif.config import DataConfig
    from pif.data.build import build

    cfg = DataConfig(
        source="carla",
        root=str(FIXTURES),
        window=8,
        lead=4,
        stride=4,
        horizon=4,
        val_fraction=0.5,
    )
    train, val = build(cfg)
    assert train and val

    train_episodes = {w.video_id for w in train}
    val_episodes = {w.video_id for w in val}
    assert not (train_episodes & val_episodes)
    assert train_episodes | val_episodes == {"ep_fixture_0000", "ep_fixture_0001"}

    gestures = np.stack([w.gesture for w in train + val])
    for i, group in enumerate(GESTURE_GROUPS):
        if group in UNLABELLED_GESTURE_GROUPS:
            assert (gestures[:, :, i] == IGNORE_INDEX).all(), group
        else:
            assert (gestures[:, :, i] != IGNORE_INDEX).any(), group


def test_build_rejects_a_directory_with_no_recordings(tmp_path):
    from pif.config import DataConfig
    from pif.data.build import MissingAnnotations, build

    with pytest.raises(MissingAnnotations, match="no CARLA recordings"):
        build(DataConfig(source="carla", root=str(tmp_path)))


def test_occlusion_mask_selects_the_windows_that_were_hidden(video):
    """The ablation reports the occluded subset separately, so the mask that
    defines it has to pick out exactly the windows with a non-zero occlusion
    channel."""
    from pif.features.context import CTX
    from scripts.run_ablation import occlusion_mask

    windows = video_windows(video, window=16, lead=30, stride=4)
    mask = occlusion_mask(windows)
    assert mask.any() and not mask.all()
    for w, hidden in zip(windows, mask):
        assert (w.context[:, CTX["occlusion"]].max() > 0.0) == bool(hidden)
    # The occluded episode's walker is the one that was behind the parked car.
    assert {w.ped_id for w, m in zip(windows, mask) if m} == {"ep_fixture_0000_w2"}
