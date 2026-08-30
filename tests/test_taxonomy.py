"""The taxonomy and the annotation-string adapter.

These are the tests that would catch a silent relabelling: if someone reorders
an enum or changes a mapping, every trained checkpoint on disk starts meaning
something different, and nothing else in the pipeline would notice.
"""

from __future__ import annotations

import numpy as np

from pif.data.schema import PedTrack
from pif.features.gesture import (
    GESTURE_GROUPS,
    GESTURE_LOGIT_DIM,
    GROUP_SLICES,
    IGNORE_INDEX,
    apply_phone_override,
    encode_track,
    one_hot,
)
from pif.taxonomy import (
    Atomic,
    Communicative,
    Contextual,
    GESTURE_DIM,
    GROUP_SIZES,
    Transportive,
    map_atomic,
    map_communicative,
    map_contextual,
)


def test_group_widths_agree():
    assert GESTURE_LOGIT_DIM == GESTURE_DIM == sum(GROUP_SIZES.values())
    assert sum(s.stop - s.start for s in GROUP_SLICES.values()) == GESTURE_LOGIT_DIM
    # Slices must tile the vector with no gap and no overlap.
    edges = sorted((GROUP_SLICES[g].start, GROUP_SLICES[g].stop) for g in GESTURE_GROUPS)
    assert edges[0][0] == 0
    for (_, prev_stop), (start, _) in zip(edges, edges[1:]):
        assert prev_stop == start


def test_atomic_mapping():
    assert map_atomic("walking") == Atomic.WALKING
    assert map_atomic("standing") == Atomic.STANDING
    assert map_atomic("running") == Atomic.RUNNING
    # Unknown and missing both fall back rather than raising: the released XML
    # is not perfectly uniform.
    assert map_atomic("__undefined__") == Atomic.STANDING
    assert map_atomic(None) == Atomic.STANDING


def test_gesture_beats_gaze():
    # An explicit hand signal outranks where the head is pointing.
    assert map_communicative("yield", "looking") == Communicative.WAVING_THROUGH
    assert map_communicative("__undefined__", "looking") == Communicative.LOOKING_AT_TRAFFIC
    assert map_communicative(None, "not-looking") == Communicative.NONE


def test_contextual_separates_waiting_from_walking():
    # This is the distinction the whole project turns on: standing still and
    # not crossing is "waiting", moving and not crossing is "walking along".
    assert map_contextual("not-crossing", "standing") == Contextual.WAITING_TO_CROSS
    assert map_contextual("not-crossing", "walking") == Contextual.WALKING_ALONG
    assert map_contextual("crossing", "walking") == Contextual.CROSSING
    # PIE's third value. Crossing somewhere the ego vehicle does not care about
    # is not a crossing event for a planner.
    assert map_contextual("crossing-irrelevant", "walking") == Contextual.WALKING_ALONG


def _track(n: int = 4) -> PedTrack:
    t = PedTrack(ped_id="0_1_1b")
    t.frames = list(range(n))
    t.bbox = [(10.0, 10.0, 30.0, 70.0)] * n
    t.occlusion = [0] * n
    t.action = ["standing"] * n
    t.look = ["looking"] * n
    t.gesture = ["__undefined__"] * n
    t.cross = ["not-crossing"] * n
    return t


def test_encode_track_shapes_and_ignore():
    labels = encode_track(_track())
    assert labels.shape == (4, len(GESTURE_GROUPS))
    assert labels.dtype == np.int64
    assert (labels[:, 0] == Atomic.STANDING).all()
    assert (labels[:, 1] == Communicative.LOOKING_AT_TRAFFIC).all()
    assert (labels[:, 2] == Contextual.WAITING_TO_CROSS).all()
    # No appearance file, so transportive is unsupervised and must be ignored,
    # not defaulted to NONE - defaulting would teach the head a constant.
    assert (labels[:, 3] == IGNORE_INDEX).all()


def test_box_only_track_is_all_ignore():
    t = PedTrack(ped_id="0_1_5")
    t.frames = list(range(3))
    t.bbox = [(0.0, 0.0, 1.0, 1.0)] * 3
    t.occlusion = [0] * 3
    labels = encode_track(t)
    assert (labels == IGNORE_INDEX).all()


def test_appearance_gives_transportive_and_phone():
    t = _track()
    t.appearance = {
        0: {"backpack": 1, "phone": 0},
        1: {"stroller_cart": 1, "phone": 0},
        2: {"phone": 1},
    }
    labels = apply_phone_override(t, encode_track(t))
    assert labels[0, 3] == Transportive.CARRYING
    assert labels[1, 3] == Transportive.PUSHING
    assert labels[2, 3] == Transportive.NONE
    assert labels[3, 3] == IGNORE_INDEX          # no appearance row for frame 3
    assert labels[2, 1] == Communicative.PHONE_DISTRACTED
    assert labels[0, 1] == Communicative.LOOKING_AT_TRAFFIC  # phone flag off


def test_phone_does_not_override_an_explicit_gesture():
    t = _track()
    t.gesture = ["yield"] * 4
    t.appearance = {i: {"phone": 1} for i in range(4)}
    labels = apply_phone_override(t, encode_track(t))
    assert (labels[:, 1] == Communicative.WAVING_THROUGH).all()


def test_one_hot_layout():
    labels = np.array([[1, 2, 3, IGNORE_INDEX]], dtype=np.int64)
    oh = one_hot(labels)
    assert oh.shape == (1, GESTURE_LOGIT_DIM)
    assert oh.sum() == 3.0  # the ignored group contributes nothing
    for gi, group in enumerate(GESTURE_GROUPS):
        c = int(labels[0, gi])
        if c >= 0:
            assert oh[0, GROUP_SLICES[group].start + c] == 1.0
