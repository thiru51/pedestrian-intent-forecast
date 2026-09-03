"""Parser for recordings made by scripts/collect_carla.py.

Why this module exists: JAAD and PIE both need the video clips before any of
this project's pose work is possible, and both are access-gated. CARLA is a
driving simulator that is free to download and run, so it can produce crossing
scenarios with labels that are exact rather than hand-annotated. This parser
turns one recorded episode into the same PedTrack / VideoAnnotation structures
the JAAD and PIE parsers produce, so windowing, features and models do not
change at all.

What CARLA gives that the real datasets do not:

  exact boxes        the walker's 3D bounding box is known, so the 2D box is a
                     projection and not a human's guess
  exact crossing     "on the road" is a map query, not an annotator's opinion
  exact occlusion    visibility is measured by casting rays at the walker

What CARLA does NOT give, and this is the important one:

  gestures           A CARLA walker has no hand signals, no gaze target and no
                     phone. The `action`, `look` and `gesture` lists on every
                     track this parser returns are empty on purpose. Nothing
                     here invents them. That is precisely why real JAAD/PIE
                     evaluation still matters once access lands: the gesture
                     branch of this project cannot be trained or judged on
                     simulator data.

Use `mask_unlabelled_gestures` on the output of pif.features.gesture.encode_track
before training on a CARLA recording. See that function for why.

Recording layout, one episode per pair of files under the output directory:

    ep_0000.json    the annotation, the format this module reads
    ep_0000.npz     the RGB frames, optional, only needed for pose extraction

Frame rate: CARLA is recorded at 20 Hz (a fixed 0.05 s step), while JAAD and PIE
are 30 fps. Frame counts therefore mean different amounts of time in the two
sources. `lead_frames` converts a lead time in seconds into the right number of
frames for a given recording, and `CARLARecordings.fps` reports what a
recording was captured at. Passing the JAAD default of 45 frames to a CARLA
recording would silently ask for 2.25 s of look-ahead instead of 1.5 s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import OCCLUSION, PedTrack, TrafficObject, VideoAnnotation

DATASET = "carla"

# Written into every recording so a stale file from an older collector is a
# clear error rather than a confusing KeyError.
FORMAT = "pif-carla-recording"
FORMAT_VERSION = 1

# Per-frame `cross` strings. These are the same vocabulary JAAD and PIE use, and
# pif.data.windows treats exactly "crossing" as the positive class, so the
# strings have to match those datasets rather than read nicely.
CROSSING = "crossing"
NOT_CROSSING = "not-crossing"
# On a driving lane, but not one the ego vehicle will pass through. PIE calls
# this "irrelevant" and counts it as a negative, because for a planner it is.
IRRELEVANT = "irrelevant"

# Fraction of the walker that has to be visible from the camera for each
# occlusion code. The collector measures the fraction by casting rays from the
# camera at points on the walker's 3D box and counting how many arrive without
# hitting something else first, so these thresholds are applied to a real
# measurement rather than to a guess. A collector that could not measure
# partial visibility writes only 1.0 or 0.0, which lands on codes 0 and 2 and
# never on 1.
FULLY_VISIBLE_ABOVE = 0.9
FULLY_OCCLUDED_BELOW = 0.05

# Gesture groups a CARLA recording cannot supervise. pif.features.gesture reads
# `track.action`, `track.look` and `track.gesture`, which this parser leaves
# empty, and its mapping functions fall back to a default class instead of
# returning "unknown". Left alone that would train the gesture head towards
# STANDING and NONE on every CARLA frame, which is a label the simulator never
# actually made. `mask_unlabelled_gestures` replaces those with IGNORE_INDEX.
# "contextual" is not in this list: it is derived from the per-frame `cross`
# string, which CARLA does know exactly.
UNLABELLED_GESTURE_GROUPS: tuple[str, ...] = ("atomic", "communicative", "transportive")


class CARLARecordings:
    """Reads a directory of episodes written by scripts/collect_carla.py.

    Same shape of interface as JAADAnnotations and PIEAnnotations: list the ids,
    then load one at a time.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.root} is not a directory. Collect some episodes first: "
                "see scripts/collect_carla.py"
            )
        self._cache: dict[str, dict[str, Any]] = {}

    def video_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def fps(self, video_id: str) -> float:
        """Frames per second this episode was captured at. Feed it to
        `lead_frames` rather than assuming the 30 fps of JAAD and PIE."""
        return float(self._raw(video_id).get("fps", 20.0))

    def frames_path(self, video_id: str) -> Path | None:
        """Where the RGB frames for this episode are, if they were saved."""
        name = self._raw(video_id).get("frames_file")
        if not name:
            return None
        path = self.root / str(name)
        return path if path.is_file() else None

    def load(self, video_id: str) -> VideoAnnotation:
        raw = self._raw(video_id)
        video = VideoAnnotation(
            video_id=video_id,
            dataset=DATASET,
            width=int(raw["width"]),
            height=int(raw["height"]),
            num_frames=int(raw["num_frames"]),
            road_type=raw.get("road_type"),
        )
        for frame, row in (raw.get("ego") or {}).items():
            video.ego[int(frame)] = dict(row)
        for frame, row in (raw.get("scene") or {}).items():
            video.scene[int(frame)] = dict(row)
        for entry in raw.get("walkers") or []:
            track = _walker_track(entry)
            video.pedestrians[track.ped_id] = track
        for entry in raw.get("traffic") or []:
            obj = _traffic_object(entry)
            video.traffic[obj.obj_id] = obj
        return video

    def _raw(self, video_id: str) -> dict[str, Any]:
        if video_id not in self._cache:
            path = self.root / f"{video_id}.json"
            if not path.is_file():
                raise FileNotFoundError(f"no CARLA recording at {path}")
            self._cache[video_id] = _validated(json.loads(path.read_text()), path)
        return self._cache[video_id]


def _validated(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    if raw.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    version = int(raw.get("version", 0))
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} is recording format v{version}, this parser reads "
            f"v{FORMAT_VERSION}. Recollect the episode."
        )
    return raw


def _walker_track(entry: dict[str, Any]) -> PedTrack:
    track = PedTrack(ped_id=str(entry["ped_id"]))
    track.frames = [int(f) for f in entry["frames"]]
    track.bbox = [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in entry["bbox"]
    ]
    track.occlusion = [
        occlusion_code(float(v)) for v in entry.get("visible_fraction", [])
    ]
    track.cross = [str(c) for c in entry["cross"]]

    # action / look / gesture stay empty. See the module docstring: CARLA has
    # nothing truthful to put in them.

    track.attributes.update(entry.get("attributes") or {})
    track.attributes["scenario"] = entry.get("scenario", "unknown")
    ego_relevant = bool(entry.get("ego_relevant", True))
    track.attributes["ego_relevant"] = ego_relevant
    label, point = crossing_summary(track.frames, track.cross, ego_relevant)
    track.attributes["crossing"] = label
    track.attributes["crossing_point"] = point
    return track


def _traffic_object(entry: dict[str, Any]) -> TrafficObject:
    return TrafficObject(
        obj_id=str(entry["obj_id"]),
        obj_class=str(entry["obj_class"]),
        obj_type=entry.get("obj_type"),
        frames=[int(f) for f in entry.get("frames", [])],
        bbox=[
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for b in entry.get("bbox", [])
        ],
        state=[str(s) for s in entry.get("state", [])],
    )


def occlusion_code(visible_fraction: float) -> int:
    """Visible fraction -> the {none: 0, part: 1, full: 2} codes JAAD uses."""
    if visible_fraction >= FULLY_VISIBLE_ABOVE:
        return OCCLUSION["none"]
    if visible_fraction < FULLY_OCCLUDED_BELOW:
        return OCCLUSION["full"]
    return OCCLUSION["part"]


def crossing_summary(
    frames: list[int], cross: list[str], ego_relevant: bool
) -> tuple[int, int]:
    """Track-level (crossing, crossing_point) from the per-frame cross strings.

    The convention is JAAD's and is copied exactly, because pif.data.windows
    reads it and must not have to care which dataset it came from:

      crossing        -1 the pedestrian is not relevant to the ego vehicle
                       0 relevant, but never entered the road
                       1 relevant, and crossed
      crossing_point  the frame number where they first stepped into the road,
                      or -1 when that never happened

    crossing_point is a frame *number*, not a position in the list. Windowing
    compares it against frame numbers, and a track that starts partway into the
    episode would be cut in the wrong place if this returned an index.
    """
    if not ego_relevant:
        return -1, -1
    for frame, value in zip(frames, cross):
        if value == CROSSING:
            return 1, int(frame)
    return 0, -1


def lead_frames(seconds: float, fps: float) -> int:
    """Look-ahead in seconds -> frames, for a recording captured at `fps`.

    The windowing defaults in pif.data.windows are in frames and assume the
    30 fps of JAAD and PIE. CARLA runs at 20 Hz, so 1.5 s is 30 frames here and
    45 frames there.
    """
    return int(round(seconds * fps))


def mask_unlabelled_gestures(labels):
    """Blank out the gesture groups CARLA cannot supervise, in place.

    `labels` is what pif.features.gesture.encode_track returns: (T, 4) int64,
    one column per group in GESTURE_GROUPS order. Any column named in
    UNLABELLED_GESTURE_GROUPS is set to IGNORE_INDEX so the loss skips it,
    rather than being trained against a default that the simulator never
    claimed. Only the contextual column survives, because that one comes from
    the per-frame crossing state, which CARLA knows exactly.
    """
    from ..features.gesture import GESTURE_GROUPS, IGNORE_INDEX

    for gi, group in enumerate(GESTURE_GROUPS):
        if group in UNLABELLED_GESTURE_GROUPS:
            labels[:, gi] = IGNORE_INDEX
    return labels
