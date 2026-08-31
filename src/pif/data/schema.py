"""Dataset-neutral containers that both the JAAD and the PIE parser fill in.

JAAD and PIE are annotated by the same group and share most of their vocabulary,
but not their file layout: JAAD keeps four XML trees per video in four sibling
directories, PIE keeps one tree per video inside per-set zip archives and adds
OBD telemetry. Normalising both into these structures keeps everything
downstream (windowing, features, models) from having to know which one it got.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# JAAD and PIE both encode occlusion the same way; the models want a scalar.
OCCLUSION = {"none": 0, "part": 1, "full": 2}

# The JAAD ego-speed proxy table used to live here. It moved to
# pif.features.context, which is the only thing that reads it: it is a choice
# about how to encode a feature, not a fact about the annotation format, and
# keeping it here made pif.features import pif.data at module level, which
# closed an import cycle (pif.data.windows already imports pif.features).


@dataclass
class PedTrack:
    """One annotated pedestrian in one video."""

    ped_id: str
    frames: list[int] = field(default_factory=list)
    bbox: list[tuple[float, float, float, float]] = field(default_factory=list)
    occlusion: list[int] = field(default_factory=list)

    # Per-frame behaviour tags. JAAD names the gesture column `hand_gesture` and
    # adds `nod`/`reaction`; PIE names it `gesture` and has neither. Missing
    # columns just stay empty rather than being filled with a fake default.
    action: list[str] = field(default_factory=list)
    look: list[str] = field(default_factory=list)
    gesture: list[str] = field(default_factory=list)
    cross: list[str] = field(default_factory=list)

    # Track-level attributes from the *_attributes.xml file.
    attributes: dict[str, object] = field(default_factory=dict)

    # Per-frame appearance bits, JAAD only and only for the high-visibility
    # subset. Keyed by frame number so a partially annotated track is not a
    # problem.
    appearance: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def has_behavior(self) -> bool:
        """JAAD only tags behaviour on ids ending in 'b'; the rest are boxes only."""
        return bool(self.cross)

    @property
    def crossing_point(self) -> int | None:
        """Frame at which the pedestrian steps into the road, or None if never."""
        cp = self.attributes.get("crossing_point")
        if cp is None:
            return None
        cp = int(cp)  # type: ignore[arg-type]
        return cp if cp >= 0 else None

    @property
    def crossing_label(self) -> int:
        """-1 = not relevant to the ego lane, 0 = did not cross, 1 = crossed."""
        c = self.attributes.get("crossing")
        return int(c) if c is not None else -1  # type: ignore[arg-type]

    def frame_index(self, frame: int) -> int | None:
        try:
            return self.frames.index(frame)
        except ValueError:
            return None


@dataclass
class TrafficObject:
    """A non-pedestrian annotated box: crosswalk, sign, traffic light, vehicle.

    Only PIE boxes these; in JAAD the same information arrives as per-frame
    flags on the whole scene (see VideoAnnotation.scene).
    """

    obj_id: str
    obj_class: str
    obj_type: str | None = None
    frames: list[int] = field(default_factory=list)
    bbox: list[tuple[float, float, float, float]] = field(default_factory=list)
    state: list[str] = field(default_factory=list)


@dataclass
class VideoAnnotation:
    video_id: str
    dataset: str
    width: int
    height: int
    num_frames: int
    pedestrians: dict[str, PedTrack] = field(default_factory=dict)
    traffic: dict[str, TrafficObject] = field(default_factory=dict)

    # Per-frame ego state. JAAD: {"action": "moving_slow"}. PIE: the full OBD
    # row (OBD_speed, GPS_speed, accX..., yaw, ...).
    ego: dict[int, dict[str, object]] = field(default_factory=dict)

    # Per-frame scene flags. JAAD fills these from *_traffic.xml
    # (ped_crossing, ped_sign, stop_sign, traffic_light); PIE leaves them empty
    # because the same facts are boxed objects there instead.
    scene: dict[int, dict[str, object]] = field(default_factory=dict)

    road_type: str | None = None

    def behavior_tracks(self) -> list[PedTrack]:
        return [t for t in self.pedestrians.values() if t.has_behavior]
