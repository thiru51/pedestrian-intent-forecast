"""Scene-context features, one vector per pedestrian per frame.

Why this branch exists at all: a person standing still at the edge of the road
with their body squared up to traffic looks, in pose space, exactly like a
person standing still at a bus stop. The difference is entirely in where they
are standing and what is around them. Pose tells you the posture; context tells
you what the posture is aimed at.

Everything here is derived from annotations that JAAD and PIE actually ship, or
from the bounding-box geometry, so it can be computed without the video files.
The three that carry most of the weight:

  crosswalk proxy  - PIE boxes crosswalks explicitly, JAAD gives a per-frame
                     ped_crossing flag for the whole scene. Both are folded
                     into one channel plus a "this is a boxed crosswalk right
                     next to them" channel.
  curb distance    - no dataset labels the kerb, so it is estimated from the
                     ego-lane corridor: how far the pedestrian's feet are, in
                     box-widths, from the edge of the wedge the car will drive
                     through. Crossing means that number goes to zero.
  ego proximity    - box height over image height is a monotone stand-in for
                     1/distance under a fixed focal length, so height times ego
                     speed is a usable inverse time-to-contact.
"""

from __future__ import annotations

import numpy as np

from ..data.schema import JAAD_EGO_SPEED_PROXY, VideoAnnotation, PedTrack

CONTEXT_FEATURES: tuple[str, ...] = (
    "cx",               # box centre, image-normalised
    "cy",
    "box_h",            # box height / image height
    "box_ar",           # width / height
    "vx",               # centre velocity per frame, box-widths
    "vy",
    "scale_rate",       # d(log height)/dt: positive = approaching
    "curb_dist",        # feet to ego-corridor edge, in box widths, clipped
    "corridor_inside",  # 1 when the feet are already inside the ego corridor
    "crosswalk_scene",  # scene-level: a marked crossing is present
    "crosswalk_near",   # box-level: overlap with a boxed crosswalk (PIE only)
    "ego_speed",        # m/s / 20
    "ego_accel",        # d(ego_speed)/dt
    "inv_ttc",          # box_h * ego_speed: crude inverse time-to-contact
    "occlusion",        # 0 / 0.5 / 1
    "tl_none",
    "tl_red",
    "tl_yellow",
    "tl_green",
    "ped_sign",
    "stop_sign",
    "signalized",       # track attribute: crossing is signal-controlled
    "num_lanes",        # / 6
)
CONTEXT_DIM = len(CONTEXT_FEATURES)
CTX = {name: i for i, name in enumerate(CONTEXT_FEATURES)}

# The ego corridor is a wedge in image space: about one lane wide at the bottom
# of the frame, narrowing towards the horizon. These are fractions of image
# width at the bottom edge and at the vanishing row, and the vanishing row is a
# fraction of image height. Coarse, but both datasets are forward-facing dashcam
# with a similar mount, so a fixed wedge is a reasonable stand-in for the
# calibration PIE ships and JAAD does not.
CORRIDOR_HALF_BOTTOM = 0.28
CORRIDOR_HALF_TOP = 0.03
HORIZON_Y = 0.55

_TRAFFIC_LIGHT_STATES = ("n/a", "red", "yellow", "green")


def _corridor_half_width(y_norm: float) -> float:
    t = np.clip((y_norm - HORIZON_Y) / (1.0 - HORIZON_Y), 0.0, 1.0)
    return float(CORRIDOR_HALF_TOP + t * (CORRIDOR_HALF_BOTTOM - CORRIDOR_HALF_TOP))


def _ego_speed(video: VideoAnnotation, frame: int) -> float:
    row = video.ego.get(frame)
    if not row:
        return 0.0
    if "OBD_speed" in row:  # PIE: real telemetry, km/h
        return float(row["OBD_speed"]) / 3.6  # type: ignore[arg-type]
    if "GPS_speed" in row:
        return float(row["GPS_speed"]) / 3.6  # type: ignore[arg-type]
    action = str(row.get("action", "stopped"))  # JAAD: categorical
    return JAAD_EGO_SPEED_PROXY.get(action, 0.0)


def _crosswalk_overlap(video: VideoAnnotation, frame: int, box) -> float:
    """Horizontal overlap of the pedestrian's feet with any boxed crosswalk,
    as a fraction of the pedestrian's box width. PIE only; JAAD has no boxes
    for infrastructure and falls back to the scene flag."""
    x1, _, x2, y2 = box
    width = max(x2 - x1, 1.0)
    best = 0.0
    for obj in video.traffic.values():
        if obj.obj_class != "crosswalk":
            continue
        idx = _nearest_frame_index(obj.frames, frame)
        if idx is None:
            continue
        cx1, cy1, cx2, cy2 = obj.bbox[idx]
        if not (cy1 - 0.25 * (cy2 - cy1) <= y2 <= cy2 + 0.5 * (cy2 - cy1)):
            continue
        overlap = min(x2, cx2) - max(x1, cx1)
        best = max(best, overlap / width)
    return float(np.clip(best, 0.0, 1.0))


def _nearest_frame_index(frames: list[int], frame: int) -> int | None:
    if not frames:
        return None
    if frames[0] <= frame <= frames[-1]:
        lo, hi = 0, len(frames) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if frames[mid] < frame:
                lo = mid + 1
            else:
                hi = mid
        return lo
    return None


def _traffic_light_state(video: VideoAnnotation, frame: int) -> str:
    scene = video.scene.get(frame)
    if scene is not None:
        return str(scene.get("traffic_light", "n/a"))
    # PIE: state lives on the boxed traffic-light tracks instead. Red anywhere
    # in view is the conservative reading.
    seen = "n/a"
    for obj in video.traffic.values():
        if obj.obj_class != "traffic_light" or not obj.state:
            continue
        idx = _nearest_frame_index(obj.frames, frame)
        if idx is None or idx >= len(obj.state):
            continue
        state = obj.state[idx]
        if state in ("red", "yellow", "green"):
            seen = state if seen == "n/a" or state == "red" else seen
    return seen


class ContextEncoder:
    """Builds context vectors for one video's pedestrians.

    Held as a class rather than a function because the scene-level lookups
    (crosswalk boxes, traffic light state) are the same for every pedestrian in
    the video and are worth resolving once.
    """

    def __init__(self, video: VideoAnnotation):
        self.video = video
        self.w = float(video.width or 1920)
        self.h = float(video.height or 1080)

    def encode_track(self, track: PedTrack) -> np.ndarray:
        """(len(track.frames), CONTEXT_DIM) float32."""
        n = len(track.frames)
        out = np.zeros((n, CONTEXT_DIM), dtype=np.float32)
        if n == 0:
            return out

        signalized = 1.0 if str(track.attributes.get("signalized", "n/a")) not in (
            "n/a",
            "NS",
            "",
        ) else 0.0
        try:
            num_lanes = float(track.attributes.get("num_lanes", 2) or 2) / 6.0
        except (TypeError, ValueError):
            num_lanes = 2.0 / 6.0

        prev_speed = None
        prev_centre = None
        prev_log_h = None

        for i, frame in enumerate(track.frames):
            x1, y1, x2, y2 = track.bbox[i]
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            cx = (x1 + x2) / 2.0 / self.w
            cy = (y1 + y2) / 2.0 / self.h
            foot_x = (x1 + x2) / 2.0 / self.w
            foot_y = y2 / self.h

            half = _corridor_half_width(foot_y)
            lateral = abs(foot_x - 0.5)
            # Distance to the corridor edge measured in box widths, so it means
            # roughly "how many of their own body widths from the car's path".
            curb_dist = float(
                np.clip((lateral - half) * self.w / bw, -4.0, 8.0) / 8.0
            )

            speed = _ego_speed(self.video, frame)
            accel = 0.0 if prev_speed is None else (speed - prev_speed)
            prev_speed = speed

            log_h = float(np.log(bh))
            scale_rate = 0.0 if prev_log_h is None else (log_h - prev_log_h) * 30.0
            prev_log_h = log_h

            if prev_centre is None:
                vx = vy = 0.0
            else:
                vx = (cx * self.w - prev_centre[0]) / bw
                vy = (cy * self.h - prev_centre[1]) / bh
            prev_centre = (cx * self.w, cy * self.h)

            scene = self.video.scene.get(frame, {})
            crosswalk_scene = float(scene.get("ped_crossing", 0) or 0)
            crosswalk_near = _crosswalk_overlap(
                self.video, frame, (x1, y1, x2, y2)
            ) if self.video.traffic else 0.0
            if crosswalk_near > 0.0:
                crosswalk_scene = max(crosswalk_scene, 1.0)

            light = _traffic_light_state(self.video, frame)
            occ = track.occlusion[i] / 2.0 if i < len(track.occlusion) else 0.0

            row = out[i]
            row[CTX["cx"]] = cx
            row[CTX["cy"]] = cy
            row[CTX["box_h"]] = bh / self.h
            row[CTX["box_ar"]] = bw / bh
            row[CTX["vx"]] = float(np.clip(vx, -2.0, 2.0))
            row[CTX["vy"]] = float(np.clip(vy, -2.0, 2.0))
            row[CTX["scale_rate"]] = float(np.clip(scale_rate, -5.0, 5.0))
            row[CTX["curb_dist"]] = curb_dist
            row[CTX["corridor_inside"]] = 1.0 if lateral <= half else 0.0
            row[CTX["crosswalk_scene"]] = crosswalk_scene
            row[CTX["crosswalk_near"]] = crosswalk_near
            row[CTX["ego_speed"]] = speed / 20.0
            row[CTX["ego_accel"]] = float(np.clip(accel, -5.0, 5.0))
            row[CTX["inv_ttc"]] = float(np.clip(bh / self.h * speed / 20.0, 0.0, 1.0))
            row[CTX["occlusion"]] = occ
            for k, state in enumerate(_TRAFFIC_LIGHT_STATES):
                row[CTX[f"tl_{'none' if state == 'n/a' else state}"]] = (
                    1.0 if light == state else 0.0
                )
            row[CTX["ped_sign"]] = float(scene.get("ped_sign", 0) or 0)
            row[CTX["stop_sign"]] = float(scene.get("stop_sign", 0) or 0)
            row[CTX["signalized"]] = signalized
            row[CTX["num_lanes"]] = num_lanes

        return out
