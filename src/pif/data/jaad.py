"""Parser for the JAAD annotation release (Rasouli, Kotseruba & Tsotsos, ICCVW 2017).

Layout, all under external/jaad:

    annotations/video_XXXX.xml             CVAT dump: per-frame boxes + behaviour
    annotations_attributes/..._attributes  per-track: crossing, crossing_point, ...
    annotations_traffic/..._traffic.xml    per-frame scene flags + road_type
    annotations_vehicle/..._vehicle.xml    per-frame ego action
    annotations_appearance/..._appearance  per-frame appearance bits (subset only)
    split_ids/{default,high_visibility,all_videos}/{train,val,test}.txt

Only the XML is needed here. The video clips are a separate, much larger
download and nothing in this module touches them.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .schema import PedTrack, VideoAnnotation

DATASET = "jaad"

# Per-frame behaviour columns in annotations/*.xml, mapped to PedTrack fields.
# JAAD also carries `nod` and `reaction`; neither is used by the current feature
# set, so they are read past rather than stored.
_BEHAVIOR_COLUMNS = {
    "action": "action",
    "look": "look",
    "hand_gesture": "gesture",
    "cross": "cross",
}

_ATTR_INT_KEYS = {
    "crossing",
    "crossing_point",
    "decision_point",
    "group_size",
    "num_lanes",
}


class JAADAnnotations:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / "annotations").is_dir():
            raise FileNotFoundError(
                f"{self.root} does not look like a JAAD checkout "
                "(no annotations/ directory)"
            )

    def video_ids(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "annotations").glob("video_*.xml"))

    def split(self, name: str, subset: str = "default") -> list[str]:
        path = self.root / "split_ids" / subset / f"{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"no such JAAD split: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def load(self, video_id: str, with_appearance: bool = True) -> VideoAnnotation:
        video = self._load_boxes(video_id)
        self._load_attributes(video_id, video)
        self._load_traffic(video_id, video)
        self._load_vehicle(video_id, video)
        if with_appearance:
            self._load_appearance(video_id, video)
        return video

    def _load_boxes(self, video_id: str) -> VideoAnnotation:
        tree = ET.parse(self.root / "annotations" / f"{video_id}.xml")
        task = tree.find("./meta/task")
        assert task is not None
        video = VideoAnnotation(
            video_id=video_id,
            dataset=DATASET,
            width=int(task.findtext("./original_size/width", "1920")),
            height=int(task.findtext("./original_size/height", "1080")),
            num_frames=int(task.findtext("./size", "0")),
        )

        for track in tree.findall("./track"):
            boxes = track.findall("./box")
            if not boxes:
                continue
            ped_id = _attr_text(boxes[0], "id")
            if ped_id is None:
                continue
            ped = PedTrack(ped_id=ped_id)
            old_id = _attr_text(boxes[0], "old_id")
            # JAAD reuses one CVAT label for three kinds of object and only the
            # ones whose old_id says "pedestrian" carry behaviour columns; the
            # "ped"/"people" tracks are bounding boxes with occlusion only.
            behavioral = old_id is not None and "pedestrian" in old_id
            ped.attributes["old_id"] = old_id

            for box in boxes:
                if box.get("outside") == "1":
                    continue
                ped.frames.append(int(box.get("frame", "0")))
                ped.bbox.append(
                    (
                        float(box.get("xtl", "0")),
                        float(box.get("ytl", "0")),
                        float(box.get("xbr", "0")),
                        float(box.get("ybr", "0")),
                    )
                )
                ped.occlusion.append(_occlusion_scalar(_attr_text(box, "occlusion")))
                if behavioral:
                    for column, field_name in _BEHAVIOR_COLUMNS.items():
                        getattr(ped, field_name).append(_attr_text(box, column) or "")

            if ped.frames:
                video.pedestrians[ped_id] = ped
        return video

    def _load_attributes(self, video_id: str, video: VideoAnnotation) -> None:
        path = self.root / "annotations_attributes" / f"{video_id}_attributes.xml"
        if not path.is_file():
            return
        for ped in ET.parse(path).getroot().findall("./pedestrian"):
            ped_id = ped.get("id")
            track = video.pedestrians.get(ped_id or "")
            if track is None:
                continue
            for key, value in ped.attrib.items():
                if key in ("id", "old_id"):
                    continue
                track.attributes[key] = int(value) if key in _ATTR_INT_KEYS else value

    def _load_traffic(self, video_id: str, video: VideoAnnotation) -> None:
        path = self.root / "annotations_traffic" / f"{video_id}_traffic.xml"
        if not path.is_file():
            return
        root = ET.parse(path).getroot()
        video.road_type = root.findtext("./road_type")
        for frame in root.findall("./frame"):
            fid = int(frame.get("id", "0"))
            video.scene[fid] = {
                "ped_crossing": int(frame.get("ped_crossing", "0")),
                "ped_sign": int(frame.get("ped_sign", "0")),
                "stop_sign": int(frame.get("stop_sign", "0")),
                "traffic_light": frame.get("traffic_light", "n/a"),
            }

    def _load_vehicle(self, video_id: str, video: VideoAnnotation) -> None:
        path = self.root / "annotations_vehicle" / f"{video_id}_vehicle.xml"
        if not path.is_file():
            return
        for frame in ET.parse(path).getroot().findall("./frame"):
            video.ego[int(frame.get("id", "0"))] = {
                "action": frame.get("action", "stopped")
            }

    def _load_appearance(self, video_id: str, video: VideoAnnotation) -> None:
        # Only the 60-odd high-visibility videos have this file, so a miss is
        # normal and not an error. It is the only place JAAD records phone use,
        # which is what the PHONE_DISTRACTED gesture class needs.
        path = self.root / "annotations_appearance" / f"{video_id}_appearance.xml"
        if not path.is_file():
            return
        for track in ET.parse(path).getroot().findall("./track"):
            ped = video.pedestrians.get(track.get("id") or "")
            if ped is None:
                continue
            for box in track.findall("./box"):
                fid = int(box.get("frame", "0"))
                ped.appearance[fid] = {
                    k: int(v) for k, v in box.attrib.items() if k != "frame"
                }


def _attr_text(box: ET.Element, name: str) -> str | None:
    node = box.find(f'./attribute[@name="{name}"]')
    return node.text if node is not None else None


def _occlusion_scalar(value: str | None) -> int:
    return {"none": 0, "part": 1, "full": 2}.get(value or "none", 0)
