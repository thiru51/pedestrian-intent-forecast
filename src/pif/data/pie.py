"""Parser for the PIE annotation release (Rasouli et al., ICCV 2019).

PIE ships its annotations as three zip archives under external/pie/annotations:

    annotations.zip             set??/video_????_annt.xml  boxes + behaviour +
                                boxed traffic objects (crosswalk, sign, light)
    annotations_attributes.zip  set??/video_????_attributes.xml
    annotations_vehicle.zip     set??/video_????_obd.xml    OBD/GPS/IMU per frame

They are read straight out of the archives. Extracted they come to a few hundred
megabytes of XML for something we only ever stream through once, and leaving
them packed keeps external/ the size git actually got it at.

Two things PIE has that JAAD does not, and that this project cares about:

  intention_prob   human-rated probability that the pedestrian intends to cross,
                   collected from annotators watching only the frames *before*
                   the crossing decision. That is the closest thing either
                   dataset has to a direct label for the quantity we forecast.
  critical_point   frame at which the ego vehicle had to react. crossing_point
                   is the later frame where the pedestrian actually enters the
                   road, so critical_point - crossing_point is roughly the lead
                   time a planner would have wanted.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .schema import PedTrack, TrafficObject, VideoAnnotation

DATASET = "pie"

_MEMBER_RE = re.compile(r"(set\d+)/video_(\d+)_annt\.xml$")

_BEHAVIOR_COLUMNS = {
    "action": "action",
    "look": "look",
    "gesture": "gesture",
    "cross": "cross",
}

_ATTR_INT_KEYS = {
    "crossing",
    "crossing_point",
    "critical_point",
    "exp_start_point",
    "num_lanes",
}


class PIEAnnotations:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        base = self.root / "annotations"
        self._annt_zip = base / "annotations.zip"
        self._attr_zip = base / "annotations_attributes.zip"
        self._obd_zip = base / "annotations_vehicle.zip"
        if not self._annt_zip.is_file():
            raise FileNotFoundError(
                f"{self._annt_zip} not found; expected a PIE checkout at {self.root}"
            )
        self._zips: dict[Path, zipfile.ZipFile] = {}

    def _zip(self, path: Path) -> zipfile.ZipFile:
        if path not in self._zips:
            self._zips[path] = zipfile.ZipFile(path)
        return self._zips[path]

    def close(self) -> None:
        for z in self._zips.values():
            z.close()
        self._zips.clear()

    def video_ids(self) -> list[str]:
        """Ids are "setNN/video_NNNN" — PIE video numbers repeat across sets."""
        out = []
        for name in self._zip(self._annt_zip).namelist():
            m = _MEMBER_RE.search(name)
            if m:
                out.append(f"{m.group(1)}/video_{m.group(2)}")
        return sorted(out)

    def load(self, video_id: str) -> VideoAnnotation:
        set_id, vid = video_id.split("/")
        video = self._load_boxes(set_id, vid)
        self._load_attributes(set_id, vid, video)
        self._load_obd(set_id, vid, video)
        return video

    def _read(self, archive: Path, member: str) -> bytes | None:
        z = self._zip(archive)
        try:
            return z.read(member)
        except KeyError:
            return None

    def _load_boxes(self, set_id: str, vid: str) -> VideoAnnotation:
        raw = self._read(self._annt_zip, f"annotations/{set_id}/{vid}_annt.xml")
        if raw is None:
            raise FileNotFoundError(f"no PIE annotation for {set_id}/{vid}")
        tree = ET.fromstring(raw)
        task = tree.find("./meta/task")
        assert task is not None
        video = VideoAnnotation(
            video_id=f"{set_id}/{vid}",
            dataset=DATASET,
            width=int(task.findtext("./original_size/width", "1920")),
            height=int(task.findtext("./original_size/height", "1080")),
            num_frames=int(task.findtext("./size", "0")),
        )

        for track in tree.findall("./track"):
            boxes = track.findall("./box")
            if not boxes:
                continue
            label = track.get("label", "")
            obj_id = _attr_text(boxes[0], "id")
            if obj_id is None:
                continue
            if label == "pedestrian":
                video.pedestrians[obj_id] = _parse_ped(obj_id, boxes)
            else:
                video.traffic[obj_id] = _parse_traffic(obj_id, label, boxes)
        return video

    def _load_attributes(self, set_id: str, vid: str, video: VideoAnnotation) -> None:
        raw = self._read(
            self._attr_zip, f"annotations_attributes/{set_id}/{vid}_attributes.xml"
        )
        if raw is None:
            return
        for ped in ET.fromstring(raw).findall("./pedestrian"):
            track = video.pedestrians.get(ped.get("id") or "")
            if track is None:
                continue
            for key, value in ped.attrib.items():
                if key == "id":
                    continue
                if key == "intention_prob":
                    track.attributes[key] = float(value)
                elif key in _ATTR_INT_KEYS:
                    track.attributes[key] = int(value)
                else:
                    track.attributes[key] = value

    def _load_obd(self, set_id: str, vid: str, video: VideoAnnotation) -> None:
        raw = self._read(self._obd_zip, f"annotations_vehicle/{set_id}/{vid}_obd.xml")
        if raw is None:
            return
        for frame in ET.fromstring(raw).findall("./frame"):
            fid = int(frame.get("id", "0"))
            video.ego[fid] = {
                k: float(v) for k, v in frame.attrib.items() if k != "id"
            }


def _parse_ped(ped_id: str, boxes: list[ET.Element]) -> PedTrack:
    ped = PedTrack(ped_id=ped_id)
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
        ped.occlusion.append(
            {"none": 0, "part": 1, "full": 2}.get(_attr_text(box, "occlusion") or "none", 0)
        )
        for column, field_name in _BEHAVIOR_COLUMNS.items():
            getattr(ped, field_name).append(_attr_text(box, column) or "")
    return ped


def _parse_traffic(obj_id: str, label: str, boxes: list[ET.Element]) -> TrafficObject:
    obj = TrafficObject(
        obj_id=obj_id, obj_class=label, obj_type=_attr_text(boxes[0], "type")
    )
    for box in boxes:
        if box.get("outside") == "1":
            continue
        obj.frames.append(int(box.get("frame", "0")))
        obj.bbox.append(
            (
                float(box.get("xtl", "0")),
                float(box.get("ytl", "0")),
                float(box.get("xbr", "0")),
                float(box.get("ybr", "0")),
            )
        )
        if label == "traffic_light":
            obj.state.append(_attr_text(box, "state") or "__undefined__")
    return obj


def _attr_text(box: ET.Element, name: str) -> str | None:
    node = box.find(f'./attribute[@name="{name}"]')
    return node.text if node is not None else None
