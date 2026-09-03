"""One entry point that turns a DataConfig into train/val window lists.

Handles every source the same way so nothing above this layer needs to know
which one it got:

  synthetic  generated in memory, split by synthetic track id
  jaad       parsed from external/jaad XML, split by JAAD's own split_ids files
  pie        parsed from the zips in external/pie/annotations, split by set
  carla      recordings from scripts/collect_carla.py, split by episode

Pose is the one thing that differs. JAAD and PIE ship annotations, not frames,
so unless a pose cache built from the video files is provided the pose channel
is zeros and the fused model is running on context and gesture alone. CARLA
records its own frames, so a pose cache for it is built by
scripts/extract_pose.py --dataset carla. The loaders say so out loud rather
than quietly training on a dead branch.

Every split here is by video or episode, never by window. Consecutive windows
from one track share 12 of their 16 frames, so a window-level split puts near
copies of the same sample on both sides and every number that comes out is
inflated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .carla import CARLARecordings, mask_unlabelled_gestures
from .jaad import JAADAnnotations
from .pie import PIEAnnotations
from .synthetic import make_windows, split_windows
from .windows import Window, video_windows


class MissingAnnotations(FileNotFoundError):
    pass


def _load_pose_cache(path: str | Path | None) -> dict[str, dict[str, np.ndarray]]:
    """Optional .npz of per-(video, pedestrian) pose feature arrays, produced by
    scripts/extract_pose.py once the video clips exist. Keys are
    "<video_id>|<ped_id>"."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    cache: dict[str, dict[str, np.ndarray]] = {}
    with np.load(p) as data:
        for key in data.files:
            video_id, _, ped_id = key.partition("|")
            cache.setdefault(video_id, {})[ped_id] = data[key]
    return cache


def build_jaad(cfg, pose_cache: str | Path | None = None) -> tuple[list[Window], list[Window]]:
    root = Path(cfg.root or "external/jaad")
    if not (root / "annotations").is_dir():
        raise MissingAnnotations(
            f"no JAAD annotations under {root}. See the README section "
            "'Getting the data' — the XML lives in the JAAD repo, the video "
            "clips are a separate 3.1 GB download."
        )
    db = JAADAnnotations(root)
    poses = _load_pose_cache(pose_cache)

    try:
        train_ids = db.split("train", cfg.split_subset)
        val_ids = db.split("val", cfg.split_subset)
    except FileNotFoundError:
        # split_ids/ is part of the annotation repo, but be tolerant: fall back
        # to a deterministic 75/25 by video number.
        ids = db.video_ids()
        cut = int(len(ids) * 0.75)
        train_ids, val_ids = ids[:cut], ids[cut:]

    if cfg.max_videos:
        train_ids = train_ids[: cfg.max_videos]
        val_ids = val_ids[: max(cfg.max_videos // 4, 1)]

    return (
        _windows_for(db, train_ids, cfg, poses),
        _windows_for(db, val_ids, cfg, poses),
    )


def build_pie(cfg, pose_cache: str | Path | None = None) -> tuple[list[Window], list[Window]]:
    root = Path(cfg.root or "external/pie")
    if not (root / "annotations" / "annotations.zip").is_file():
        raise MissingAnnotations(
            f"no PIE annotations under {root}. See the README section "
            "'Getting the data'."
        )
    db = PIEAnnotations(root)
    poses = _load_pose_cache(pose_cache)

    ids = db.video_ids()
    # PIE's published protocol splits by set: sets 1-4 train, 5-6 test. Set 5
    # and 6 are the downtown routes, so this is a route-level split and not a
    # random one.
    train_ids = [v for v in ids if v.split("/")[0] in ("set01", "set02", "set03", "set04")]
    val_ids = [v for v in ids if v.split("/")[0] in ("set05", "set06")]
    if not val_ids:
        cut = int(len(ids) * 0.75)
        train_ids, val_ids = ids[:cut], ids[cut:]

    if cfg.max_videos:
        train_ids = train_ids[: cfg.max_videos]
        val_ids = val_ids[: max(cfg.max_videos // 4, 1)]

    return (
        _windows_for(db, train_ids, cfg, poses),
        _windows_for(db, val_ids, cfg, poses),
    )


def build_carla(cfg, pose_cache: str | Path | None = None) -> tuple[list[Window], list[Window]]:
    """CARLA episodes, split by episode.

    An episode is one ego run with its own walkers, so it is the natural unit:
    two windows from the same episode share the same pedestrian, the same
    lighting and the same stretch of road, and separating them would leak.

    The split is a deterministic interleave rather than "first 75% of the
    sorted ids". Episodes are written in scenario order (straight, hesitate,
    occluded, no_cross, repeating), so a contiguous cut would hand the
    validation set a different scenario mix from the training set and the
    comparison would be measuring that instead of the model.
    """
    root = Path(cfg.root or "data/carla")
    if not root.is_dir() or not list(root.glob("*.json")):
        raise MissingAnnotations(
            f"no CARLA recordings under {root}. Collect some first: see the "
            "README section 'Getting CARLA data'."
        )
    db = CARLARecordings(root)
    poses = _load_pose_cache(pose_cache)

    ids = db.video_ids()
    if cfg.max_videos:
        ids = ids[: cfg.max_videos]

    every = max(int(round(1.0 / max(cfg.val_fraction, 1e-6))), 2)
    val_ids = [v for i, v in enumerate(ids) if i % every == every - 1]
    train_ids = [v for v in ids if v not in set(val_ids)]

    return (
        _windows_for(db, train_ids, cfg, poses, carla=True),
        _windows_for(db, val_ids, cfg, poses, carla=True),
    )


def _windows_for(db, video_ids, cfg, poses, carla: bool = False) -> list[Window]:
    out: list[Window] = []
    for vid in video_ids:
        try:
            video = db.load(vid)
        except FileNotFoundError:
            continue
        windows = video_windows(
            video,
            window=cfg.window,
            lead=cfg.lead,
            stride=cfg.stride,
            horizon=cfg.horizon,
            pose_by_ped=poses.get(vid),
        )
        if carla:
            # CARLA annotates no gestures, gaze or actions. The shared encoder
            # falls back to a default class for a missing column, so without
            # this the gesture head would be trained against three labels the
            # simulator never produced.
            for w in windows:
                mask_unlabelled_gestures(w.gesture)
        out.extend(windows)
    return out


def build(cfg, pose_cache: str | Path | None = None) -> tuple[list[Window], list[Window]]:
    source = cfg.source.lower()
    if source == "synthetic":
        windows = make_windows(
            n_tracks=cfg.n_tracks,
            seed=cfg.seed,
            window=cfg.window,
            lead=cfg.lead,
            horizon=cfg.horizon,
            stride=cfg.stride,
            track_length=cfg.track_length,
            positive_fraction=cfg.positive_fraction,
        )
        return split_windows(windows, cfg.val_fraction)
    if source == "jaad":
        return build_jaad(cfg, pose_cache)
    if source == "pie":
        return build_pie(cfg, pose_cache)
    if source == "carla":
        return build_carla(cfg, pose_cache)
    raise ValueError(f"unknown data source '{cfg.source}'")
