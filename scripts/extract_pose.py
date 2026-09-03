"""Run pose estimation over recorded frames and cache the features.

This is the one script in the repo that needs pixels. Everything else runs off
annotations or synthetic data. It walks every annotated pedestrian box, crops
it, runs MediaPipe Pose on the crop, encodes the result with
pif.features.pose.encode_pose, and writes one .npz keyed "<video_id>|<ped_id>".

Three sources. JAAD and PIE read .mp4 clips, which are a separate gated
download. CARLA reads the frames that scripts/collect_carla.py saved beside
each episode, so it needs nothing but a collection run.

    python -m scripts.extract_pose --dataset carla --root data/carla \
        --out data/pose_carla.npz --workers 5

    python -m scripts.extract_pose --dataset jaad \
        --clips external/jaad/JAAD_clips --out data/pose_jaad.npz

    python -m scripts.train --config configs/carla.yaml \
        --pose-cache data/pose_carla.npz

Running the same MediaPipe extractor over CARLA frames is a deliberate choice
over reading the simulator's own skeleton. CARLA knows every walker's bone
transforms exactly, and projecting those would give flawless pose labels - but
the deployed model gets its pose from MediaPipe on a real camera, and a model
trained on flawless pose and run on noisy pose is a sim-to-real trap that would
not show up until the day it mattered. Whatever MediaPipe gets wrong on a
CARLA render, it gets wrong at training time too.

Cost: MediaPipe at model_complexity=1 runs roughly real time per crop on one
CPU core, and JAAD is 346 clips of ~10 s at 30 fps with several pedestrians in
frame. Budget hours, not minutes, and use --workers.

HRNet-W32 (Sun et al., CVPR 2019) is the heavier alternative and is what I
would use for a real run: same COCO-17 output, markedly better on small and
partly occluded pedestrians, but it wants the GPU. Swapping it in means
replacing the extractor here and nothing else downstream, because both write
the same PoseFrame.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pif.data.carla import CARLARecordings
from pif.data.jaad import JAADAnnotations
from pif.data.pie import PIEAnnotations
from pif.features.pose import POSE_DIM, PoseExtractor, empty_pose, encode_pose


def clip_path(clips_root: Path, dataset: str, video_id: str) -> Path:
    if dataset == "jaad":
        return clips_root / f"{video_id}.mp4"
    # PIE video ids are "setNN/video_NNNN" and the clips mirror that layout.
    set_id, vid = video_id.split("/")
    return clips_root / set_id / f"{vid}.mp4"


def pose_for_boxes(frame_rgb, items, out, extractor, empty) -> None:
    """Crop each listed box out of one RGB frame and write its pose encoding.

    `items` is a list of (ped_id, row index, box) for this frame. Shared by the
    clip readers and the CARLA reader so the crop geometry - the padding, and
    the size below which MediaPipe is not worth calling - is identical for all
    three sources.
    """
    h, w = frame_rgb.shape[:2]
    for ped_id, i, (x1, y1, x2, y2) in items:
        # A little padding: MediaPipe does better with some margin than with a
        # box cropped exactly to the silhouette.
        pad_x = 0.1 * (x2 - x1)
        pad_y = 0.05 * (y2 - y1)
        cx1 = max(int(x1 - pad_x), 0)
        cy1 = max(int(y1 - pad_y), 0)
        cx2 = min(int(x2 + pad_x), w)
        cy2 = min(int(y2 + pad_y), h)
        if cx2 - cx1 < 12 or cy2 - cy1 < 24:
            out[ped_id][i] = empty  # too small to be worth trying
            continue
        pose = extractor(frame_rgb[cy1:cy2, cx1:cx2])
        out[ped_id][i] = empty if pose is None else encode_pose(pose)


def _wanted_boxes(tracks):
    """frame number -> [(ped_id, row index, box)], so a recording is read once
    front to back instead of once per pedestrian."""
    wanted: dict[int, list] = {}
    for t in tracks:
        for i, f in enumerate(t.frames):
            wanted.setdefault(int(f), []).append((t.ped_id, i, t.bbox[i]))
    return wanted


def extract_carla(
    root: str, video_id: str, complexity: int, model_path: str = ""
) -> dict[str, np.ndarray]:
    """One CARLA episode. Frames come from the .npz the collector wrote."""
    db = CARLARecordings(root)
    video = db.load(video_id)
    tracks = video.behavior_tracks()
    if not tracks:
        return {}
    path = db.frames_path(video_id)
    if path is None:
        raise FileNotFoundError(
            f"episode {video_id} has no frames on disk. Collect without "
            "--no-frames, or the frames were deleted after a previous run."
        )

    out = {
        t.ped_id: np.zeros((len(t.frames), POSE_DIM), dtype=np.float32) for t in tracks
    }
    wanted = _wanted_boxes(tracks)
    extractor = PoseExtractor(
        model_complexity=complexity, model_path=model_path or None
    )
    empty = encode_pose(empty_pose())
    try:
        with np.load(path) as z:
            frames = z["frames"]  # (T, H, W, 3) uint8, already RGB
            for frame_no, items in wanted.items():
                if frame_no < len(frames):
                    pose_for_boxes(frames[frame_no], items, out, extractor, empty)
    finally:
        extractor.close()
    return {f"{video_id}|{ped}": arr for ped, arr in out.items()}


def extract_video(
    dataset: str,
    root: str,
    clips_root: str,
    video_id: str,
    complexity: int,
    model_path: str = "",
) -> dict[str, np.ndarray]:
    """One video, one process. Returns {"<video>|<ped>": (T, POSE_DIM)}."""
    import cv2  # imported here so the module stays importable without opencv

    db = JAADAnnotations(root) if dataset == "jaad" else PIEAnnotations(root)
    video = db.load(video_id)
    tracks = video.behavior_tracks()
    if not tracks:
        return {}

    path = clip_path(Path(clips_root), dataset, video_id)
    if not path.is_file():
        raise FileNotFoundError(f"no clip at {path}")

    out = {
        t.ped_id: np.zeros((len(t.frames), POSE_DIM), dtype=np.float32) for t in tracks
    }
    wanted = _wanted_boxes(tracks)

    extractor = PoseExtractor(
        model_complexity=complexity, model_path=model_path or None
    )
    empty = encode_pose(empty_pose())

    cap = cv2.VideoCapture(str(path))
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            items = wanted.get(frame_no, ())
            if items:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_for_boxes(rgb, items, out, extractor, empty)
            frame_no += 1
    finally:
        cap.release()
        extractor.close()

    return {f"{video_id}|{ped}": arr for ped, arr in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="jaad", choices=("jaad", "pie", "carla"))
    parser.add_argument(
        "--root", default="", help="default: external/<dataset>, or data/carla"
    )
    parser.add_argument(
        "--clips",
        default="",
        help="directory holding the .mp4 files. jaad and pie only: CARLA "
        "keeps its frames beside the episodes under --root",
    )
    parser.add_argument("--out", default="", help="default: data/pose_<dataset>.npz")
    parser.add_argument("--limit", type=int, default=0, help="videos to process, 0 = all")
    parser.add_argument("--workers", type=int, default=4, help="parallel video decoders")
    parser.add_argument(
        "--complexity", type=int, default=1, choices=(0, 1, 2),
        help="MediaPipe model_complexity: 0 fastest, 2 most accurate",
    )
    parser.add_argument(
        "--pose-model",
        default="",
        help="path to a MediaPipe pose_landmarker .task bundle. Needed by "
        "mediapipe 1.x, which no longer ships the weights in the wheel. "
        "Defaults to data/pose_landmarker.task; see scripts/fetch_pose_model.sh",
    )
    parser.add_argument(
        "--append",
        default="",
        help="merge into this existing cache before writing. Lets a long CARLA "
        "collection be processed in batches, deleting each batch's frames as "
        "it goes, instead of needing the whole thing on disk at once.",
    )
    args = parser.parse_args()

    if args.dataset != "carla" and not args.clips:
        parser.error("--clips is required for jaad and pie")

    root = args.root or (
        "data/carla" if args.dataset == "carla" else f"external/{args.dataset}"
    )
    out_path = Path(args.out or f"data/pose_{args.dataset}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "carla":
        db = CARLARecordings(root)
    elif args.dataset == "jaad":
        db = JAADAnnotations(root)
    else:
        db = PIEAnnotations(root)
    ids = db.video_ids()
    if args.dataset == "carla":
        # An episode whose frames were already turned into pose and deleted has
        # nothing left to read, so skip it rather than fail the batch.
        ids = [v for v in ids if db.frames_path(v) is not None]
    if args.limit:
        ids = ids[: args.limit]
    where = root if args.dataset == "carla" else args.clips
    print(f"{args.dataset}: {len(ids)} recordings with frames, from {where}")

    cache: dict[str, np.ndarray] = {}
    if args.append:
        prior = Path(args.append)
        if prior.is_file():
            with np.load(prior) as data:
                cache.update({k: data[k] for k in data.files})
            print(f"  starting from {len(cache)} tracks already in {prior}")
    started = time.perf_counter()
    def submit(pool, vid):
        if args.dataset == "carla":
            return pool.submit(
                extract_carla, root, vid, args.complexity, args.pose_model
            )
        return pool.submit(
            extract_video,
            args.dataset,
            root,
            args.clips,
            vid,
            args.complexity,
            args.pose_model,
        )

    def run_one(vid):
        if args.dataset == "carla":
            return extract_carla(root, vid, args.complexity, args.pose_model)
        return extract_video(
            args.dataset, root, args.clips, vid, args.complexity, args.pose_model
        )

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {submit(pool, vid): vid for vid in ids}
            for n, (fut, vid) in enumerate(futures.items(), 1):
                try:
                    cache.update(fut.result())
                except FileNotFoundError as exc:
                    print(f"  skip {vid}: {exc}")
                    continue
                print(f"  [{n}/{len(ids)}] {vid}: {len(cache)} tracks cached")
    else:
        for n, vid in enumerate(ids, 1):
            try:
                cache.update(run_one(vid))
            except FileNotFoundError as exc:
                print(f"  skip {vid}: {exc}")
                continue
            print(f"  [{n}/{len(ids)}] {vid}: {len(cache)} tracks cached")

    if not cache:
        print("nothing extracted - are the frames where --root/--clips says?")
        return 1

    np.savez_compressed(out_path, **cache)
    elapsed = time.perf_counter() - started
    frames = sum(a.shape[0] for a in cache.values())
    print()
    print(f"wrote {out_path} ({out_path.stat().st_size / 2**20:.1f} MiB)")
    print(f"{len(cache)} tracks, {frames:,} pedestrian-frames, {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
