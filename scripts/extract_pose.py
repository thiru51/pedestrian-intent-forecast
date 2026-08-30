"""Run pose estimation over the JAAD or PIE video clips and cache the features.

This is the one script in the repo that needs the video files. Everything else
runs off annotations or synthetic data. Once the clips are downloaded (see the
README section "Getting the data"), this walks every annotated pedestrian box,
crops it, runs MediaPipe Pose on the crop, encodes the result with
pif.features.pose.encode_pose, and writes one .npz keyed "<video_id>|<ped_id>".

    python -m scripts.extract_pose --dataset jaad \
        --clips external/jaad/JAAD_clips --out data/pose_jaad.npz

    python -m scripts.train --config configs/fusion.yaml \
        --source jaad --pose-cache data/pose_jaad.npz

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

from pif.data.jaad import JAADAnnotations
from pif.data.pie import PIEAnnotations
from pif.features.pose import POSE_DIM, PoseExtractor, empty_pose, encode_pose


def clip_path(clips_root: Path, dataset: str, video_id: str) -> Path:
    if dataset == "jaad":
        return clips_root / f"{video_id}.mp4"
    # PIE video ids are "setNN/video_NNNN" and the clips mirror that layout.
    set_id, vid = video_id.split("/")
    return clips_root / set_id / f"{vid}.mp4"


def extract_video(
    dataset: str, root: str, clips_root: str, video_id: str, complexity: int
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

    # frame number -> list of (ped_id, row index, box), so the video is read
    # once front to back instead of once per pedestrian.
    wanted: dict[int, list] = {}
    out = {
        t.ped_id: np.zeros((len(t.frames), POSE_DIM), dtype=np.float32) for t in tracks
    }
    for t in tracks:
        for i, f in enumerate(t.frames):
            wanted.setdefault(int(f), []).append((t.ped_id, i, t.bbox[i]))

    extractor = PoseExtractor(model_complexity=complexity)
    empty = encode_pose(empty_pose())

    cap = cv2.VideoCapture(str(path))
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for ped_id, i, (x1, y1, x2, y2) in wanted.get(frame_no, ()):
                h, w = frame.shape[:2]
                # A little padding: MediaPipe does better with some margin than
                # with a box cropped exactly to the silhouette.
                pad_x = 0.1 * (x2 - x1)
                pad_y = 0.05 * (y2 - y1)
                cx1 = max(int(x1 - pad_x), 0)
                cy1 = max(int(y1 - pad_y), 0)
                cx2 = min(int(x2 + pad_x), w)
                cy2 = min(int(y2 + pad_y), h)
                if cx2 - cx1 < 12 or cy2 - cy1 < 24:
                    out[ped_id][i] = empty  # too small to be worth trying
                    continue
                crop = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
                pose = extractor(crop)
                out[ped_id][i] = empty if pose is None else encode_pose(pose)
            frame_no += 1
    finally:
        cap.release()
        extractor.close()

    return {f"{video_id}|{ped}": arr for ped, arr in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="jaad", choices=("jaad", "pie"))
    parser.add_argument("--root", default="", help="default: external/<dataset>")
    parser.add_argument("--clips", required=True, help="directory holding the .mp4 files")
    parser.add_argument("--out", default="", help="default: data/pose_<dataset>.npz")
    parser.add_argument("--limit", type=int, default=0, help="videos to process, 0 = all")
    parser.add_argument("--workers", type=int, default=4, help="parallel video decoders")
    parser.add_argument(
        "--complexity", type=int, default=1, choices=(0, 1, 2),
        help="MediaPipe model_complexity: 0 fastest, 2 most accurate",
    )
    args = parser.parse_args()

    root = args.root or f"external/{args.dataset}"
    out_path = Path(args.out or f"data/pose_{args.dataset}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = JAADAnnotations(root) if args.dataset == "jaad" else PIEAnnotations(root)
    ids = db.video_ids()
    if args.limit:
        ids = ids[: args.limit]
    print(f"{args.dataset}: {len(ids)} videos, clips from {args.clips}")

    cache: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    extract_video, args.dataset, root, args.clips, vid, args.complexity
                ): vid
                for vid in ids
            }
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
                cache.update(
                    extract_video(args.dataset, root, args.clips, vid, args.complexity)
                )
            except FileNotFoundError as exc:
                print(f"  skip {vid}: {exc}")
                continue
            print(f"  [{n}/{len(ids)}] {vid}: {len(cache)} tracks cached")

    if not cache:
        print("nothing extracted - are the clips where --clips says they are?")
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
