"""Parse the real JAAD or PIE annotations and print what came out.

No model, no torch, no video files. This is the script that proves the parsers
actually understand the released annotation schema, and it is the first thing
to run when you are wondering whether external/ is in the right shape.

    python -m scripts.parse_annotations --dataset jaad --limit 20
    python -m scripts.parse_annotations --dataset pie --limit 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pif.data.jaad import JAADAnnotations
from pif.data.pie import PIEAnnotations
from pif.data.windows import summarise, video_windows
from pif.features.gesture import CLASS_NAMES, GESTURE_GROUPS, IGNORE_INDEX


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="jaad", choices=("jaad", "pie"))
    parser.add_argument("--root", default="", help="default: external/<dataset>")
    parser.add_argument("--limit", type=int, default=20, help="videos to parse, 0 = all")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--lead", type=int, default=45)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--out", default="", help="write a JSON summary here")
    args = parser.parse_args()

    root = Path(args.root or f"external/{args.dataset}")
    db = JAADAnnotations(root) if args.dataset == "jaad" else PIEAnnotations(root)
    ids = db.video_ids()
    print(f"{args.dataset}: {len(ids)} annotated videos under {root}")
    if args.limit:
        ids = ids[: args.limit]
    print(f"parsing {len(ids)}")

    all_windows = []
    totals = Counter()
    gesture_counts = {g: Counter() for g in GESTURE_GROUPS}
    cross_values = Counter()
    per_video = []

    for vid in ids:
        video = db.load(vid)
        tracks = video.behavior_tracks()
        totals["videos"] += 1
        totals["pedestrians"] += len(video.pedestrians)
        totals["behavior_tracks"] += len(tracks)
        totals["traffic_objects"] += len(video.traffic)
        totals["ego_frames"] += len(video.ego)
        totals["scene_frames"] += len(video.scene)
        for t in tracks:
            cross_values.update(v for v in t.cross if v)
        windows = video_windows(
            video, window=args.window, lead=args.lead, stride=args.stride
        )
        all_windows.extend(windows)
        for w in windows:
            for gi, group in enumerate(GESTURE_GROUPS):
                for c in w.gesture[:, gi]:
                    if int(c) != IGNORE_INDEX:
                        gesture_counts[group][CLASS_NAMES[group][int(c)]] += 1
        per_video.append(
            {
                "video": vid,
                "frames": video.num_frames,
                "pedestrians": len(video.pedestrians),
                "behavior_tracks": len(tracks),
                "traffic": len(video.traffic),
                "windows": len(windows),
            }
        )

    print()
    print("totals:", dict(totals))
    print("per-frame cross values seen:", dict(cross_values.most_common()))
    print()
    stats = summarise(all_windows)
    print(f"windows: {json.dumps({k: round(v, 4) for k, v in stats.items()})}")
    print()
    for group in GESTURE_GROUPS:
        counts = gesture_counts[group]
        if counts:
            print(f"  {group:<14} {dict(counts.most_common())}")
        else:
            print(f"  {group:<14} (not annotated in this dataset)")

    if all_windows:
        w = all_windows[0]
        print()
        print(
            f"example window: {w.video_id} ped {w.ped_id} frames "
            f"{int(w.frames[0])}..{int(w.frames[-1])} intent={w.intent} "
            f"lead={w.lead} pose{w.pose.shape} context{w.context.shape}"
        )
        if w.intention_prob >= 0:
            print(f"  PIE human intention_prob for this track: {w.intention_prob:.3f}")
        print(
            "  pose is zeros: no video clips present, so no pose estimation. "
            "Everything else is parsed from the real XML."
        )

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "root": str(root),
                    "videos_parsed": len(ids),
                    "totals": dict(totals),
                    "cross_values": dict(cross_values),
                    "window_stats": stats,
                    "gesture_class_counts": {
                        g: dict(gesture_counts[g]) for g in GESTURE_GROUPS
                    },
                    "per_video": per_video,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
