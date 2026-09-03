"""The ablation: fused (pose + context + gesture) against pose-only.

Both models get the same windows, the same split, the same optimiser, the same
schedule, the same number of epochs and the same seed. The only difference is
which input branches exist. That is the whole point — if the fused model wins,
the win has to be attributable to the extra inputs and not to a bigger backbone
or a longer run.

The pose-only side is the family of methods this project is arguing against:
crossing prediction from body pose alone, as in Fang & Lopez (IV 2018).

Read the caveat the script prints at the end before quoting any of the numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import torch

from pif.config import load
from pif.data.dataset import PedestrianWindows, make_loader
from pif.data.build import build
from pif.features.context import CTX
from pif.metrics import intent_metrics
from pif.models.fusion import build_model
from pif.train.losses import MultiTaskLoss
from pif.train.loop import train

from ._common import add_hardware_args, out_dir, resolve

# Frames per second of each source, for turning a lead in frames back into
# seconds. CARLA is recorded at 20 Hz; JAAD and PIE are 30 fps.
SOURCE_FPS = {"carla": 20.0}
DEFAULT_FPS = 30.0


@torch.no_grad()
def val_scores(model, loader, info) -> tuple[np.ndarray, np.ndarray]:
    """Crossing probability and label for every validation window, in order.

    The loader does not shuffle and does not drop the last batch, so the order
    matches the window list the split was built from. That is what lets a
    subset - the occluded windows, say - be sliced out afterwards.
    """
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(info.device, non_blocking=True)
        with torch.autocast(
            device_type=info.device.type,
            dtype=info.amp_dtype or torch.float32,
            enabled=info.amp_dtype is not None,
        ):
            out = model(batch.pose, batch.context, batch.motion)
        scores.append(torch.sigmoid(out.intent_logit.float()).cpu().numpy())
        labels.append(batch.intent.cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels)


def occlusion_mask(windows) -> np.ndarray:
    """True for windows where the pedestrian was hidden at some point.

    The context encoder writes occlusion as 0, 0.5 or 1 for none, partial and
    full, so anything above zero means the observation was not clean. This is
    the subset the project cares most about: a pedestrian stepping out from
    behind a parked car is its documented failure mode.
    """
    return np.array(
        [float(w.context[:, CTX["occlusion"]].max()) > 0.0 for w in windows], dtype=bool
    )


def _measured(metrics: dict) -> dict:
    """Drop the fields that were not computed here.

    val_scores only runs the intent head, so ADE, FDE and gesture accuracy come
    back as NaN. Carrying them would put the literal NaN into the JSON, which
    strict parsers reject, and would look like a measurement that was taken and
    failed rather than one that was never taken.
    """
    return {k: v for k, v in metrics.items() if not (isinstance(v, float) and v != v)}


def subset_metrics(scores, labels, mask) -> dict | None:
    """Metrics on a slice, or None when the slice has only one class in it.

    Average precision and ROC AUC are undefined without both classes present,
    and reporting them from a single-class slice would be inventing a number.
    """
    if mask.sum() < 2:
        return None
    sub_labels = labels[mask]
    if len(set(sub_labels.tolist())) < 2:
        return None
    return _measured(intent_metrics(scores[mask], sub_labels).as_dict())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--source", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--pose-cache", default="")
    parser.add_argument("--repeats", type=int, default=3, help="seeds per arm")
    add_hardware_args(parser)
    args = parser.parse_args()

    cfg = load(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.source:
        cfg.data.source = args.source
    if args.data_root:
        cfg.data.root = args.data_root

    info, cfg = resolve(args, cfg)
    print(info.summary())

    train_windows, val_windows = build(cfg.data, args.pose_cache or None)
    train_ds = PedestrianWindows(train_windows)
    val_ds = PedestrianWindows(val_windows)
    workers = None if cfg.num_workers < 0 else cfg.num_workers
    train_loader = make_loader(train_ds, cfg.batch_size, info, True, workers)
    val_loader = make_loader(val_ds, cfg.batch_size, info, False, workers)

    fps = SOURCE_FPS.get(cfg.data.source.lower(), DEFAULT_FPS)
    occluded = occlusion_mask(val_windows)
    train_episodes = len({w.video_id for w in train_windows})
    val_episodes = len({w.video_id for w in val_windows})

    # A window-level split would put near copies of the same sample on both
    # sides. Cheap to assert, and the whole comparison is void without it.
    assert not (
        {w.video_id for w in train_windows} & {w.video_id for w in val_windows}
    ), "train and val share an episode: the split leaked"

    print(
        f"data: source={cfg.data.source} {len(train_ds)} train / {len(val_ds)} val "
        f"windows, positive rate {train_ds.positive_rate:.3f} / {val_ds.positive_rate:.3f}, "
        f"lead {cfg.data.lead} frames ({cfg.data.lead / fps:.2f}s at {fps:g} Hz)"
    )
    print(
        f"split: by episode, {train_episodes} train / {val_episodes} val, no overlap"
    )
    print(
        f"val subsets: {int(occluded.sum())} windows where the pedestrian was "
        f"occluded at some point, {int((~occluded).sum())} clear"
    )
    print(
        f"protocol: {args.repeats} seeds per arm, {cfg.train.epochs} epochs, "
        f"batch {cfg.batch_size}, identical data and schedule for both arms. "
        f"Metrics are from the final epoch, with no best-epoch selection."
    )
    print()

    pos_weight = train_ds.pos_weight()
    runs: dict[str, list] = {"fusion": [], "pose_only": []}
    # Final-epoch metrics, kept beside the RunResults rather than inside them so
    # nothing in pif.train has to know this script slices subsets.
    finals: dict[str, list[dict]] = {"fusion": [], "pose_only": []}
    started = time.perf_counter()

    for seed in range(args.repeats):
        for arm in ("fusion", "pose_only"):
            torch.manual_seed(seed)
            if info.is_cuda:
                torch.cuda.manual_seed_all(seed)
            model = build_model(arm, replace(cfg.model))
            loss_fn = MultiTaskLoss(cfg.loss, pos_weight=pos_weight)
            print(f"--- {arm}  seed {seed}  ({model.num_parameters():,} params)")
            result = train(
                model,
                train_loader,
                val_loader,
                info,
                cfg.train,
                loss_fn,
                log=lambda s: None,  # per-epoch lines would drown the comparison
            )
            scores, labels = val_scores(model, val_loader, info)
            final = intent_metrics(scores, labels)
            finals[arm].append(
                {
                    "seed": seed,
                    "overall": _measured(final.as_dict()),
                    "occluded": subset_metrics(scores, labels, occluded),
                    "clear": subset_metrics(scores, labels, ~occluded),
                }
            )
            runs[arm].append(result)
            print(
                f"    final n={final.n} pos={final.positive_rate:.3f} "
                f"AP={final.ap:.4f} AUC={final.auc:.4f} F1={final.f1:.4f} "
                f"prec={final.precision:.4f} recall={final.recall:.4f} "
                f"miss={final.miss_rate:.4f}"
            )
            print(
                f"    {result.total_seconds:.1f}s  "
                f"{result.mean_samples_per_sec:,.0f} samples/s  "
                f"peak VRAM {result.peak_vram_gb:.3f} GiB"
            )

    summary = {
        "source": cfg.data.source,
        "synthetic": cfg.data.source == "synthetic",
        "device": str(info.device),
        "amp": "off" if info.amp_dtype is None else str(info.amp_dtype).replace("torch.", ""),
        "epochs": cfg.train.epochs,
        "batch_size": cfg.batch_size,
        "seeds": args.repeats,
        "lead_frames": cfg.data.lead,
        "carla": cfg.data.source.lower() == "carla",
        "fps": fps,
        "lead_seconds": round(cfg.data.lead / fps, 3),
        "split": "by episode",
        "n_train_episodes": train_episodes,
        "n_val_episodes": val_episodes,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_val_occluded_windows": int(occluded.sum()),
        "n_val_clear_windows": int((~occluded).sum()),
        "val_positive_rate": val_ds.positive_rate,
        "metric_source": "final epoch, no best-epoch selection",
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
        "arms": {},
    }

    for arm, results in runs.items():
        rows = finals[arm]
        overall = [r["overall"] for r in rows]
        summary["arms"][arm] = {
            "params": results[0].params,
            "ap_mean": _mean([m["ap"] for m in overall]),
            "ap_std": _std([m["ap"] for m in overall]),
            "ap_all": [round(m["ap"], 4) for m in overall],
            "auc_mean": _mean([m["auc"] for m in overall]),
            "accuracy_mean": _mean([m["accuracy"] for m in overall]),
            "precision_mean": _mean([m["precision"] for m in overall]),
            "recall_mean": _mean([m["recall"] for m in overall]),
            "f1_mean": _mean([m["f1"] for m in overall]),
            "miss_rate_mean": _mean([m["miss_rate"] for m in overall]),
            "false_alarm_rate_mean": _mean([m["false_alarm_rate"] for m in overall]),
            "occluded": _subset_summary(rows, "occluded"),
            "clear": _subset_summary(rows, "clear"),
            "mean_samples_per_sec": _mean([r.mean_samples_per_sec for r in results]),
            "peak_vram_gb": max(r.peak_vram_gb for r in results),
            "seconds_per_run": _mean([r.total_seconds for r in results]),
            "per_seed": rows,
            "runs": [r.as_dict() for r in results],
        }

    fused = summary["arms"]["fusion"]
    pose = summary["arms"]["pose_only"]
    summary["delta_ap"] = round(fused["ap_mean"] - pose["ap_mean"], 4)
    summary["delta_auc"] = round(fused["auc_mean"] - pose["auc_mean"], 4)
    summary["relative_ap_gain"] = (
        round((fused["ap_mean"] - pose["ap_mean"]) / max(pose["ap_mean"], 1e-9), 4)
    )

    art = out_dir(cfg)
    path = art / "ablation.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print("=" * 78)
    print(
        f"{'arm':<11}{'params':>9}{'AP':>17}{'AUC':>8}{'prec':>8}"
        f"{'recall':>8}{'F1':>8}{'miss':>8}"
    )
    for arm in ("pose_only", "fusion"):
        a = summary["arms"][arm]
        print(
            f"{arm:<11}{a['params']:>9,}"
            f"{a['ap_mean']:>11.4f} +-{a['ap_std']:.4f}"
            f"{a['auc_mean']:>8.4f}{a['precision_mean']:>8.4f}"
            f"{a['recall_mean']:>8.4f}{a['f1_mean']:>8.4f}"
            f"{a['miss_rate_mean']:>8.4f}"
        )
    print("-" * 78)
    print(
        f"fused - pose_only: dAP {summary['delta_ap']:+.4f}  "
        f"dAUC {summary['delta_auc']:+.4f}  "
        f"({summary['relative_ap_gain'] * 100:+.1f}% relative AP)"
    )

    for name, title in (("occluded", "occluded subset"), ("clear", "clear subset")):
        rows = {arm: summary["arms"][arm][name] for arm in ("pose_only", "fusion")}
        if not all(rows.values()):
            print()
            print(
                f"{title}: not reported. It has fewer than two windows, or only "
                "one class in it, and average precision on a single-class slice "
                "is not a number worth printing."
            )
            continue
        print()
        print(
            f"{title}: n={rows['fusion']['n']} "
            f"positive rate {rows['fusion']['positive_rate']:.3f}"
        )
        print(f"{'  arm':<13}{'AP':>17}{'AUC':>8}{'prec':>8}{'recall':>8}{'miss':>8}")
        for arm in ("pose_only", "fusion"):
            r = rows[arm]
            print(
                f"  {arm:<11}{r['ap_mean']:>11.4f} +-{r['ap_std']:.4f}"
                f"{r['auc_mean']:>8.4f}{r['precision_mean']:>8.4f}"
                f"{r['recall_mean']:>8.4f}{r['miss_rate_mean']:>8.4f}"
            )
        print(
            f"  fused - pose_only: dAP "
            f"{rows['fusion']['ap_mean'] - rows['pose_only']['ap_mean']:+.4f}"
        )

    print()
    print(f"total wall clock {summary['wall_clock_seconds']:.1f}s")
    print(f"wrote {path}")

    if summary["synthetic"]:
        print()
        print("CAVEAT, and it is not a small one: this ran on SYNTHETIC data.")
        print("The generator was written with context cues that pose does not")
        print("carry, so a gap here shows the fusion path is wired up and can")
        print("use those channels. It says nothing about real pedestrians. The")
        print("same command with --source jaad and a pose cache is the run that")
        print("would mean something, and it needs the video clips.")
    elif summary["carla"]:
        print()
        print("CAVEAT. These numbers are on SIMULATED pedestrians, from CARLA.")
        print("The labels are exact, which is the point, but the people are not")
        print("real: they walk on a navmesh, they are lit by a game engine, and")
        print("they never do any of the things a gesture head exists to catch.")
        print("This is a measurement of the pipeline on synthetic video with")
        print("ground-truth labels. It is NOT accuracy on real pedestrians, and")
        print("it should never be quoted as one. The run that would mean that")
        print("is --source jaad with a pose cache, and it needs the clips.")
    return 0


def _subset_summary(rows, key: str) -> dict | None:
    present = [r[key] for r in rows if r[key] is not None]
    if not present:
        return None
    return {
        "n": present[0]["n"],
        "positive_rate": round(present[0]["positive_rate"], 4),
        "ap_mean": _mean([m["ap"] for m in present]),
        "ap_std": _std([m["ap"] for m in present]),
        "auc_mean": _mean([m["auc"] for m in present]),
        "precision_mean": _mean([m["precision"] for m in present]),
        "recall_mean": _mean([m["recall"] for m in present]),
        "f1_mean": _mean([m["f1"] for m in present]),
        "miss_rate_mean": _mean([m["miss_rate"] for m in present]),
    }


def _mean(values) -> float:
    values = [v for v in values if v == v]  # drop NaN
    return round(sum(values) / len(values), 4) if values else float("nan")


def _std(values) -> float:
    values = [v for v in values if v == v]
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return round((sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5, 4)


if __name__ == "__main__":
    raise SystemExit(main())
