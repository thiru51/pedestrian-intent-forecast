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

import torch

from pif.config import load
from pif.data.dataset import PedestrianWindows, make_loader
from pif.data.build import build
from pif.models.fusion import build_model
from pif.train.losses import MultiTaskLoss
from pif.train.loop import train

from ._common import add_hardware_args, out_dir, resolve


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

    print(
        f"data: source={cfg.data.source} {len(train_ds)} train / {len(val_ds)} val "
        f"windows, positive rate {train_ds.positive_rate:.3f} / {val_ds.positive_rate:.3f}, "
        f"lead {cfg.data.lead} frames ({cfg.data.lead / 30:.2f}s)"
    )
    print(
        f"protocol: {args.repeats} seeds per arm, {cfg.train.epochs} epochs, "
        f"batch {cfg.batch_size}, identical data and schedule for both arms"
    )
    print()

    pos_weight = train_ds.pos_weight()
    runs: dict[str, list] = {"fusion": [], "pose_only": []}
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
            runs[arm].append(result)
            print(f"    best {result.best.line() if result.best else 'n/a'}")
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
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "val_positive_rate": val_ds.positive_rate,
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
        "arms": {},
    }

    for arm, results in runs.items():
        aps = [r.best.ap for r in results if r.best]
        aucs = [r.best.auc for r in results if r.best]
        f1s = [r.best.f1 for r in results if r.best]
        misses = [r.best.miss_rate for r in results if r.best]
        summary["arms"][arm] = {
            "params": results[0].params,
            "ap_mean": _mean(aps),
            "ap_std": _std(aps),
            "ap_all": [round(v, 4) for v in aps],
            "auc_mean": _mean(aucs),
            "f1_mean": _mean(f1s),
            "miss_rate_mean": _mean(misses),
            "mean_samples_per_sec": _mean([r.mean_samples_per_sec for r in results]),
            "peak_vram_gb": max(r.peak_vram_gb for r in results),
            "seconds_per_run": _mean([r.total_seconds for r in results]),
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
    print("=" * 72)
    print(f"{'arm':<12}{'params':>10}{'AP':>18}{'AUC':>10}{'F1':>10}{'miss':>10}")
    for arm in ("pose_only", "fusion"):
        a = summary["arms"][arm]
        print(
            f"{arm:<12}{a['params']:>10,}"
            f"{a['ap_mean']:>12.4f} +-{a['ap_std']:.4f}"
            f"{a['auc_mean']:>10.4f}{a['f1_mean']:>10.4f}{a['miss_rate_mean']:>10.4f}"
        )
    print("-" * 72)
    print(
        f"fused - pose_only: dAP {summary['delta_ap']:+.4f}  "
        f"dAUC {summary['delta_auc']:+.4f}  "
        f"({summary['relative_ap_gain'] * 100:+.1f}% relative AP)"
    )
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
    return 0


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
