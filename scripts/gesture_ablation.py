#!/usr/bin/env python
"""Does the gesture head help, measured against real gesture labels?

This is the experiment the repository has not been able to run. CARLA supplies no
gesture annotations, so on synthetic data the gesture label groups are masked out and
the gesture head is never scored against anything. JAAD does annotate gestures
(`hand_gesture`, plus `nod` and `reaction`), so the head can finally be tested.

The video clips are a separate 2.9 GB download and are not needed here: without them
there is no pose channel, so **both arms run without pose**. That makes this a clean
isolation of the gesture head rather than a pose-versus-fusion comparison -- the two
arms differ in `use_gesture` and in nothing else.

    context           scene context + trajectory, no gesture head
    context+gesture   the same, plus the gesture head feeding the intent tokens

Reporting note: do not compare these numbers to the CARLA ones in RESULTS.md. Different
dataset, different label definition, and no pose in either arm here.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pif.config import load
from pif.data.build import build
from pif.data.dataset import PedestrianWindows, make_loader
from pif.metrics import intent_metrics
from pif.models.fusion import IntentForecaster
from pif.train.losses import MultiTaskLoss
from pif.train.loop import train

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_hardware_args, resolve
from scripts.run_ablation import val_scores, occlusion_mask, subset_metrics, _measured

ARMS = {"context": False, "context+gesture": True}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/fusion.yaml")
    p.add_argument("--source", default="jaad")
    p.add_argument("--data-root", default="")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3, help="seeds per arm")
    p.add_argument("--out", default="artifacts/gesture_ablation.json")
    add_hardware_args(p)
    args = p.parse_args()

    cfg = load(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    cfg.data.source = args.source
    if args.data_root:
        cfg.data.root = args.data_root

    info, cfg = resolve(args, cfg)
    print(info.summary())

    # No pose cache on purpose: the clips are not required for this comparison.
    train_windows, val_windows = build(cfg.data, None)
    train_ds, val_ds = PedestrianWindows(train_windows), PedestrianWindows(val_windows)
    workers = None if cfg.num_workers < 0 else cfg.num_workers
    train_loader = make_loader(train_ds, cfg.batch_size, info, True, workers)
    val_loader = make_loader(val_ds, cfg.batch_size, info, False, workers)

    assert not ({w.video_id for w in train_windows} & {w.video_id for w in val_windows}), \
        "train and val share a video: the split leaked"
    occluded = occlusion_mask(val_windows)

    print(f"data: {cfg.data.source} {len(train_ds)} train / {len(val_ds)} val windows, "
          f"positive rate {train_ds.positive_rate:.3f} / {val_ds.positive_rate:.3f}")
    print(f"split: by video, {len({w.video_id for w in train_windows})} train / "
          f"{len({w.video_id for w in val_windows})} val, no overlap")
    print(f"protocol: {args.repeats} seeds per arm, {cfg.train.epochs} epochs, "
          f"identical data and schedule. Both arms run WITHOUT pose.\n")

    pos_weight = train_ds.pos_weight()
    finals = {a: [] for a in ARMS}
    t0 = time.perf_counter()

    for seed in range(args.repeats):
        for arm, use_gesture in ARMS.items():
            torch.manual_seed(seed)
            if info.is_cuda:
                torch.cuda.manual_seed_all(seed)
            mcfg = replace(cfg.model, use_gesture=use_gesture, use_context=True)
            model = IntentForecaster(mcfg)
            loss_fn = MultiTaskLoss(cfg.loss, pos_weight=pos_weight)
            print(f"--- {arm}  seed {seed}  ({model.num_parameters():,} params)")
            train(model, train_loader, val_loader, info, cfg.train, loss_fn, log=lambda s: None)
            scores, labels = val_scores(model, val_loader, info)
            m = intent_metrics(scores, labels)
            finals[arm].append({"seed": seed, "overall": _measured(m.as_dict()),
                                "occluded": subset_metrics(scores, labels, occluded),
                                "clear": subset_metrics(scores, labels, ~occluded)})
            print(f"    AP={m.ap:.4f} AUC={m.auc:.4f} F1={m.f1:.4f} "
                  f"prec={m.precision:.4f} recall={m.recall:.4f} miss={m.miss_rate:.4f}")

    def agg(arm, key="ap"):
        v = [f["overall"][key] for f in finals[arm]]
        return float(np.mean(v)), float(np.std(v)), v

    print(f"\n{'arm':18s} {'AP':>18s} {'AUC':>8s}")
    for arm in ARMS:
        ap_m, ap_s, _ = agg(arm); auc_m, _, _ = agg(arm, "auc")
        print(f"{arm:18s} {ap_m:11.4f} +-{ap_s:5.4f} {auc_m:8.4f}")
    d_ap = agg("context+gesture")[0] - agg("context")[0]
    d_auc = agg("context+gesture", "auc")[0] - agg("context", "auc")[0]
    print(f"\ngesture - context: dAP {d_ap:+.4f}  dAUC {d_auc:+.4f}")

    out = {"source": cfg.data.source, "device": str(info.device),
           "epochs": cfg.train.epochs, "seeds": args.repeats,
           "pose": "absent -- both arms; JAAD clips not downloaded",
           "n_train_windows": len(train_ds), "n_val_windows": len(val_ds),
           "val_positive_rate": val_ds.positive_rate,
           "arms": {a: {"ap_mean": agg(a)[0], "ap_std": agg(a)[1], "ap_all": agg(a)[2],
                        "auc_mean": agg(a, "auc")[0], "runs": finals[a]} for a in ARMS},
           "delta_ap": d_ap, "delta_auc": d_auc,
           "wall_clock_seconds": round(time.perf_counter() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
