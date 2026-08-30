"""Evaluate a saved checkpoint on a validation split.

    python -m scripts.evaluate --checkpoint artifacts/fusion.pt

Separate from training so a checkpoint can be re-scored against a different
data source (train on synthetic, look at what it does on parsed JAAD windows,
for instance) without retraining anything. Also prints the operating curve,
because a single F1 at threshold 0.5 hides the trade-off a planner actually
has to pick a point on: every missed crossing is a late brake, every false
alarm is a phantom stop.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from pif.data.dataset import PedestrianWindows, make_loader
from pif.data.build import build
from pif.metrics import binary_stats
from pif.models.fusion import ModelConfig, build_model
from pif.train.loop import evaluate

from ._common import add_hardware_args, load_config, out_dir, resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="artifacts/*.pt")
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--source", default="", help="override: synthetic | jaad | pie")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--pose-cache", default="")
    parser.add_argument("--out", default="", help="write metrics JSON here")
    add_hardware_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, "configs/fusion.yaml")
    if args.source:
        cfg.data.source = args.source
    if args.data_root:
        cfg.data.root = args.data_root
    info, cfg = resolve(args, cfg)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(
        payload.get("model_name", cfg.model_type), ModelConfig(**payload.get("config", {}))
    )
    model.load_state_dict(payload["state_dict"])
    print(info.summary())
    print(f"loaded {args.checkpoint}: {model.name}, {model.num_parameters():,} params")

    _, val_windows = build(cfg.data, args.pose_cache or None)
    if not val_windows:
        raise SystemExit(f"data source '{cfg.data.source}' produced no val windows")
    val_ds = PedestrianWindows(val_windows)
    workers = None if cfg.num_workers < 0 else cfg.num_workers
    val_loader = make_loader(val_ds, cfg.batch_size, info, False, workers)
    print(
        f"val: {len(val_ds)} windows, positive rate {val_ds.positive_rate:.3f}, "
        f"source {cfg.data.source}"
    )

    metrics = evaluate(model, val_loader, info)
    print()
    print(metrics.line())

    scores, labels = _scores(model, val_loader, info)
    print()
    print(f"{'thresh':>8}{'precision':>12}{'recall':>10}{'F1':>10}{'miss':>10}{'false alarm':>14}")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        s = binary_stats(scores, labels, t)
        print(
            f"{t:>8.1f}{s['precision']:>12.4f}{s['recall']:>10.4f}{s['f1']:>10.4f}"
            f"{s['miss_rate']:>10.4f}{s['false_alarm_rate']:>14.4f}"
        )

    payload_out = {
        "checkpoint": args.checkpoint,
        "source": cfg.data.source,
        "synthetic": cfg.data.source == "synthetic",
        "n_val_windows": len(val_ds),
        "val_positive_rate": val_ds.positive_rate,
        "metrics": metrics.as_dict(),
        "operating_points": {
            f"{t:.1f}": binary_stats(scores, labels, t)
            for t in (0.1, 0.3, 0.5, 0.7, 0.9)
        },
    }
    path = args.out or str(out_dir(cfg) / f"eval_{model.name}_{cfg.data.source}.json")
    with open(path, "w") as fh:
        json.dump(payload_out, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {path}")
    if cfg.data.source == "synthetic":
        print("source=synthetic: pipeline verification, not a result.")
    return 0


@torch.no_grad()
def _scores(model, loader, info) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(info.device).eval()
    scores, labels = [], []
    for batch in loader:
        batch = batch.to(info.device, non_blocking=True)
        out = model(batch.pose, batch.context, batch.motion)
        scores.append(torch.sigmoid(out.intent_logit.float()).cpu().numpy())
        labels.append(batch.intent.cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels)


if __name__ == "__main__":
    raise SystemExit(main())
