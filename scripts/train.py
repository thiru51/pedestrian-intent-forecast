"""Train one model. `python -m scripts.train --config configs/fusion.yaml`"""

from __future__ import annotations

import argparse
import json

from pif.models.fusion import build_model
from pif.train.losses import MultiTaskLoss
from pif.train.loop import save_checkpoint, train

from ._common import add_hardware_args, banner, load_config, make_datasets, out_dir, resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--epochs", type=int, default=0, help="override the config")
    parser.add_argument("--source", default="", help="synthetic | jaad | pie")
    parser.add_argument("--data-root", default="", help="external/jaad or external/pie")
    parser.add_argument("--pose-cache", default="", help=".npz from scripts/extract_pose.py")
    parser.add_argument("--tag", default="", help="suffix for the artifact filenames")
    add_hardware_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, "configs/fusion.yaml")
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.source:
        cfg.data.source = args.source
    if args.data_root:
        cfg.data.root = args.data_root

    info, cfg = resolve(args, cfg)
    train_ds, val_ds, train_loader, val_loader = make_datasets(
        cfg, info, args.pose_cache or None
    )
    banner(info, cfg, train_ds, val_ds)

    model = build_model(cfg.model_type, cfg.model)
    print(f"model: {model.name}, {model.num_parameters():,} parameters")

    loss_fn = MultiTaskLoss(cfg.loss, pos_weight=train_ds.pos_weight())
    result = train(model, train_loader, val_loader, info, cfg.train, loss_fn)

    tag = args.tag or cfg.name
    art = out_dir(cfg)
    result_path = result.save(art / f"train_{tag}.json")
    ckpt = save_checkpoint(
        model,
        art / f"{tag}.pt",
        extra={"data": cfg.data.__dict__, "metrics": result.best.as_dict() if result.best else None},
    )

    print()
    print(f"best epoch {result.best_epoch}: {result.best.line() if result.best else 'n/a'}")
    print(
        f"wall clock {result.total_seconds:.1f}s  "
        f"mean {result.mean_samples_per_sec:,.0f} samples/s  "
        f"peak VRAM {result.peak_vram_gb:.3f} GiB"
    )
    print(f"wrote {result_path}")
    print(f"wrote {ckpt}")
    print(json.dumps({"best": result.best.as_dict() if result.best else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
