"""Argument parsing shared by the runnable scripts.

Every script takes the same hardware flags so that "how do I make this use the
GPU properly" has one answer regardless of which script you are running.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pif.config import RunConfig, load
from pif.data.build import build
from pif.data.dataset import PedestrianWindows, make_loader
from pif.device import DeviceInfo, auto_batch_size, setup


def add_hardware_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument(
        "--batch-size", type=int, default=0, help="0 = auto-scale from detected VRAM"
    )
    parser.add_argument(
        "--num-workers", type=int, default=-1, help="-1 = auto from os.cpu_count()"
    )
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true", default=True)
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="torch.compile the model. Off by default: on a model this small "
        "the compile costs more than it saves on short runs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def resolve(args, cfg: RunConfig) -> tuple[DeviceInfo, RunConfig]:
    info = setup(args.device, amp=getattr(args, "amp", True), seed=args.seed)
    if getattr(args, "batch_size", 0):
        cfg.batch_size = args.batch_size
    if not cfg.batch_size:
        cfg.batch_size = auto_batch_size(info)
    if getattr(args, "num_workers", -1) >= 0:
        cfg.num_workers = args.num_workers
    cfg.train.compile_model = bool(getattr(args, "compile_model", False))
    return info, cfg


def load_config(path: str | None, default: str) -> RunConfig:
    return load(path or default)


def make_datasets(cfg: RunConfig, info: DeviceInfo, pose_cache: str | None = None):
    train_windows, val_windows = build(cfg.data, pose_cache)
    if not train_windows or not val_windows:
        raise SystemExit(
            f"data source '{cfg.data.source}' produced "
            f"{len(train_windows)} train / {len(val_windows)} val windows. "
            "For jaad/pie this usually means the annotations are not where the "
            "config says they are."
        )
    train_ds = PedestrianWindows(train_windows)
    val_ds = PedestrianWindows(val_windows)
    workers = None if cfg.num_workers < 0 else cfg.num_workers
    return (
        train_ds,
        val_ds,
        make_loader(train_ds, cfg.batch_size, info, shuffle=True, num_workers=workers),
        make_loader(val_ds, cfg.batch_size, info, shuffle=False, num_workers=workers),
    )


def out_dir(cfg: RunConfig) -> Path:
    p = Path(cfg.out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def banner(info: DeviceInfo, cfg: RunConfig, train_ds=None, val_ds=None) -> None:
    print(info.summary())
    print(
        f"config: {cfg.name} model={cfg.model_type} source={cfg.data.source} "
        f"window={cfg.data.window} lead={cfg.data.lead}f "
        f"({cfg.data.lead / 30:.2f}s) batch={cfg.batch_size} "
        f"workers={'auto' if cfg.num_workers < 0 else cfg.num_workers}"
    )
    if train_ds is not None and val_ds is not None:
        print(
            f"data: {len(train_ds)} train / {len(val_ds)} val windows, "
            f"positive rate {train_ds.positive_rate:.3f} / {val_ds.positive_rate:.3f}"
        )
    if cfg.data.source == "synthetic":
        print(
            "NOTE: source=synthetic. These are generated sequences, not JAAD or "
            "PIE. Numbers below verify the pipeline runs, nothing else."
        )
