"""End-to-end smoke test on synthetic data.

Walks the whole path once and prints what each stage produced: generate
windows -> build tensors -> build the model -> train a few epochs -> evaluate
-> export ONNX -> verify the round trip. If this runs clean, the plumbing is
intact. It is not a result and it is not evidence about pedestrians; the data
is fabricated by pif.data.synthetic and that module says so at length.

Takes well under a minute on a GPU and a few minutes on CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pif import FPS
from pif.config import DataConfig, RunConfig
from pif.data.build import build
from pif.data.dataset import PedestrianWindows, make_loader
from pif.data.synthetic import STATE_NAMES
from pif.data.windows import summarise
from pif.device import auto_batch_size, peak_vram_gb, setup
from pif.export.onnx import export_and_verify
from pif.features.context import CONTEXT_DIM
from pif.features.gesture import GESTURE_LOGIT_DIM
from pif.features.pose import POSE_DIM
from pif.models.fusion import ModelConfig, build_model
from pif.train.losses import MultiTaskLoss
from pif.train.loop import TrainConfig, evaluate, train

from ._common import add_hardware_args


def rule(title: str) -> None:
    print()
    print(f"== {title} " + "=" * max(0, 68 - len(title)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--tracks", type=int, default=400)
    parser.add_argument("--out", default="artifacts")
    add_hardware_args(parser)
    args = parser.parse_args()

    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rule("device")
    info = setup(args.device, amp=args.amp, seed=args.seed)
    print(info.summary())
    batch_size = args.batch_size or auto_batch_size(info)
    print(f"batch size {batch_size} ({'--batch-size' if args.batch_size else 'auto from VRAM'})")

    rule("synthetic windows")
    print("SYNTHETIC DATA. Generated sequences, not JAAD/PIE. See")
    print("src/pif/data/synthetic.py for exactly what is fabricated and why.")
    data_cfg = DataConfig(
        source="synthetic", n_tracks=args.tracks, seed=args.seed, positive_fraction=0.4
    )
    t0 = time.perf_counter()
    train_windows, val_windows = build(data_cfg)
    print(f"generated in {time.perf_counter() - t0:.2f}s")
    print(f"train {summarise(train_windows)}")
    print(f"val   {summarise(val_windows)}")
    states = {}
    for w in train_windows + val_windows:
        key = w.ped_id.split("_", 1)[1]
        states[key] = states.get(key, 0) + 1
    print(f"latent states: {dict(sorted(states.items()))}  (names: {list(STATE_NAMES)})")

    rule("tensors")
    train_ds = PedestrianWindows(train_windows)
    val_ds = PedestrianWindows(val_windows)
    workers = None if args.num_workers < 0 else args.num_workers
    train_loader = make_loader(train_ds, batch_size, info, True, workers)
    val_loader = make_loader(val_ds, batch_size, info, False, workers)
    sample = next(iter(train_loader))
    print(
        f"pose {tuple(sample.pose.shape)}  context {tuple(sample.context.shape)}  "
        f"motion {tuple(sample.motion.shape)}  gesture {tuple(sample.gesture.shape)}"
    )
    print(
        f"future {tuple(sample.future.shape)}  intent {tuple(sample.intent.shape)}  "
        f"positive rate train {train_ds.positive_rate:.3f} val {val_ds.positive_rate:.3f}"
    )
    print(
        f"feature widths: pose {POSE_DIM} + context {CONTEXT_DIM} + "
        f"gesture logits {GESTURE_LOGIT_DIM} = {POSE_DIM + CONTEXT_DIM + GESTURE_LOGIT_DIM} "
        f"per frame, window {train_ds.window} frames "
        f"({train_ds.window / FPS:.2f}s), lead {data_cfg.lead} frames "
        f"({data_cfg.lead / FPS:.2f}s)"
    )
    if not np.isfinite(sample.pose.numpy()).all():
        raise SystemExit("pose features contain non-finite values")
    if not np.isfinite(sample.context.numpy()).all():
        raise SystemExit("context features contain non-finite values")
    print("all features finite")

    rule("model")
    model_cfg = ModelConfig(window=train_ds.window, horizon=train_ds.horizon)
    model = build_model("fusion", model_cfg)
    print(f"{model.name}: {model.num_parameters():,} parameters")
    with torch.no_grad():
        out_before = model(sample.pose, sample.context, sample.motion)
    print(
        f"forward ok: intent {tuple(out_before.intent_logit.shape)} "
        f"trajectory {tuple(out_before.trajectory.shape)} "
        f"gesture {tuple(out_before.gesture_logits.shape)}"
    )

    rule("untrained baseline metrics")
    untrained = evaluate(model, val_loader, info)
    print(untrained.line())

    rule(f"train {args.epochs} epochs")
    loss_fn = MultiTaskLoss(pos_weight=train_ds.pos_weight())
    train_cfg = TrainConfig(epochs=args.epochs, compile_model=args.compile_model)
    result = train(model, train_loader, val_loader, info, train_cfg, loss_fn)

    rule("result")
    print(f"before training: {untrained.line()}")
    print(f"after training:  {result.best.line() if result.best else 'n/a'}")
    print(
        f"wall clock {result.total_seconds:.2f}s   "
        f"mean {result.mean_samples_per_sec:,.0f} samples/s   "
        f"peak VRAM {peak_vram_gb(info):.3f} GiB"
    )
    first, last = result.epochs[0], result.epochs[-1]
    print(f"train loss {first.train_loss:.4f} -> {last.train_loss:.4f}")
    print(f"loss parts (last epoch): {json.dumps({k: round(v, 4) for k, v in last.parts.items()})}")

    rule("onnx export + round trip")
    report = export_and_verify(
        model, out / "smoke_model.onnx", out / "onnx_roundtrip.json"
    )
    print(report.line())
    print(f"file {report.file_bytes:,} bytes, opset {report.opset}")

    result.save(out / "smoke_train.json")
    (out / "smoke_summary.json").write_text(
        json.dumps(
            {
                "data": "SYNTHETIC - pipeline verification only, not a result",
                "device": str(info.device),
                "amp": result.amp,
                "batch_size": batch_size,
                "n_train_windows": len(train_ds),
                "n_val_windows": len(val_ds),
                "params": model.num_parameters(),
                "untrained": untrained.as_dict(),
                "trained": result.best.as_dict() if result.best else None,
                "mean_samples_per_sec": result.mean_samples_per_sec,
                "peak_vram_gb": result.peak_vram_gb,
                "total_seconds": result.total_seconds,
                "onnx": report.as_dict(),
            },
            indent=2,
        )
        + "\n"
    )

    rule("done")
    print(f"artifacts in {out.resolve()}")
    print(f"total {time.perf_counter() - started:.1f}s")
    if not report.passed:
        print("ONNX round trip FAILED - see above")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
