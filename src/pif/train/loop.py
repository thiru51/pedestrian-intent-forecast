"""Training and evaluation loop, with the GPU bits kept explicit.

The three things that are easy to get wrong and are therefore written out
rather than hidden in a helper:

  autocast wraps the forward pass and the loss, not the backward pass. The
  backward automatically runs in whatever precision each op recorded on the
  forward, so putting it inside the context does nothing useful and putting the
  optimiser step inside it is actively wrong.

  GradScaler is created only for fp16. bf16 has fp32's exponent range so its
  gradients do not underflow, and PyTorch will warn you if you scale it.

  set_to_none=True on zero_grad frees the gradient buffers instead of filling
  them with zeros. Slightly less memory, and a marginally cheaper kernel.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..device import DeviceInfo, peak_vram_gb, reset_peak_vram, synchronize
from ..features.gesture import GESTURE_GROUPS, GROUP_SLICES, IGNORE_INDEX
from ..metrics import IntentMetrics, gesture_accuracy, intent_metrics
from .losses import MultiTaskLoss


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    log_every: int = 0        # 0 = once per epoch only
    compile_model: bool = False


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    parts: dict[str, float]
    val: IntentMetrics | None
    seconds: float
    samples_per_sec: float
    peak_vram_gb: float
    lr: float


@dataclass
class RunResult:
    model_name: str
    epochs: list[EpochStats] = field(default_factory=list)
    best: IntentMetrics | None = None
    best_epoch: int = -1
    total_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    mean_samples_per_sec: float = 0.0
    device: str = ""
    amp: str = "off"
    params: int = 0

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "amp": self.amp,
            "params": self.params,
            "epochs": len(self.epochs),
            "total_seconds": round(self.total_seconds, 3),
            "peak_vram_gb": round(self.peak_vram_gb, 4),
            "mean_samples_per_sec": round(self.mean_samples_per_sec, 1),
            "best_epoch": self.best_epoch,
            "best": self.best.as_dict() if self.best else None,
            "history": [
                {
                    "epoch": e.epoch,
                    "train_loss": round(e.train_loss, 5),
                    "lr": e.lr,
                    "seconds": round(e.seconds, 3),
                    "samples_per_sec": round(e.samples_per_sec, 1),
                    "peak_vram_gb": round(e.peak_vram_gb, 4),
                    "parts": {k: round(v, 5) for k, v in e.parts.items()},
                    "val": e.val.as_dict() if e.val else None,
                }
                for e in self.epochs
            ],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path


def _lr_at(cfg: TrainConfig, epoch: int) -> float:
    """Linear warmup then cosine decay. Warmup matters less with pre-norm
    blocks, but a couple of epochs of it costs nothing."""
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / max(cfg.warmup_epochs, 1)
    progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
    return cfg.lr * 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0))))


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    info: DeviceInfo,
    cfg: TrainConfig,
    loss_fn: MultiTaskLoss,
    log=print,
) -> RunResult:
    model = model.to(info.device)
    loss_fn = loss_fn.to(info.device)
    params = sum(p.numel() for p in model.parameters())

    forward = model
    if cfg.compile_model:
        # Off by default: on a model this small the compile takes longer than
        # the whole training run, and it is the first thing to break on a new
        # torch. Worth turning on for a long real-data run.
        forward = torch.compile(model)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=info.needs_grad_scaler)

    result = RunResult(
        model_name=getattr(model, "name", model.__class__.__name__),
        device=str(info.device),
        amp="off" if info.amp_dtype is None else str(info.amp_dtype).replace("torch.", ""),
        params=params,
    )
    reset_peak_vram(info)
    run_start = time.perf_counter()
    rates: list[float] = []

    for epoch in range(cfg.epochs):
        lr = _lr_at(cfg, epoch)
        for group in optim.param_groups:
            group["lr"] = lr

        model.train()
        seen = 0
        loss_sum = 0.0
        part_sums: dict[str, float] = {}
        synchronize(info)
        t0 = time.perf_counter()

        for step, batch in enumerate(train_loader):
            batch = batch.to(info.device, non_blocking=True)
            optim.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=info.device.type,
                dtype=info.amp_dtype or torch.float32,
                enabled=info.amp_dtype is not None,
            ):
                out = forward(batch.pose, batch.context, batch.motion)
                loss, parts = loss_fn(out, batch)

            if info.needs_grad_scaler:
                scaler.scale(loss).backward()
                # Unscale before clipping or the clip threshold is applied to
                # scaled gradients and means nothing.
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

            n = len(batch)
            seen += n
            loss_sum += parts["total"] * n
            for k, v in parts.items():
                part_sums[k] = part_sums.get(k, 0.0) + v * n

            if cfg.log_every and step % cfg.log_every == 0:
                log(f"  epoch {epoch} step {step} loss {parts['total']:.4f}")

        synchronize(info)
        elapsed = time.perf_counter() - t0
        rate = seen / elapsed if elapsed > 0 else 0.0
        rates.append(rate)

        val = evaluate(model, val_loader, info) if val_loader is not None else None
        stats = EpochStats(
            epoch=epoch,
            train_loss=loss_sum / max(seen, 1),
            parts={k: v / max(seen, 1) for k, v in part_sums.items()},
            val=val,
            seconds=elapsed,
            samples_per_sec=rate,
            peak_vram_gb=peak_vram_gb(info),
            lr=lr,
        )
        result.epochs.append(stats)

        msg = (
            f"epoch {epoch:3d} lr {lr:.2e} loss {stats.train_loss:.4f} "
            f"{rate:,.0f} samples/s {elapsed:.2f}s"
        )
        if val is not None:
            msg += f" | val {val.line()}"
        log(msg)

        if val is not None and (result.best is None or _better(val, result.best)):
            result.best = val
            result.best_epoch = epoch

    result.total_seconds = time.perf_counter() - run_start
    result.peak_vram_gb = peak_vram_gb(info)
    result.mean_samples_per_sec = float(np.mean(rates)) if rates else 0.0
    if result.best is None and val_loader is not None:
        result.best = evaluate(model, val_loader, info)
        result.best_epoch = cfg.epochs - 1
    return result


def _better(a: IntentMetrics, b: IntentMetrics) -> bool:
    if np.isnan(b.ap):
        return True
    return bool(a.ap > b.ap)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader | None, info: DeviceInfo) -> IntentMetrics:
    if loader is None:
        raise ValueError("evaluate() needs a loader")
    model = model.to(info.device)
    model.eval()

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    traj_pred: list[np.ndarray] = []
    traj_true: list[np.ndarray] = []
    traj_mask: list[np.ndarray] = []
    gest_logits: dict[str, list[np.ndarray]] = {g: [] for g in GESTURE_GROUPS}
    gest_true: dict[str, list[np.ndarray]] = {g: [] for g in GESTURE_GROUPS}

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
        if out.trajectory is not None:
            traj_pred.append(out.trajectory.float().cpu().numpy())
            traj_true.append(batch.future.cpu().numpy())
            traj_mask.append(batch.future_mask.cpu().numpy())
        if out.gesture_logits is not None:
            logits = out.gesture_logits.float().cpu().numpy()
            targets = batch.gesture.cpu().numpy()
            for gi, group in enumerate(GESTURE_GROUPS):
                sl = GROUP_SLICES[group]
                gest_logits[group].append(logits[..., sl].reshape(-1, sl.stop - sl.start))
                gest_true[group].append(targets[..., gi].reshape(-1))

    gacc = float("nan")
    if any(gest_true[g] for g in GESTURE_GROUPS):
        gacc = gesture_accuracy(
            [np.concatenate(gest_logits[g]) for g in GESTURE_GROUPS if gest_logits[g]],
            [np.concatenate(gest_true[g]) for g in GESTURE_GROUPS if gest_true[g]],
            ignore_index=IGNORE_INDEX,
        )

    return intent_metrics(
        np.concatenate(scores),
        np.concatenate(labels),
        traj_pred=np.concatenate(traj_pred) if traj_pred else None,
        traj_target=np.concatenate(traj_true) if traj_true else None,
        traj_mask=np.concatenate(traj_mask) if traj_mask else None,
        gesture_acc=gacc,
    )


def save_checkpoint(model: nn.Module, path: str | Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_name": getattr(model, "name", model.__class__.__name__),
        "config": getattr(model, "cfg", None).as_dict() if hasattr(model, "cfg") else {},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path
