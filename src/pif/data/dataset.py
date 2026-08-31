"""Torch Dataset and DataLoader construction.

Windows are small dense float arrays, not images, so the whole split is stacked
into contiguous tensors once at construction time and indexing is a slice. That
makes the loader almost free and means the GPU is the thing we are actually
measuring, which is the point of the throughput numbers in the training log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..device import DeviceInfo, auto_num_workers
from ..features.context import CONTEXT_DIM
from ..features.gesture import GESTURE_GROUPS
from ..features.pose import POSE_DIM
from .windows import Window


@dataclass(frozen=True)
class Batch:
    pose: torch.Tensor        # (B, T, POSE_DIM)
    context: torch.Tensor     # (B, T, CONTEXT_DIM)
    motion: torch.Tensor      # (B, T, 4)
    gesture: torch.Tensor     # (B, T, 4) int64 class ids, -1 = ignore
    intent: torch.Tensor      # (B,) float32
    future: torch.Tensor      # (B, H, 2)
    future_mask: torch.Tensor  # (B, H)

    def to(self, device: torch.device, non_blocking: bool = True) -> "Batch":
        return Batch(
            **{
                k: getattr(self, k).to(device, non_blocking=non_blocking)
                for k in (
                    "pose", "context", "motion", "gesture",
                    "intent", "future", "future_mask",
                )
            }
        )

    def __len__(self) -> int:
        return self.intent.shape[0]


class PedestrianWindows(Dataset):
    def __init__(self, windows: list[Window]):
        if not windows:
            raise ValueError("PedestrianWindows got an empty window list")
        self.meta = [(w.video_id, w.ped_id, w.t_end) for w in windows]
        self.pose = torch.from_numpy(np.stack([w.pose for w in windows]))
        self.context = torch.from_numpy(np.stack([w.context for w in windows]))
        self.motion = torch.from_numpy(np.stack([w.motion for w in windows]))
        self.gesture = torch.from_numpy(np.stack([w.gesture for w in windows]))
        self.intent = torch.tensor([w.intent for w in windows], dtype=torch.float32)
        self.future = torch.from_numpy(np.stack([w.future for w in windows]))
        self.future_mask = torch.from_numpy(np.stack([w.future_mask for w in windows]))

        t = self.pose.shape[1]
        assert self.pose.shape[2] == POSE_DIM, self.pose.shape
        assert self.context.shape[1:] == (t, CONTEXT_DIM), self.context.shape
        assert self.gesture.shape[1:] == (t, len(GESTURE_GROUPS)), self.gesture.shape

    def __len__(self) -> int:
        return self.intent.shape[0]

    def __getitem__(self, i: int):
        return (
            self.pose[i],
            self.context[i],
            self.motion[i],
            self.gesture[i],
            self.intent[i],
            self.future[i],
            self.future_mask[i],
        )

    @property
    def window(self) -> int:
        return int(self.pose.shape[1])

    @property
    def horizon(self) -> int:
        return int(self.future.shape[1])

    @property
    def positive_rate(self) -> float:
        return float(self.intent.mean())

    def pos_weight(self) -> torch.Tensor:
        """BCE positive weight, so a 15%-positive split does not train a model
        that just says "no" to everything."""
        pos = float(self.intent.sum())
        neg = float(len(self) - pos)
        if pos == 0:
            return torch.tensor(1.0)
        return torch.tensor(max(neg / pos, 1e-3), dtype=torch.float32)


def collate(items) -> Batch:
    pose, context, motion, gesture, intent, future, future_mask = zip(*items)
    return Batch(
        pose=torch.stack(pose),
        context=torch.stack(context),
        motion=torch.stack(motion),
        gesture=torch.stack(gesture),
        intent=torch.stack(intent),
        future=torch.stack(future),
        future_mask=torch.stack(future_mask),
    )


def make_loader(
    dataset: PedestrianWindows,
    batch_size: int,
    info: DeviceInfo,
    shuffle: bool = True,
    num_workers: int | None = None,
    prefetch_factor: int = 4,
    drop_last: bool = False,
) -> DataLoader:
    workers = auto_num_workers(num_workers)
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate,
        "drop_last": drop_last,
        # Pinned host memory is what makes the .to(non_blocking=True) in the
        # training loop actually asynchronous; without it the copy blocks.
        "pin_memory": info.is_cuda,
    }
    if workers > 0:
        # Respawning workers every epoch costs more than the epoch does on a
        # dataset this size.
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)  # type: ignore[arg-type]
