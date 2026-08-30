"""The multi-task loss.

Three terms, all of them optional depending on which branches the model has:

  intent      BCE with a positive weight, because crossing windows are the
              minority class.
  trajectory  masked L2 over the future displacements. Masked because a track
              that ends inside the horizon has no ground truth to compare
              against and averaging zeros in would teach the model to shrink
              its predictions.
  gesture     mean of four cross-entropies, one per TITAN-style group, with
              ignore_index for the groups a dataset does not annotate.

The weights are fixed, not learned. Uncertainty weighting (Kendall et al.) is
the obvious upgrade, but with three terms whose scales are known it mostly adds
two parameters that drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..features.gesture import GESTURE_GROUPS, GROUP_SLICES, IGNORE_INDEX


@dataclass
class LossWeights:
    intent: float = 1.0
    trajectory: float = 0.5
    gesture: float = 0.3


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        weights: LossWeights | None = None,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.w = weights or LossWeights()
        self.register_buffer(
            "pos_weight",
            torch.as_tensor(1.0 if pos_weight is None else pos_weight, dtype=torch.float32),
        )

    def forward(self, out, batch) -> tuple[torch.Tensor, dict[str, float]]:
        parts: dict[str, float] = {}

        intent = F.binary_cross_entropy_with_logits(
            out.intent_logit.float(), batch.intent.float(), pos_weight=self.pos_weight
        )
        total = self.w.intent * intent
        parts["intent"] = float(intent.detach())

        if out.trajectory is not None and self.w.trajectory > 0:
            err = (out.trajectory.float() - batch.future.float()).pow(2).sum(-1)
            mask = batch.future_mask.float()
            denom = mask.sum().clamp_min(1.0)
            traj = (err * mask).sum() / denom
            total = total + self.w.trajectory * traj
            parts["trajectory"] = float(traj.detach())

        if out.gesture_logits is not None and self.w.gesture > 0:
            gest = self._gesture_loss(out.gesture_logits, batch.gesture)
            total = total + self.w.gesture * gest
            parts["gesture"] = float(gest.detach())

        parts["total"] = float(total.detach())
        return total, parts

    @staticmethod
    def _gesture_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = []
        for gi, group in enumerate(GESTURE_GROUPS):
            sl = GROUP_SLICES[group]
            group_logits = logits[..., sl].reshape(-1, sl.stop - sl.start).float()
            group_targets = targets[..., gi].reshape(-1)
            if (group_targets != IGNORE_INDEX).sum() == 0:
                continue
            losses.append(
                F.cross_entropy(
                    group_logits, group_targets, ignore_index=IGNORE_INDEX
                )
            )
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()
