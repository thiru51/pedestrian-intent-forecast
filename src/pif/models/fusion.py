"""The two models the ablation compares.

IntentForecaster is the full thing: pose + scene context -> gesture head ->
temporal transformer over [pose | context | gesture logits] -> crossing intent,
with a trajectory GRU conditioned on the intent summary.

PoseOnlyBaseline is the same transformer over pose alone. No context branch, no
gesture head, no fusion, no trajectory conditioning. That is roughly the family
of pose-only crossing classifiers Fang & Lopez (IV 2018) reported, and it is
the thing the extra machinery has to beat to have earned its place.

The two share the transformer implementation and the intent head on purpose:
if the baseline used a different backbone the comparison would be measuring the
backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from ..features.context import CONTEXT_DIM
from ..features.gesture import GESTURE_LOGIT_DIM
from ..features.pose import POSE_DIM
from .heads import GestureHead, IntentHead, TrajectoryHead
from .transformer import TemporalTransformer


@dataclass
class ModelConfig:
    window: int = 16
    horizon: int = 15
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    gesture_hidden: int = 96
    intent_hidden: int = 64
    traj_hidden: int = 64
    use_context: bool = True
    use_gesture: bool = True
    fuse_trajectory: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Output:
    intent_logit: torch.Tensor            # (B,)
    trajectory: torch.Tensor              # (B, H, 2)
    gesture_logits: torch.Tensor | None   # (B, T, GESTURE_LOGIT_DIM)


class IntentForecaster(nn.Module):
    name = "fusion"

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        per_frame = POSE_DIM + (CONTEXT_DIM if c.use_context else 0)
        self.gesture_head = (
            GestureHead(per_frame, c.gesture_hidden, c.dropout) if c.use_gesture else None
        )
        token_dim = per_frame + (GESTURE_LOGIT_DIM if c.use_gesture else 0)

        self.encoder = TemporalTransformer(
            in_dim=token_dim,
            d_model=c.d_model,
            n_layers=c.n_layers,
            n_heads=c.n_heads,
            mlp_ratio=c.mlp_ratio,
            dropout=c.dropout,
            window=c.window,
        )
        self.intent = IntentHead(c.d_model, c.intent_hidden, c.dropout)
        self.trajectory = TrajectoryHead(
            motion_dim=4,
            hidden=c.traj_hidden,
            horizon=c.horizon,
            cond_dim=c.d_model if c.fuse_trajectory else None,
            dropout=c.dropout,
        )

    def forward(
        self,
        pose: torch.Tensor,
        context: torch.Tensor,
        motion: torch.Tensor,
    ) -> Output:
        frame = torch.cat([pose, context], dim=-1) if self.cfg.use_context else pose

        gesture_logits = None
        tokens = frame
        if self.gesture_head is not None:
            gesture_logits = self.gesture_head(frame)
            # Detached from the temporal branch's gradient: the gesture head is
            # supervised directly, and letting the intent loss also push it
            # turns it into an unconstrained feature extractor that stops
            # matching the taxonomy it is supposed to represent.
            tokens = torch.cat([frame, gesture_logits.detach()], dim=-1)

        _, summary = self.encoder(tokens)
        intent_logit = self.intent(summary)
        traj = self.trajectory(motion, summary if self.cfg.fuse_trajectory else None)
        return Output(intent_logit, traj, gesture_logits)

    @torch.no_grad()
    def predict_proba(
        self, pose: torch.Tensor, context: torch.Tensor, motion: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self(pose, context, motion).intent_logit)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PoseOnlyBaseline(IntentForecaster):
    """Same backbone, pose channel only. Constructed through IntentForecaster
    with the branches switched off so there is no second implementation to
    drift."""

    name = "pose_only"

    def __init__(self, cfg: ModelConfig | None = None):
        cfg = cfg or ModelConfig()
        stripped = ModelConfig(
            **{
                **cfg.as_dict(),
                "use_context": False,
                "use_gesture": False,
                "fuse_trajectory": False,
            }
        )
        super().__init__(stripped)

    def forward(
        self, pose: torch.Tensor, context: torch.Tensor, motion: torch.Tensor
    ) -> Output:
        # Signature kept identical to the fused model so the training loop, the
        # eval loop and the ONNX exporter do not need to know which one they
        # were handed. context is accepted and dropped.
        return super().forward(pose, torch.zeros_like(context), motion)


class ExportWrapper(nn.Module):
    """Plain-tensor forward for ONNX: the dataclass Output does not survive
    tracing, and a downstream runtime wants (probability, trajectory) anyway."""

    def __init__(self, model: IntentForecaster):
        super().__init__()
        self.model = model

    def forward(
        self, pose: torch.Tensor, context: torch.Tensor, motion: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(pose, context, motion)
        return torch.sigmoid(out.intent_logit), out.trajectory


def build_model(name: str, cfg: ModelConfig | None = None) -> IntentForecaster:
    if name in ("fusion", "fused", "full"):
        return IntentForecaster(cfg)
    if name in ("pose_only", "baseline", "pose"):
        return PoseOnlyBaseline(cfg)
    raise ValueError(f"unknown model '{name}' (expected 'fusion' or 'pose_only')")
