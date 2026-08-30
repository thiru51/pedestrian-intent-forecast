"""Prediction heads: gesture attributes, crossing intent, future trajectory."""

from __future__ import annotations

import torch
from torch import nn

from ..features.gesture import GESTURE_GROUPS, GESTURE_LOGIT_DIM, GROUP_SLICES
from ..taxonomy import GROUP_SIZES


class GestureHead(nn.Module):
    """Per-frame gesture attributes from per-frame features.

    Four independent softmaxes over one shared trunk rather than one softmax
    over the product of the groups. TITAN's point is that the groups are
    parallel — a person can be walking and waving and pushing a pram at the
    same time — so a joint class list would be 3*5*4*4 = 240 mostly empty
    classes.

    Its output is also an input: the concatenated logits are what the temporal
    model sees on the gesture channel, so the head is both supervised and part
    of the forward path.
    """

    def __init__(self, in_dim: int, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.groups = nn.ModuleDict(
            {name: nn.Linear(hidden, GROUP_SIZES[name]) for name in GESTURE_GROUPS}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, in_dim) -> (B, T, GESTURE_LOGIT_DIM), groups concatenated in
        GESTURE_GROUPS order."""
        h = self.trunk(x)
        return torch.cat([self.groups[name](h) for name in GESTURE_GROUPS], dim=-1)

    @staticmethod
    def split(logits: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: logits[..., GROUP_SLICES[name]] for name in GESTURE_GROUPS}


class IntentHead(nn.Module):
    """Window summary -> one logit: P(crossing at t_end + lead)."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TrajectoryHead(nn.Module):
    """GRU over the recent box track, conditioned on the intent branch.

    The conditioning is the whole reason the two heads share a model. Where a
    pedestrian will be in half a second depends on whether they are about to
    step off the kerb, and a motion-only predictor has no way to know that
    until the motion has already changed — which is the same one-second-late
    problem the intent branch exists to fix.

    Fusion is done by initialising the GRU's hidden state from the intent
    summary, rather than concatenating it to every input step: it gives the
    recurrence a starting belief and leaves the per-step inputs purely
    kinematic.
    """

    def __init__(
        self,
        motion_dim: int = 4,
        hidden: int = 64,
        horizon: int = 15,
        cond_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden = hidden
        self.gru = nn.GRU(motion_dim, hidden, batch_first=True)
        self.cond = nn.Linear(cond_dim, hidden) if cond_dim else None
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * 2),
        )

    def forward(
        self, motion: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Velocities, not absolute positions: where in the image the person is
        # belongs to the context branch, and feeding it here again just makes
        # the GRU memorise camera framing.
        deltas = torch.diff(
            motion, dim=1, prepend=motion[:, :1]
        )
        h0 = None
        if self.cond is not None and cond is not None:
            h0 = torch.tanh(self.cond(cond)).unsqueeze(0).contiguous()
        _, hn = self.gru(deltas, h0)
        return self.out(hn[-1]).view(-1, self.horizon, 2)


GESTURE_OUT_DIM = GESTURE_LOGIT_DIM
