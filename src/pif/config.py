"""YAML experiment configs, flattened into the dataclasses the code actually
uses.

Command-line flags override the file, the file overrides the defaults. That
order matters for the ablation script, which loads one config and then flips
the branch switches from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .data.windows import DEFAULT_HORIZON, DEFAULT_LEAD, DEFAULT_STRIDE, DEFAULT_WINDOW
from .models.fusion import ModelConfig
from .train.losses import LossWeights
from .train.loop import TrainConfig


@dataclass
class DataConfig:
    source: str = "synthetic"        # synthetic | jaad | pie | carla
    root: str = ""                   # external/jaad, external/pie or data/carla
    split_subset: str = "default"    # JAAD split_ids subdirectory
    window: int = DEFAULT_WINDOW
    lead: int = DEFAULT_LEAD
    stride: int = DEFAULT_STRIDE
    horizon: int = DEFAULT_HORIZON
    n_tracks: int = 600              # synthetic only
    track_length: int = 140          # synthetic only
    positive_fraction: float = 0.4   # synthetic only
    val_fraction: float = 0.25
    seed: int = 0
    max_videos: int = 0              # 0 = all; a cap for quick runs


@dataclass
class RunConfig:
    name: str = "fusion"
    model_type: str = "fusion"       # fusion | pose_only
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    batch_size: int = 0              # 0 = auto from detected VRAM
    num_workers: int = -1            # -1 = auto from os.cpu_count()
    out_dir: str = "artifacts"

    def as_dict(self) -> dict:
        return asdict(self)


def _fill(cls, values: dict | None):
    if not values:
        return cls()
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    return cls(**values)


def load(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = RunConfig(
        name=raw.get("name", "fusion"),
        model_type=raw.get("model_type", "fusion"),
        data=_fill(DataConfig, raw.get("data")),
        model=_fill(ModelConfig, raw.get("model")),
        train=_fill(TrainConfig, raw.get("train")),
        loss=_fill(LossWeights, raw.get("loss")),
        batch_size=int(raw.get("batch_size", 0)),
        num_workers=int(raw.get("num_workers", -1)),
        out_dir=raw.get("out_dir", "artifacts"),
    )
    # The window length lives in two places because the data layer and the
    # model layer both legitimately need it; keeping them in sync here is
    # cheaper than threading one through everything.
    cfg.model.window = cfg.data.window
    cfg.model.horizon = cfg.data.horizon
    if cfg.model_type == "pose_only":
        cfg.model.use_context = False
        cfg.model.use_gesture = False
        cfg.model.fuse_trajectory = False
    return cfg
