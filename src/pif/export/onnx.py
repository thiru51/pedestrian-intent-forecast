"""ONNX export plus a round-trip check that actually compares numbers.

Exporting is the easy half. The half that catches real bugs is reloading the
graph in onnxruntime and asserting the outputs match torch's within tolerance,
on inputs the exporter never saw — a traced graph that baked in a batch size or
dropped a branch passes a shape check and fails this.

The target for this model is a Jetson-class board running TensorRT, which
consumes ONNX. That is why the export exists at all. On-device latency has not
been benchmarked, because there is no device to benchmark on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from ..features.context import CONTEXT_DIM
from ..features.pose import POSE_DIM
from ..models.fusion import ExportWrapper, IntentForecaster

OPSET = 17

INPUT_NAMES = ("pose", "context", "motion")
OUTPUT_NAMES = ("crossing_prob", "trajectory")


@dataclass
class RoundTripReport:
    path: str
    opset: int
    onnx_ok: bool
    providers: list[str]
    batches_checked: list[int]
    max_abs_diff_prob: float
    max_abs_diff_traj: float
    tolerance: float
    passed: bool
    n_parameters: int
    file_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        return (
            f"{state} onnx round-trip: max|dp|={self.max_abs_diff_prob:.3e} "
            f"max|dtraj|={self.max_abs_diff_traj:.3e} tol={self.tolerance:.1e} "
            f"batches={self.batches_checked} providers={self.providers}"
        )


def dummy_inputs(
    batch: int, window: int, horizon_motion: int = 4, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, window, POSE_DIM, generator=g),
        torch.rand(batch, window, CONTEXT_DIM, generator=g),
        torch.rand(batch, window, horizon_motion, generator=g),
    )


def export(
    model: IntentForecaster,
    path: str | Path,
    window: int | None = None,
    batch: int = 2,
    opset: int = OPSET,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    window = window or model.cfg.window
    # Export from CPU in fp32. The graph is precision-agnostic; whatever runs it
    # (TensorRT, onnxruntime) decides on fp16 at build time, and exporting from
    # an autocast context bakes casts into the graph that you then cannot undo.
    wrapper = ExportWrapper(model).eval().cpu()
    args = dummy_inputs(batch, window)

    torch.onnx.export(
        wrapper,
        args,
        str(path),
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        # Batch stays dynamic so the same file serves a single pedestrian and a
        # batched offline run. Time does not: the window length is part of the
        # model definition (the positional table is sized to it).
        dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES},
        opset_version=opset,
        do_constant_folding=True,
    )
    return path


def verify(
    model: IntentForecaster,
    path: str | Path,
    window: int | None = None,
    batches: tuple[int, ...] = (1, 3, 8),
    tolerance: float = 1e-4,
) -> RoundTripReport:
    import onnx
    import onnxruntime as ort

    path = Path(path)
    window = window or model.cfg.window

    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    wrapper = ExportWrapper(model).eval().cpu()

    max_dp = 0.0
    max_dt = 0.0
    for i, b in enumerate(batches):
        # A different seed per batch size, so none of these are the tensors the
        # export was traced on.
        args = dummy_inputs(b, window, seed=1000 + i)
        with torch.no_grad():
            ref_prob, ref_traj = wrapper(*args)
        got = session.run(
            list(OUTPUT_NAMES),
            {name: t.numpy() for name, t in zip(INPUT_NAMES, args)},
        )
        max_dp = max(max_dp, float(np.abs(ref_prob.numpy() - got[0]).max()))
        max_dt = max(max_dt, float(np.abs(ref_traj.numpy() - got[1]).max()))

    return RoundTripReport(
        path=str(path),
        opset=OPSET,
        onnx_ok=True,
        providers=list(session.get_providers()),
        batches_checked=list(batches),
        max_abs_diff_prob=max_dp,
        max_abs_diff_traj=max_dt,
        tolerance=tolerance,
        passed=bool(max_dp <= tolerance and max_dt <= tolerance),
        n_parameters=model.num_parameters(),
        file_bytes=path.stat().st_size,
    )


def export_and_verify(
    model: IntentForecaster,
    path: str | Path,
    report_path: str | Path | None = None,
    tolerance: float = 1e-4,
) -> RoundTripReport:
    export(model, path)
    report = verify(model, path, tolerance=tolerance)
    if report_path:
        p = Path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    return report
