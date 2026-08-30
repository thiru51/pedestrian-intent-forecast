"""Export a checkpoint (or a fresh model) to ONNX and verify the round trip.

`python -m scripts.export_onnx --checkpoint artifacts/fusion.pt`

The verification is the part that matters: it reloads the graph in onnxruntime
and compares against torch on inputs the exporter never traced, at three batch
sizes. Exit code is non-zero if they disagree beyond tolerance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pif.models.fusion import ModelConfig, build_model
from pif.export.onnx import export_and_verify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="", help="artifacts/*.pt, optional")
    parser.add_argument("--model", default="fusion", help="fusion | pose_only")
    parser.add_argument("--out", default="artifacts/model.onnx")
    parser.add_argument("--report", default="artifacts/onnx_roundtrip.json")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**payload.get("config", {}))
        model = build_model(payload.get("model_name", args.model), cfg)
        model.load_state_dict(payload["state_dict"])
        print(f"loaded {args.checkpoint}: {model.name}, {model.num_parameters():,} params")
    else:
        model = build_model(args.model)
        print(
            f"no --checkpoint given, exporting a freshly initialised {model.name} "
            f"({model.num_parameters():,} params). The round-trip check is about "
            "the graph, not the weights, so this is still a real check."
        )

    model.eval()
    report = export_and_verify(model, args.out, args.report, tolerance=args.tolerance)

    print()
    print(report.line())
    print(f"wrote {Path(args.out).resolve()} ({report.file_bytes:,} bytes)")
    print(f"wrote {Path(args.report).resolve()}")
    print()
    print(
        "This is the artifact a Jetson-class deployment would consume via "
        "TensorRT. On-device latency has not been benchmarked - there is no "
        "device here to benchmark on."
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
