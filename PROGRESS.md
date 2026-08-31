# Progress

State of the repository. `[x]` means done and checked, `[ ]` means not done.
Nothing is marked done on the strength of "the code exists" — it has to have
been run.

Last updated: 2026-08-31.

## Data

- [x] JAAD annotation parser, against the real released XML (346 videos)
- [x] PIE annotation parser, reading the three zips without unpacking them
- [x] Dataset-neutral schema both parsers fill in
- [x] Windowing with the look-ahead label, and the two rules that keep it a
      forecast (no observation past the crossing point; target frame inside the
      annotated track)
- [x] Occlusion-gap windows dropped rather than silently spanning a hole
- [x] JAAD split files honoured; PIE's published set-level split (01-04 train,
      05-06 test) honoured
- [x] Synthetic generator, so the whole path runs without any download
- [x] Verified parsing on real annotations: 20 JAAD videos -> 589 windows,
      25.5% positive; 5 PIE videos -> 12,370 windows, 4.6% positive
- [ ] **JAAD video clips downloaded** (3.1 GB) — OPEN, blocks everything below
- [ ] **PIE video clips downloaded** (74 GB) — OPEN
- [ ] Pose cache built from real clips (`scripts/extract_pose.py`)

## Features

- [x] Pose encoder: 17 COCO joints + confidence + 8 derived scalars = 59 dims
- [x] MediaPipe (BlazePose) adapter into the COCO-17 interchange order
- [x] Scene-context encoder: 23 channels, all computable from annotations alone
- [x] Gesture adapter: behaviour strings -> four TITAN-style groups, with an
      ignore index for groups a dataset does not annotate
- [x] Phone-distraction override from JAAD's appearance file
- [x] Tests that the named features actually respond to the thing they are
      named after (lean, shoulder yaw, arm extension, curb distance, scale rate)
- [ ] Pose features from real video (needs the clips)
- [ ] HRNet-W32 extractor as a drop-in replacement for MediaPipe

## Model

- [x] Hand-written pre-norm temporal transformer over the 16-frame window
- [x] Gesture head: four parallel softmaxes over one trunk, output also feeds
      the temporal branch (detached)
- [x] Intent head
- [x] Trajectory GRU, hidden state initialised from the intent summary
- [x] `PoseOnlyBaseline` built through the same class with branches off, so
      there is no second implementation to drift
- [x] Multi-task loss: intent BCE with positive weighting, masked trajectory
      L2, four masked gesture cross-entropies
- [ ] Uncertainty-weighted loss (Kendall et al.) instead of fixed weights —
      considered, deliberately not done yet

## Training

- [x] Training loop with AMP, TF32, grad clipping, warmup + cosine schedule
- [x] Automatic device / batch size / worker selection in one module
- [x] Per-epoch history written to JSON: loss parts, LR, seconds, samples/s,
      peak VRAM, full validation metrics
- [x] Checkpoints carry their model config, so evaluate and export can rebuild
      the model without being told which one it was
- [x] Runs end to end on synthetic data
- [ ] **Trained on real JAAD** — OPEN, waiting on clips
- [ ] **Trained on real PIE** — OPEN
- [ ] Ablation run on real data (fused vs pose-only, 3 seeds per arm)
- [ ] Context-only arm, to separate "context does the work" from "fusion does
      the work"
- [ ] AP-versus-lead-time curve (1.0 s / 1.5 s / 2.0 s)

## Metrics

- [x] Average precision, ROC AUC, precision/recall/F1
- [x] Miss rate and false alarm rate reported separately, because a planner
      cares about a missed crossing specifically
- [x] ADE / FDE for the trajectory head, masked
- [x] Gesture accuracy per group, ignoring unsupervised labels
- [x] All checked against worked examples with known answers, not against
      another library
- [x] Operating curve (thresholds 0.1 to 0.9) in `scripts/evaluate.py`

## Export and deployment

- [x] ONNX export with a dynamic batch axis and a fixed window axis
- [x] Round-trip verification: reload in onnxruntime, compare against torch on
      inputs the exporter never traced, at three batch sizes
- [x] Non-zero exit code when the comparison fails
- [ ] TensorRT engine built from the ONNX
- [ ] **Latency measured on a Jetson** — OPEN, and it will stay open until
      there is a physical device. No FPS number will be claimed before then
- [ ] INT8 or fp16 calibration

## Repository

- [x] pixi environment (`pixi.toml` + `pixi.lock`)
- [x] `requirements.txt` for the venv + pip path, generated from real imports
- [x] Dockerfile
- [x] `scripts/fetch_external.sh` to re-clone the JAAD and PIE reference repos
- [x] `external/` git-ignored, never committed
- [x] 37 unit tests passing
- [x] README, HANDOFF, END_GOAL, PROGRESS
- [ ] CI

## Known open items, in the order they block things

1. **Dataset access.** The annotations are in hand; the video clips are not.
   Nothing about real pedestrians can be said until they are. This blocks pose
   extraction, which blocks training, which blocks the ablation, which is the
   whole point.
2. **No Jetson.** The ONNX file is verified numerically but has never run on
   the target class of device.
3. **`transportive` is barely supervised.** JAAD only annotates it for the
   high-visibility subset, PIE not at all. The head exists to keep the taxonomy
   faithful to TITAN, and the loss ignores it where there is no label — but on
   PIE it will train on nothing.
4. **The ego corridor is a fixed wedge**, not a calibrated projection. PIE
   ships camera parameters; JAAD does not. Good enough as a stand-in, and it
   should be replaced with the real thing for PIE.
