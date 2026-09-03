# Progress

State of the repository. `[x]` means done and checked, `[ ]` means not done.
Nothing is marked done on the strength of "the code exists" — it has to have
been run.

Last updated: 2026-09-03.

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

### CARLA, the way around the dataset gate

JAAD and PIE video are access-gated. CARLA is a free driving simulator and can
produce crossing episodes with exact labels, so the project can start training
before that access lands. The trade is stated everywhere it matters: **CARLA
gives perfect labels but is synthetic. Train on CARLA, evaluate on real JAAD
and PIE once access lands, and report the two numbers separately.**

- [x] CARLA recording parser (`src/pif/data/carla.py`), filling the same
      `PedTrack` / `VideoAnnotation` schema as JAAD and PIE, so nothing
      downstream changes
- [x] Crossing label derived to JAAD's convention exactly: -1 not relevant to
      the ego, 0 did not cross, 1 crossed, and `crossing_point` as the frame
      number they first entered the road or -1
- [x] Occlusion mapped onto the existing `{none: 0, part: 1, full: 2}` codes,
      from a ray-cast visibility measurement rather than a guess
- [x] Collector (`scripts/collect_carla.py`): synchronous mode, fixed 0.05 s
      step, fixed seed, ego with an RGB camera whose intrinsics are recorded,
      walkers on `WalkerAIController`, four scripted scenarios (straight,
      hesitate at the kerb, step out from behind a parked car, never cross),
      and teardown that destroys every actor and restores async mode on
      exception
- [x] `scripts/fetch_carla.sh` to download and unpack the 8.3 GB server
- [x] Parser tests that run in the pixi environment with no carla installed and
      no server, against a small committed fixture
- [ ] **Episodes actually collected from a running server** — OPEN. The
      collector has never talked to a live CARLA. No frame count, episode
      count or timing from it is quoted anywhere in this repository
- [ ] Training run on CARLA episodes
- [ ] `carla` added as a source in `pif/data/build.py` so `--config` can point
      at it directly. Right now recordings are loaded through
      `CARLARecordings` by hand

**Gestures are the one thing CARLA cannot supply.** A CARLA walker has no hand
signals, no gaze target and no phone, so `action`, `look` and `gesture` come
back empty and the parser refuses to fill them. Three of the four taxonomy
groups are unsupervised on CARLA data; use
`pif.data.carla.mask_unlabelled_gestures` so the loss skips them instead of
learning a fallback default. This is the clearest single reason real JAAD and
PIE evaluation is still needed.

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
- [x] `scripts/fetch_carla.sh` to fetch and unpack the simulator
- [x] `external/` git-ignored, never committed
- [x] The CARLA server, its tarball and collected episodes git-ignored too
- [x] Second environment `.venv-carla` for the collector, because the carla
      wheel is tagged `manylinux_2_31` and the conda-forge python pixi installs
      declares only `manylinux_2_28`, so pip refuses it there. System glibc is
      2.39, so nothing is really too old — it is a tagging disagreement. The
      two environments share files on disk and never a process
- [x] 58 unit tests passing
- [x] README, HANDOFF, END_GOAL, PROGRESS
- [ ] CI

## Known open items, in the order they block things

1. **Collect the first CARLA episodes.** The collector is written and its
   arithmetic is unit tested, but it has never run against a server. This is
   the shortest path to a real training run, because it does not wait on
   anybody's access request.
2. **Dataset access.** The annotations are in hand; the video clips are not.
   Nothing about real *pedestrians* can be said until they are — a CARLA score
   is a statement about a simulator. This still blocks the gesture branch
   entirely, since CARLA has no gesture labels at all.
3. **No Jetson.** The ONNX file is verified numerically but has never run on
   the target class of device.
4. **`transportive` is barely supervised.** JAAD only annotates it for the
   high-visibility subset, PIE not at all. The head exists to keep the taxonomy
   faithful to TITAN, and the loss ignores it where there is no label — but on
   PIE it will train on nothing.
5. **The ego corridor is a fixed wedge**, not a calibrated projection. PIE
   ships camera parameters; JAAD does not. Good enough as a stand-in, and it
   should be replaced with the real thing for PIE. CARLA recordings carry real
   camera intrinsics, so for those it could be replaced outright.
