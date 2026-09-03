# pedestrian-intent-forecast

Predict whether a pedestrian is about to cross the road **1.5 seconds before
they step off the kerb**, by combining body pose, gesture and scene context
instead of waiting for their trajectory to give it away.

The repository contains the full pipeline: parsers for the JAAD and PIE
annotation releases, a CARLA simulator collector that generates crossing
scenarios with exact labels, a pose feature encoder, a scene-context encoder, a
gesture taxonomy modelled on TITAN, a small temporal transformer over 16-frame
windows, an intent head, a trajectory head, a pose-only baseline for the
ablation, a training loop, and an ONNX export with a numerical round-trip
check.

---

## Status — read this before quoting any number

| Thing | State |
| --- | --- |
| Pipeline (windows -> features -> model -> loss -> metrics -> ONNX) | Implemented, runs end to end |
| Unit tests | 58 tests, all passing |
| JAAD annotation parser | Verified against the real released XML (346 videos) |
| PIE annotation parser | Verified against the real released zips (53 videos) |
| CARLA recording parser | Written, unit-tested against a committed fixture |
| CARLA collector | Written. Not yet run against a live server, see below |
| Episodes collected from CARLA | **None yet.** No frame counts or timings are quoted anywhere |
| ONNX export + round trip | Verified numerically, see below |
| Training on real JAAD/PIE video | **Has not happened.** Dataset video clips not yet obtained |
| Accuracy on real pedestrians | **No such number exists in this repo** |
| On-device (Jetson) latency | **Not benchmarked.** There is no physical device here |

Everything that currently runs end to end runs on **synthetic data** generated
by `src/pif/data/synthetic.py`. That module fabricates sequences with the right
shapes and a plausible causal story so the code path can be exercised. Its
numbers verify that the plumbing works. They are **not results**, they are not
evidence about pedestrians, and they must never be quoted as accuracy. The
generator says so at length in its own docstring, and the scripts print a
warning whenever the source is synthetic.

CARLA changes the picture but does not finish it. The simulator produces
crossing episodes with labels that are exact rather than annotated: the
walker's 3D bounding box is known, "on the road" is a map query, and how much
of the person is hidden is measured by casting rays at them. That is real
training data in a way `synthetic.py` never was. It is still a simulation.
Simulated people move like a navigation mesh, not like people, and a model that
does well on CARLA has not been shown to do well on a street. **The plan is to
train on CARLA and evaluate on real JAAD and PIE once access lands**, and the
CARLA number and the JAAD number will always be reported separately.

One thing CARLA cannot supply at all: **gestures**. A CARLA walker has no hand
signals, no gaze target and no phone, so the `action`, `look` and `gesture`
columns come back empty and the parser refuses to fill them in. Three of the
four taxonomy groups are therefore unsupervised on CARLA data. That is the
single clearest reason real JAAD and PIE evaluation still matters.

The model is designed to be deployed on a Jetson-class board via ONNX and
TensorRT. That is why the exporter exists and why it is verified. On-device
latency has not been measured, because there is no device to measure it on.

---

## Why forecast intent instead of reacting to trajectory

A trajectory predictor watches where a pedestrian has been and extrapolates
where they are going. It works, and it is late.

The reason it is late is simple. A person's trajectory only says "crossing"
once they have already started to cross. Everything before that — standing at
the kerb, turning their shoulders towards the road, checking over their
shoulder for traffic, shortening their stride, raising a hand at the driver —
is invisible to a model that only sees the centre of a bounding box move. By
the time the box moves into the road, the decision was made about a second
ago, and the planner has spent that second doing nothing.

The arithmetic matters here. A car at 50 km/h covers about 14 metres every
second. A 1.5-second earlier warning is roughly 21 metres of extra road — the
difference between a smooth deceleration and an emergency stop. That is not a
measured result, it is just speed times time, but it is the whole reason this
project targets a forecast horizon rather than a detection.

So the task in this repo is deliberately defined as a forecast:

> Given 16 consecutive observed frames ending at frame `t_end`, is this
> pedestrian crossing the ego vehicle's path at frame `t_end + 45`?

45 frames at 30 fps is 1.5 seconds. The windowing code enforces two rules that
keep this honest, and they are tested:

1. **A window may never extend past the pedestrian's crossing point.** Both
   datasets annotate the frame where the person steps into the road. Any
   window ending after that is an observation of a crossing, not a forecast of
   one. (`tests/test_windows.py::test_never_observes_past_the_crossing_point`)
2. **The target frame must be inside the annotated track.** No extrapolating a
   label past the end of the annotation.

There is also a test that asserts at least one window is labelled positive
while ending *before* the crossing has begun — otherwise the "forecast" would
be a relabelled detection.

---

## The approach

### 1. Annotation parsing (JAAD and PIE)

Both datasets come from the same group and share most of their vocabulary, but
not their file layout.

- **JAAD** ships five parallel XML trees per video: per-frame boxes and
  behaviour tags, per-track attributes (crossing, crossing point, number of
  lanes), per-frame scene flags (crosswalk present, pedestrian sign, stop
  sign, traffic light), per-frame ego action (a categorical
  `stopped/moving_slow/moving_fast/...`), and an appearance file that only
  covers the high-visibility subset.
- **PIE** ships one XML tree per video packed inside three zip archives, plus
  real OBD telemetry (speed, GPS, yaw) instead of a categorical ego action, and
  boxes traffic infrastructure (crosswalks, signs, lights) as tracked objects.
  PIE also carries two fields JAAD does not: `intention_prob`, a human rating
  of how likely annotators thought the pedestrian was to cross, collected from
  frames *before* the decision; and `critical_point`, the frame where the ego
  vehicle had to react.

`src/pif/data/schema.py` defines dataset-neutral containers. Both parsers fill
in the same structures, so nothing downstream needs to know which dataset it
got. PIE's XML is read straight out of the zips — unpacked it is a few hundred
megabytes for something that gets streamed through once.

### 2. Pose features

`src/pif/features/pose.py` turns one person in one frame into a 59-dimensional
vector: 17 COCO joints as (x, y, confidence) plus 8 derived scalars
(`torso_lean`, `shoulder_yaw`, `hip_yaw`, `stride_width`, `knee_lift`,
`head_offset`, `arm_extension`, `visible_frac`).

Two decisions worth knowing about:

- Joint coordinates are normalised into the pedestrian's bounding box and then
  re-centred on the hip. A person 20 pixels tall at the end of the street and
  one 400 pixels tall at the kerb produce comparable vectors, and *where* they
  are in the box does not leak into the pose branch — that is the context
  branch's job.
- COCO-17 is the interchange format. The default estimator is MediaPipe Pose
  (BlazePose, 33 landmarks, a strict superset of COCO's 17, CPU-real-time). The
  estimator I would swap in for a real run is HRNet-W32 (Sun et al., CVPR
  2019): same 17 outputs, markedly better on the small partly occluded
  pedestrians that matter here, but ~28M parameters per person per frame, so it
  wants a GPU. Because everything is normalised into the same joint order,
  swapping means replacing `PoseExtractor` and nothing else.

### 3. Scene-context features — and why pose alone is not enough

This is the core argument of the project, so it is worth stating flatly:

> A person standing still at the edge of the road, body squared up to traffic,
> looks *identical in pose space* to a person standing still at a bus stop.

Posture tells you what the body is doing. It cannot tell you what the body is
aimed at. The difference between "waiting to cross" and "waiting for a bus" is
entirely in where they are standing and what is around them: how far their feet
are from the lane the car will drive through, whether there is a crosswalk,
what the light is doing, how fast the ego vehicle is closing.

`src/pif/features/context.py` produces 23 channels per frame, all computable
from annotations alone (no video needed). The three that carry most of the
weight:

- **crosswalk proxy** — PIE boxes crosswalks explicitly; JAAD gives a per-frame
  scene flag. Both fold into one channel, plus a second "there is a boxed
  crosswalk right at their feet" channel for PIE.
- **curb distance** — no dataset labels the kerb, so it is estimated from an
  ego-lane corridor: a wedge in image space, about one lane wide at the bottom
  of the frame and narrowing towards the horizon. The feature is how far the
  pedestrian's feet are from the edge of that wedge, measured in their own box
  widths. Crossing means that number goes to zero. There is a test that walks a
  synthetic pedestrian towards the lane and asserts the feature falls.
- **ego proximity** — box height over image height is a monotone stand-in for
  1/distance under a fixed focal length, so height times ego speed is a usable
  inverse time-to-contact.

### 4. Gesture taxonomy — borrowed, not invented

The label set in `src/pif/taxonomy.py` is modelled on the hierarchy from
**TITAN** (Malla, Dariush & Choi, CVPR 2020), which splits pedestrian labels
into parallel groups rather than one flat class list. The reason is that a
person can be walking *and* on the phone *and* pushing a pram at the same
time; a single softmax over the product would be 240 mostly-empty classes.

Four groups are kept (TITAN's "complex contextual" group is dropped because
neither JAAD nor PIE annotates it):

| Group | Classes |
| --- | --- |
| `atomic` (3) | standing, walking, running |
| `communicative` (5) | none, looking at traffic, waving through, requesting right of way, phone distracted |
| `contextual` (4) | walking along, waiting to cross, hesitating, crossing |
| `transportive` (4) | none, carrying, pushing, pulling |

Concatenated that is a 16-wide logit vector. `src/pif/features/gesture.py` is
the annotation-side adapter: it reads the raw behaviour strings and emits class
ids per group per frame. Groups a dataset does not annotate are emitted as
`-1` and the loss **ignores** them rather than training them towards a default
— otherwise the transportive head would just learn the constant `NONE`. On real
JAAD, `transportive` comes only from the appearance file, which covers the
high-visibility subset; on PIE it is unsupervised everywhere.

Nothing in this module reads pixels. At inference the gesture head predicts
these labels from pose plus context; the annotation strings are training signal
only.

### 5. Temporal transformer

`src/pif/models/transformer.py` is a small pre-norm transformer encoder over
the 16-frame window, written out by hand rather than pulled from
`nn.TransformerEncoder`. Two reasons: `nn.TransformerEncoder` carries a lot of
mask and nested-tensor branching that ONNX export has historically been awkward
about, and this repo has to export cleanly; and at T=16 the attention is a
16x16 matrix, so the interesting engineering is entirely in the projection
widths and the norm placement.

Defaults: `d_model=128`, 3 layers, 4 heads, MLP ratio 2.0, dropout 0.1.
Positional information is a learned per-step table, not sinusoids — T is fixed
at 16 and the steps are 33 ms apart, so there is no length generalisation to
preserve. Pooling over time is learned (a softmax over a per-step score), which
stays a plain matmul for ONNX.

Per-frame input widths:

| Model | Per-frame token |
| --- | --- |
| Fused | pose 59 + context 23 + gesture logits 16 = **98** |
| Pose-only baseline | pose 59 |

### 6. Trajectory head

A GRU over the recent box track, conditioned on the intent branch by
initialising its hidden state from the window summary. The conditioning is the
whole reason the two heads share a model: where a pedestrian will be in half a
second depends on whether they are about to step off the kerb, and a
motion-only predictor cannot know that until the motion has already changed —
the same one-second-late problem the intent branch exists to fix. It regresses
15 future centre *displacements* (0.5 s), masked where the track runs out.

### 7. The pose-only baseline, and why the ablation means something

`PoseOnlyBaseline` is the same transformer, the same intent head, the same
optimiser, the same schedule, the same windows and the same seed, with the
context branch, the gesture head and the trajectory conditioning switched off.
It is constructed *through* `IntentForecaster` with the branches disabled, so
there is no second implementation to drift.

That arm is roughly the family of methods the project argues against: crossing
prediction from body pose alone, as in **Fang & López (IV 2018)**. If the fused
model wins, the win has to be attributable to the extra inputs and not to a
bigger backbone or a longer run. That is the point of holding everything else
identical.

---

## Layout

```
src/pif/
  __init__.py          version, and FPS = 30 (both datasets were shot at 30 fps;
                       every "lead time in seconds" converts through this)
  taxonomy.py          the four TITAN-style label groups, and the mappings from
                       JAAD/PIE behaviour strings into them
  config.py            YAML -> dataclasses. CLI flags > config file > defaults
  device.py            one place that decides device, AMP dtype, TF32, batch
                       size and worker count. Nothing else calls torch.cuda.*
  metrics.py           average precision, ROC AUC, precision/recall/F1/miss
                       rate, ADE/FDE, gesture accuracy - hand-implemented so
                       sklearn is not a dependency in two install paths
  data/
    schema.py          dataset-neutral PedTrack / TrafficObject / VideoAnnotation
    jaad.py            JAAD parser: five XML trees per video
    pie.py             PIE parser: reads the three zips without unpacking them
    carla.py           CARLA parser: reads the recordings collect_carla.py
                       writes, into exactly the same structures. No carla
                       import anywhere in it
    windows.py         cuts tracks into 16-frame windows with a look-ahead label.
                       This is where "1.5 seconds early" is actually defined
    synthetic.py       fabricated sequences. Plumbing only, not data
    build.py           DataConfig -> (train windows, val windows) for all three
                       sources, with the dataset-specific splits
    dataset.py         torch Dataset / DataLoader / Batch
  features/
    pose.py            COCO-17 interchange format, MediaPipe adapter, encode_pose
    context.py         the 23 scene-context channels
    gesture.py         behaviour strings -> taxonomy class ids, with ignore
  models/
    transformer.py     the hand-written pre-norm temporal transformer
    heads.py           gesture head, intent head, trajectory GRU
    fusion.py          IntentForecaster, PoseOnlyBaseline, ExportWrapper
  train/
    losses.py          multi-task loss (intent BCE + masked trajectory L2 +
                       four masked gesture cross-entropies)
    loop.py            training and evaluation loop, AMP and timing kept explicit
  export/
    onnx.py            export, plus the round trip that compares numbers

scripts/
  check_gpu.py         run this first: torch/CUDA versions, VRAM, bf16, TF32,
                       and a matmul benchmark
  fetch_external.sh    clone the JAAD and PIE reference repos into external/
  fetch_carla.sh       download and unpack the CARLA simulator, 8.3 GB
  collect_carla.py     record scripted crossing episodes from a running CARLA
                       server. The ONE script that runs in .venv-carla
  parse_annotations.py parse the real XML and print what came out. No torch
  smoke_test.py        the whole path once, on synthetic data
  train.py             train one model
  run_ablation.py      fused vs pose-only, multiple seeds
  evaluate.py          score a checkpoint, and print the operating curve
  export_onnx.py       export to ONNX and verify the round trip
  extract_pose.py      the only script that needs the video clips
  _common.py           shared argument parsing, so every script takes the same
                       hardware flags

configs/
  fusion.yaml          the full model
  pose_only.yaml       the ablation target

tests/                 58 tests: taxonomy, features, windowing, metrics, and
  fixtures/carla/      the CARLA parser against a small committed fixture
```

---

## Prerequisites

- Linux x86-64 (this is what the lock file targets)
- Python 3.11
- An NVIDIA GPU is optional. Everything runs on CPU, just slower
- ~4 GB of disk for the environment, ~620 MB more for `external/`
- Git, and `curl` or `wget`

Two install paths. Pick one.

### Two environments, and why

There are two python environments in this repository and they are deliberately
kept apart:

| Environment | Python | Holds | Used by |
| --- | --- | --- | --- |
| `.pixi/envs/default` | 3.11 | torch, mediapipe, onnx, numpy | everything except one script |
| `.venv-carla` | system 3.12 | `carla` and numpy, nothing else | `scripts/collect_carla.py` only |

They exist separately because the `carla` package cannot be installed into the
pixi environment. The reason is worth knowing before you spend an afternoon on
it. A python wheel for Linux carries a *manylinux* tag, which is a promise
about the oldest glibc it will run against — glibc is the C library every Linux
program links to, and `manylinux_2_31` means "needs glibc 2.31 or newer". The
CARLA 0.9.16 wheel is tagged `manylinux_2_31`. The conda-forge python that pixi
installs declares itself compatible with `manylinux_2_28` and no higher, so pip
looks at the wheel, decides the interpreter is too old, and refuses it. It is
not actually too old: the system glibc here is 2.39, comfortably newer than
either number. The tag mismatch is a bookkeeping disagreement between
conda-forge and the wheel, not a real incompatibility. There is also no pip in
the pixi environment to argue with.

So: `pip install carla` into the pixi environment will fail, and forcing it
past the check is not worth the risk of a mismatched runtime.

**The two halves never share a process.** `collect_carla.py` imports `carla`
and numpy and nothing else — no torch, no mediapipe, no `pif`. It writes files.
Everything on the training side reads those files. That is the entire
interface, and it is why the split costs nothing.

Setting up the CARLA side, if it is not already there:

```bash
python3 -m venv .venv-carla
.venv-carla/bin/pip install carla==0.9.16 numpy
.venv-carla/bin/python -c "import carla; print('ok')"
```

### Path A — pixi (recommended, exact versions)

[pixi](https://pixi.sh) reads `pixi.toml` and `pixi.lock` and builds a fully
pinned environment. This is the path the lock file guarantees.

```bash
# install pixi itself
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL -l          # so the new PATH is picked up

cd /path/to/pedestrian-intent-forecast
pixi install            # first run downloads torch + CUDA runtime, ~2.5 GB
```

The named tasks in `pixi.toml`:

```bash
pixi run test            # pytest tests -q
pixi run smoke           # the synthetic end-to-end smoke test
pixi run export-onnx     # export + round-trip verification
pixi run parse-jaad      # parse the real JAAD annotations
pixi run train           # NOT YET RUN - see "Training" below
pixi run train-baseline
pixi run ablation
pixi run eval
```

Anything not covered by a task runs inside the environment with `pixi run`:

```bash
pixi run python scripts/check_gpu.py
pixi run python -m scripts.parse_annotations --dataset pie --limit 5
```

### Path B — venv + pip

`requirements.txt` is generated from what the code actually imports.

```bash
cd /path/to/pedestrian-intent-forecast
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CUDA build of torch first, so pip does not pick the CPU-only wheel.
# cu121 is what pixi.lock uses; the wheel bundles its own CUDA runtime, so it
# does not have to match the system CUDA install.
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e .          # makes `import pif` work from anywhere
```

CPU-only machine: skip the `--index-url` line and let `requirements.txt`
install the default torch wheel.

Verify either path:

```bash
python -c "import pif, torch; print(pif.__version__, torch.__version__)"
```

### Path C — Docker

```bash
docker build -t pif .
docker run --rm pif pytest tests -q
docker run --rm --gpus all pif python -m scripts.smoke_test
```

`external/` and `data/` are deliberately not baked into the image; mount them:

```bash
docker run --rm --gpus all \
  -v "$PWD/external:/app/external:ro" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/artifacts:/app/artifacts" \
  pif python -m scripts.parse_annotations --dataset jaad --limit 20
```

---

## Fetching the reference repositories

```bash
bash scripts/fetch_external.sh
```

This shallow-clones the two official annotation releases into `external/`:

- `external/jaad` from https://github.com/ykotseruba/JAAD
- `external/pie` from https://github.com/aras62/PIE

They carry the XML annotations, the split files, and the authors' own Python
interfaces (`jaad_data.py`, `pie_data.py`), which are what the parsers in
`src/pif/data` were written against.

**`external/` is git-ignored and must stay that way.** It is about 620 MB of
someone else's repository with its own history and its own licence. Vendoring
it into this repo would bloat every clone, and re-cloning takes a couple of
minutes. The script is the reproducible way to get it back.

It does **not** fetch the video clips. Those are a separate, much larger
download and nothing except pose extraction needs them.

Expected result:

```
JAAD  346 annotation files in external/jaad/annotations
PIE   annotations/annotations.zip present (25M)
PIE   annotations/annotations_attributes.zip present (56K)
PIE   annotations/annotations_vehicle.zip present (15M)
```

---

## Getting the JAAD and PIE data

The annotations arrive with the repos above. The **video clips** are separate.

### JAAD clips — 3.1 GB, no request form

```bash
cd external/jaad
bash download_clips.sh        # wget + unzip, from the York University server
```

or download and unzip manually from:
`http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip`
(Google Drive mirror is linked from `external/jaad/README.md`.)

Expected layout afterwards:

```
external/jaad/
  annotations/                video_0001.xml ... video_0346.xml
  annotations_attributes/     video_0001_attributes.xml ...
  annotations_traffic/        video_0001_traffic.xml ...
  annotations_vehicle/        video_0001_vehicle.xml ...
  annotations_appearance/     high-visibility subset only
  split_ids/default/          train.txt val.txt test.txt
  JAAD_clips/                 video_0001.mp4 ... video_0346.mp4   <- the download
```

### PIE clips — 74 GB

```bash
cd external/pie/annotations
bash download_clips.sh        # recursive wget from the York University server
```

or from `http://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/` (Google
Drive mirror linked from `external/pie/README.md`).

Expected layout afterwards:

```
external/pie/
  annotations/
    annotations.zip
    annotations_attributes.zip
    annotations_vehicle.zip
  PIE_clips/
    set01/video_0001.mp4
    set01/video_0002.mp4
    ...
    set06/...
```

Note the sets: PIE video numbers repeat across sets, so a PIE video id in this
repo is `set01/video_0001`, not `video_0001`.

### Do you actually need the clips?

Only for pose. This works right now with annotations alone:

```bash
python -m scripts.parse_annotations --dataset jaad --limit 20
python -m scripts.parse_annotations --dataset pie --limit 5
```

Without clips the pose channel is filled with zeros and the model runs on
context and gesture alone. Nothing crashes and no shape changes — but the
loaders say so out loud rather than quietly training a dead branch, and any
"fusion vs pose-only" comparison is meaningless until pose is real.

### Turning clips into pose features

```bash
python -m scripts.extract_pose \
    --dataset jaad \
    --clips external/jaad/JAAD_clips \
    --out data/pose_jaad.npz \
    --workers 8
```

This crops every annotated pedestrian box, runs MediaPipe Pose on the crop,
encodes it with `encode_pose`, and writes one `.npz` keyed
`"<video_id>|<ped_id>"`. Budget hours, not minutes: JAAD is 346 clips of about
10 seconds at 30 fps with several pedestrians in frame.

---

## Getting CARLA data

JAAD and PIE are gated. CARLA is not, and it hands over labels a human
annotator cannot produce. This is how the project gets real training sequences
before the dataset access lands.

### 1. Fetch and launch the simulator

```bash
scripts/fetch_carla.sh                  # 8.3 GB down, ~20 GB unpacked
~/carla/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen
```

`-RenderOffScreen` renders on the GPU but opens no window, which is what you
want over SSH or on a headless machine. The server listens on port 2000. Leave
it running in its own terminal; the first launch spends a minute or two
compiling shaders.

The fetch script resumes a part-finished download, checks the size before
unpacking, and puts the server in `~/carla` by default. Pass a directory as the
first argument to put it somewhere else. Nothing it downloads is ever
committed.

### 2. Collect episodes

```bash
.venv-carla/bin/python scripts/collect_carla.py \
    --episodes 40 --out-dir data/carla --seed 0 --scenario mixed
```

Note the interpreter: `.venv-carla`, not pixi. See "Two environments" above.

| Flag | What it does |
| --- | --- |
| `--episodes` | how many to record |
| `--out-dir` | where the `.json` and `.npz` pairs land (default `data/carla`) |
| `--seed` | pins python, the walker navigation and the traffic manager |
| `--scenario` | `straight`, `hesitate`, `occluded`, `no_cross`, or `mixed` |
| `--town` | which map, e.g. `Town10HD_Opt` |
| `--headless` | turn the server's rendering off: much faster, annotations only |
| `--no-frames` | keep the annotations, skip the RGB frames |

The four scenarios:

- **straight** — walks from the near kerb across the ego's lane. The base case.
- **hesitate** — walks to the kerb, stops for a second or two, then commits.
  This is the case a purely kinematic predictor gets wrong: velocity drops to
  zero and the intent does not.
- **occluded** — starts hidden behind a parked car and steps out into the road.
  This project documents occlusion as its main failure mode, and this is the
  only way to get a controlled supply of it.
- **no_cross** — walks along the far pavement and never enters the ego's lane.
  A negative that still moves, which is what the model has to separate from a
  real approach.

Recording runs in synchronous mode at a fixed 0.05 s step, so the same seed and
the same town give the same episodes. On exception it destroys every actor and
puts the server back into asynchronous mode; leaving it synchronous makes the
next run hang in a way that is not obvious.

### 3. What comes out

One pair of files per episode:

```
data/carla/ep_0000.json    the annotation
data/carla/ep_0000.npz     RGB frames, uint8 (T, H, W, 3), unless --no-frames
```

The JSON fills exactly the structures `src/pif/data/schema.py` defines, so
windowing, features and models do not know or care that it came from a
simulator:

| Field | Where it comes from |
| --- | --- |
| `bbox` | the walker's 3D box projected into the camera, 2D extent taken |
| `occlusion` | rays cast from the camera at points on the walker, then the existing `{none: 0, part: 1, full: 2}` codes |
| `cross` | per frame, whether the walker is standing on a driving lane |
| `crossing` / `crossing_point` | derived by the parser, JAAD's convention exactly: -1 not relevant, 0 did not cross, 1 crossed; and the frame they first stepped into the road, or -1 |
| `action`, `look`, `gesture` | **empty, on purpose.** CARLA has none of these |

Read them back with the pixi python:

```python
from pif.data.carla import CARLARecordings, lead_frames
from pif.data.windows import video_windows

db = CARLARecordings("data/carla")
video = db.load(db.video_ids()[0])
# CARLA runs at 20 Hz, JAAD and PIE at 30 fps, so a lead time in frames does
# not carry across. Convert from seconds.
windows = video_windows(video, window=16, lead=lead_frames(1.5, db.fps(video.video_id)))
```

That 20 Hz is easy to trip over. The windowing defaults in `pif.data.windows`
are in frames and were chosen for 30 fps footage; handing a CARLA recording the
default `lead=45` silently asks for 2.25 seconds of look-ahead instead of 1.5.

### 4. What you must not do with it

Do not train the gesture head on CARLA data without masking. The shared gesture
encoder falls back to a default class when a behaviour column is missing, and
on CARLA every one of those columns is missing, so an unmasked run trains three
of the four heads towards a label the simulator never produced. Use
`pif.data.carla.mask_unlabelled_gestures`, which blanks the atomic,
communicative and transportive groups and keeps only the contextual one — that
last group is derived from the crossing state, which CARLA does know exactly.

And do not report a CARLA score as a result about pedestrians. It is a result
about a simulator. The plan stands: train on CARLA, evaluate on real JAAD and
PIE once access lands, and report the two separately.

---

## Running things

### 1. Check the GPU first

```bash
python scripts/check_gpu.py
```

Sample output on the development machine (RTX 4080 Laptop, 12 GB):

```
CHECK_GPU_OUTPUT_PLACEHOLDER
```

Exit code is 0 even with no GPU — everything here runs on CPU. It is non-zero
only if torch itself is broken.

### 2. Tests

```bash
pytest tests -q
# or: pixi run test
```

```
TEST_OUTPUT_PLACEHOLDER
```

The tests cover the five things that would silently corrupt every downstream
number if they broke: the taxonomy mappings (reorder an enum and every saved
checkpoint quietly means something different), the pose and context encoders
(a feature called `torso_lean` that does not move when the torso leans is worse
than no feature), the windowing rules that make this a forecast, the
hand-written metrics (checked against worked examples with known answers), and
the CARLA parser — its crossing-point derivation, the -1/0/1 convention, that
it invents no gesture labels, and that a CARLA recording flows through the
existing windowing code untouched.

They need neither torch nor a GPU — `pif.data` deliberately keeps torch off its
import path so parsing works in a minimal environment. They also need neither
CARLA nor a running simulator: the CARLA tests read a small committed fixture
under `tests/fixtures/carla`, which is hand-built to exercise the parser and is
not a capture from a real simulation. Nothing in it is a measurement.

### 3. Parse the real annotations

No model, no torch, no video files. This is the script that proves the parsers
understand the released schema.

```bash
python -m scripts.parse_annotations --dataset jaad --limit 20
```

Real output from this repository:

```
jaad: 346 annotated videos under external/jaad
parsing 20

totals: {'videos': 20, 'pedestrians': 92, 'behavior_tracks': 27, 'traffic_objects': 0, 'ego_frames': 5370, 'scene_frames': 5370}
per-frame cross values seen: {'not-crossing': 3265, 'crossing': 1974}

windows: {"n": 589.0, "positive_rate": 0.2547, "n_tracks": 19.0, "lead_seconds": 1.5, ...}

  atomic         {'walking': 7184, 'standing': 2240}
  communicative  {'none': 7324, 'looking_at_traffic': 1656, 'phone_distracted': 444}
  contextual     {'walking_along': 5111, 'waiting_to_cross': 2240, 'crossing': 2073}
  transportive   {'none': 5139, 'carrying': 4285}
```

```bash
python -m scripts.parse_annotations --dataset pie --limit 5
```

```
pie: 53 annotated videos under external/pie
parsing 5

totals: {'videos': 5, 'pedestrians': 144, 'behavior_tracks': 144, 'traffic_objects': 254, 'ego_frames': 81838, 'scene_frames': 0}
per-frame cross values seen: {'not-crossing': 57558, 'crossing': 11269, 'crossing-irrelevant': 904}

windows: {"n": 12370.0, "positive_rate": 0.0464, "n_tracks": 141.0, "lead_seconds": 1.5, ...}

example window: set01/video_0001 ped 1_1_1 frames 1013..1028 intent=0 lead=45
  PIE human intention_prob for this track: 0.967
```

Reading those two: JAAD tags behaviour only on ids ending in `b`, which is why
92 pedestrians produce 27 behaviour tracks. PIE tags all of them, has boxed
traffic objects where JAAD has scene flags, and no `transportive` labels at
all. `crossing-irrelevant` is PIE's third value — the person crossed, but not
in front of the ego vehicle, which for a planner is a negative.

### 4. Synthetic smoke test

```bash
python -m scripts.smoke_test              # ~8 epochs, seconds on a GPU
python -m scripts.smoke_test --epochs 4 --tracks 200 --device cpu
```

It walks the whole path and prints what each stage produced: generate windows
-> build tensors -> build the model -> train briefly -> evaluate -> export ONNX
-> verify the round trip. If it runs clean, the plumbing is intact.

```
SMOKE_OUTPUT_PLACEHOLDER
```

Again: **synthetic**. The generator was written with context cues that pose
does not carry, because that is the hypothesis the project is built on. A
model doing well on it proves the fusion path can carry information. It says
nothing about real pedestrians.

### 5. ONNX export and verification

```bash
python -m scripts.export_onnx                                  # fresh weights
python -m scripts.export_onnx --checkpoint artifacts/fusion.pt # a trained one
```

Exporting is the easy half. The half that catches real bugs is reloading the
graph in onnxruntime and comparing the outputs against torch **on inputs the
exporter never traced**, at three different batch sizes. A traced graph that
baked in a batch size or silently dropped a branch passes a shape check and
fails this. Exit code is non-zero on disagreement.

Measured in this repository:

```
ONNX_OUTPUT_PLACEHOLDER
```

Batch stays a dynamic axis so one file serves a single pedestrian and a batched
offline run. Time does not: the window length is part of the model definition,
because the positional table is sized to it.

This ONNX file is the artifact a Jetson-class deployment would consume through
TensorRT. On-device latency is not benchmarked here.

---

## Training and the ablation — NOT YET RUN

**Neither of these has been run on real data.** They are written, they are
tested on synthetic data, and they are waiting on the JAAD/PIE video clips.
Running them on `source: synthetic` produces pipeline verification, not
results.

### Train one model

```bash
# once external/jaad and data/pose_jaad.npz exist:
python -m scripts.train \
    --config configs/fusion.yaml \
    --source jaad \
    --data-root external/jaad \
    --pose-cache data/pose_jaad.npz \
    --batch-size 512
```

Produces, in `artifacts/`:

- `train_fusion.json` — full per-epoch history: train loss and its three parts
  (intent, trajectory, gesture), learning rate, seconds, samples/s, peak VRAM,
  and validation metrics for every epoch
- `fusion.pt` — checkpoint, with the model config and data config embedded so
  `scripts/evaluate.py` and `scripts/export_onnx.py` can rebuild the model
  without being told which one it was

Validation metrics per epoch: **average precision** (the headline — the
positive class is a minority, and accuracy on an imbalanced binary problem
mostly reports the class prior), ROC AUC, accuracy, precision, recall, F1,
**miss rate** (crossings the model did not flag — for a planner, a late brake),
**false alarm rate** (phantom stops), **ADE/FDE** for the trajectory head in
normalised image units, and mean gesture accuracy across the supervised groups.

### The ablation

```bash
python -m scripts.run_ablation \
    --config configs/fusion.yaml \
    --source jaad \
    --data-root external/jaad \
    --pose-cache data/pose_jaad.npz \
    --repeats 3
```

Trains the fused model and the pose-only baseline on identical windows, splits,
optimiser, schedule, epochs and seeds — three seeds per arm by default — and
writes `artifacts/ablation.json`. It prints a table of AP (mean ± std across
seeds), AUC, F1 and miss rate per arm, plus the fused-minus-baseline delta in
absolute and relative AP.

That delta is the number the project is actually about: does knowing where the
person is standing and what their hands are doing buy you anything over knowing
how their body is arranged. On synthetic data the answer is rigged by
construction, and the script prints a caveat saying exactly that.

### Evaluate a checkpoint

```bash
python -m scripts.evaluate --checkpoint artifacts/fusion.pt --source jaad
```

Prints the metric line plus an operating curve from threshold 0.1 to 0.9,
because a single F1 at 0.5 hides the trade-off a planner has to pick a point
on: every missed crossing is a late brake, every false alarm is a phantom stop.

---

## Tuning for your hardware

By default the code picks for you. `src/pif/device.py` is the only place that
decides:

| Detected total VRAM | Auto batch size |
| --- | --- |
| >= 24 GB | 1024 |
| >= 16 GB | 512 |
| >= 10 GB | 256 |
| >= 6 GB | 128 |
| CPU or less | 64 |

Workers default to `min(8, os.cpu_count() - 2)`. Windows are small dense float
arrays, not JPEGs, so past a handful of workers you are only adding IPC.

These are deliberately conservative — the model is tiny (about 0.5M parameters)
and the ceiling is not memory. **On a bigger GPU, raise both**:

```bash
# e.g. an A100 80 GB or an RTX 6000
python -m scripts.train --config configs/fusion.yaml \
    --batch-size 4096 --num-workers 12
```

or set them in the config file:

```yaml
batch_size: 4096
num_workers: 12
```

Other switches:

- `--no-amp` turns off mixed precision. On by default on CUDA: bf16 where the
  card supports it (Ampere and later), fp16 otherwise, with a GradScaler only
  in the fp16 case — bf16 has fp32's exponent range so scaling it is pointless
  overhead.
- `--compile` runs `torch.compile`. Off by default: on a model this small the
  compile takes longer than a short run. Worth it for a long real-data run.
- `--device cuda:1` to pin a specific GPU.

If you raise the batch size a lot, raise the learning rate with it — the
default `3e-4` was chosen for batches in the hundreds, and the loop uses a
linear warmup then cosine decay.

### Out of memory

Symptom:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 11.99 GiB total capacity; 9.21 GiB already allocated; ...)
```

Fixes, in the order to try them:

1. `--batch-size 128` (or 64). Halve until it fits.
2. `--num-workers 2`. Workers hold pinned host memory, which is not VRAM but
   can starve the machine on a laptop.
3. Check nothing else is using the card: `nvidia-smi`. A browser or a desktop
   compositor can hold a gigabyte or more.
4. `--no-amp` is *not* the fix — mixed precision uses less memory, not more.

Every run prints peak VRAM, so after one epoch you know how much headroom you
have and can raise the batch size with evidence instead of guessing.

---

## Troubleshooting

**`torch.cuda.is_available()` is False**

Run `python scripts/check_gpu.py` and look at the `torch cuda build` line. If it
says `cpu-only wheel`, pip installed the CPU build — reinstall with the
`--index-url https://download.pytorch.org/whl/cu121` line above. If it shows a
CUDA version but `cuda available` is still False, the driver is the problem:
check `nvidia-smi` runs at all. The torch wheel brings its own CUDA runtime, so
the system CUDA toolkit version does not have to match.

**`ImportError: cannot import name ... from partially initialized module`**

Should not happen any more (`pif.features` no longer imports `pif.data` at
runtime), but if you add an import from `pif.features.*` back into
`pif.data.schema` it will return. The dependency direction is
`taxonomy <- features <- data <- models/train/export`, one way only.

**mediapipe will not install, or `import mediapipe` segfaults**

MediaPipe's released wheels are built against the numpy 1.x ABI, which is why
`requirements.txt` pins `numpy<2`. If pip pulled numpy 2 anyway:
`pip install "numpy<2"`. On Python 3.12+ there may be no wheel at all — use
3.11. Only `scripts/extract_pose.py` needs mediapipe; the tests, the smoke
test, the ONNX export and the annotation parsers all run without it.

**`ImportError: libGL.so.1` or `libgomp.so.1` from cv2 or mediapipe**

Missing system libraries, not Python ones:
`sudo apt install libglib2.0-0 libgomp1`. The Dockerfile installs both.

**`no JAAD annotations under external/jaad`**

`bash scripts/fetch_external.sh`. If it already ran, check
`external/jaad/annotations` has 346 `video_*.xml` files — a shallow clone that
was interrupted leaves the directory there but short.

**`no such JAAD split: .../split_ids/default/train.txt`**

The split files ship with the JAAD repo. If they are missing, `build_jaad`
falls back to a deterministic 75/25 split by video number and carries on — but
you are then not on the published protocol, so say so in any comparison.

**`data source 'jaad' produced 0 train / 0 val windows`**

The annotations parsed but nothing survived windowing. Usual causes: pointing
`--data-root` at the wrong directory, or `window`/`lead` set so long that no
track is long enough. Run `scripts/parse_annotations.py` first — it prints the
window count directly.

**ONNX round trip fails**

Look at `max|dp|` in the report. A large disagreement usually means a branch
was traced away — check the model was in `.eval()` (dropout is active in train
mode and the two will never agree) and that the export ran on CPU in fp32, not
inside an autocast context.

---

## References

- Malla, Dariush & Choi. **TITAN: Future Forecast using Action Priors.** CVPR
  2020. The source of the hierarchical, parallel-group action taxonomy this
  repo's label set is modelled on.
- Rasouli, Kotseruba & Tsotsos. **Are They Going to Cross? A Benchmark Dataset
  and Baseline for Pedestrian Crosswalk Behavior.** ICCVW 2017. The JAAD
  dataset and its annotation schema.
- Rasouli, Kotseruba, Kunic & Tsotsos. **PIE: A Large-Scale Dataset and Models
  for Pedestrian Intention Estimation and Trajectory Prediction.** ICCV 2019.
  The PIE dataset, its OBD telemetry, and the `intention_prob` rating.
- Fang & López. **Is the Pedestrian going to Cross? Answering by 2D Pose
  Estimation.** IV 2018. The pose-only crossing classifier that the ablation's
  baseline arm stands in for.
- Sun, Xiao, Liu & Wang. **Deep High-Resolution Representation Learning for
  Human Pose Estimation.** CVPR 2019. HRNet, the estimator this repo is
  structured to swap in for MediaPipe.

Any figure attributed to those papers is theirs, measured on their data. No
number in this repository is a result on real pedestrian video.

## Licence

The code here is mine. JAAD and PIE are the property of their authors and carry
their own licences — see `external/jaad/LICENSE` and `external/pie/LICENSE`
after running the fetch script. Neither dataset is redistributed here.
