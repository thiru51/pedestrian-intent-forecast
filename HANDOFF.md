# Handoff

For whoever picks this up next, including me in three months.

Read `README.md` for how to run things and `END_GOAL.md` for why any of this
exists. This file is the short version of what is real, what to do first, and
which decisions were deliberate.

---

## What is actually implemented and checked

All of it runs. None of it has seen a real pedestrian video frame.

- **Both annotation parsers work on the real released data.** JAAD's five XML
  trees, PIE's three zip archives read without unpacking. Verified by
  `scripts/parse_annotations.py` against `external/`.
- **Windowing** cuts tracks into 16-frame observations with a label 45 frames
  (1.5 s) later, and enforces the two rules that keep it a forecast.
- **Features**: 59-dim pose, 23-dim scene context, 4 gesture groups.
- **Model**: hand-written pre-norm temporal transformer, gesture head, intent
  head, trajectory GRU conditioned on the intent summary. The pose-only
  baseline is the same class with branches disabled.
- **Training loop** with AMP, TF32, warmup + cosine, per-epoch JSON history.
- **Metrics** hand-implemented and tested against worked examples.
- **ONNX export** with a verified numerical round trip.
- **CARLA collector and parser.** `scripts/collect_carla.py` records scripted
  crossing episodes from a CARLA server; `src/pif/data/carla.py` reads them
  into the same schema the JAAD and PIE parsers fill. Both have been run: one
  collection against a live CARLA 0.9.16 server produced 19 episodes of 20
  attempted, and those parse and window cleanly. The boxes were checked by
  drawing them back onto the captured frames.
- **59 unit tests**, all passing, none of which need torch, a GPU, CARLA or a
  running simulator.

What is **not** real: any accuracy number about anything. No model has been
trained on the CARLA episodes yet, the pose branch has never seen real video,
and no latency has been measured on a Jetson. The only end-to-end runs so far
are on synthetic sequences fabricated by `src/pif/data/synthetic.py`, which
exist to exercise the code path and are labelled as such everywhere they
appear.

---

## Two environments. Read this before installing anything

There are two python environments and they must stay separate.

| Environment | Python | Holds | Runs |
| --- | --- | --- | --- |
| `.pixi/envs/default` | 3.11 | torch, mediapipe, onnx | everything except one script |
| `.venv-carla` | system 3.12 | `carla`, numpy | `scripts/collect_carla.py` only |

**Do not try to pip install carla into the pixi environment.** It will not
work, and the error is misleading. The CARLA 0.9.16 wheel is tagged
`manylinux_2_31`, which is a claim about the oldest glibc (the C library every
Linux program links against) it will run on. The conda-forge python that pixi
installs declares support only up to `manylinux_2_28`, so pip decides the
interpreter is too old and refuses the wheel. The system glibc here is 2.39, so
nothing is actually too old — it is a tagging disagreement between conda-forge
and the wheel. There is also no pip inside the pixi environment to argue with.

The two halves never share a process. `collect_carla.py` imports `carla` and
numpy and nothing else: no torch, no mediapipe, not even `pif`. It writes JSON
and npz files, and the training side reads them. Keep it that way — the moment
the collector imports something from the training side, the split stops working
and the whole CARLA path becomes uninstallable.

---

## Getting CARLA running

```bash
scripts/fetch_carla.sh                        # 8.3 GB, resumable
~/carla/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen   # leave this running
.venv-carla/bin/python scripts/collect_carla.py --episodes 40 --out-dir data/carla
```

Then read the episodes back with the pixi python, which is the one that has
`pif`. See the README section "Getting CARLA data" for the flags, the four
scenarios, and what the output files contain.

Two things to know before trusting the output:

1. **CARLA runs at 20 Hz, JAAD and PIE at 30 fps.** A lead time in frames does
   not carry across. Use `pif.data.carla.lead_frames(seconds, fps)`; the
   `lead=45` default means 1.5 s on JAAD and 2.25 s on CARLA.
2. **CARLA has no gestures.** The `action`, `look` and `gesture` columns come
   back empty and the parser refuses to invent them. Call
   `pif.data.carla.mask_unlabelled_gestures` before training or three of the
   four gesture heads will be trained against a fallback default the simulator
   never produced.

The honest framing, which belongs in any writeup: CARLA gives perfect labels
but is synthetic. Train on CARLA, evaluate on real JAAD and PIE once access
lands, and report the two numbers separately. A CARLA score is a statement
about a simulator, not about pedestrians.

---

## Exactly what to run first, once the data lands

Assume the JAAD clips have just finished downloading.

```bash
# 0. Sanity: environment and hardware
pixi install
pixi run python scripts/check_gpu.py
pixi run test

# 1. Confirm external/ is intact and the parser still agrees with it
pixi run python -m scripts.parse_annotations --dataset jaad --limit 20

# 2. Confirm the clips are where you think they are
ls external/jaad/JAAD_clips | head
ls external/jaad/JAAD_clips | wc -l        # expect 346

# 3. Pose extraction on a handful of clips FIRST. Do not launch the full run
#    before you have seen one work - the full one is hours.
pixi run python -m scripts.extract_pose \
    --dataset jaad --clips external/jaad/JAAD_clips \
    --out data/pose_jaad_smoke.npz --limit 3 --workers 3

# 4. Look at what came out. Non-zero confidences on most frames means
#    MediaPipe found people. All zeros means the crops are wrong.
pixi run python -c "
import numpy as np
d = np.load('data/pose_jaad_smoke.npz')
print(len(d.files), 'tracks')
a = d[d.files[0]]
print(a.shape, 'mean confidence', a[:, 34:51].mean())
"

# 5. Now the full extraction. Hours. Run it in a tmux session.
pixi run python -m scripts.extract_pose \
    --dataset jaad --clips external/jaad/JAAD_clips \
    --out data/pose_jaad.npz --workers 8

# 6. A short training run to check the real-data path end to end
pixi run python -m scripts.train --config configs/fusion.yaml \
    --source jaad --data-root external/jaad \
    --pose-cache data/pose_jaad.npz --epochs 2

# 7. The real run, then the ablation
pixi run python -m scripts.train --config configs/fusion.yaml \
    --source jaad --data-root external/jaad --pose-cache data/pose_jaad.npz
pixi run python -m scripts.run_ablation --config configs/fusion.yaml \
    --source jaad --data-root external/jaad \
    --pose-cache data/pose_jaad.npz --repeats 3
```

Step 4 is the one people skip and regret. If the pose cache is all zeros,
everything downstream still runs, still produces numbers, and those numbers are
a context-only model wearing a fusion label.

**The first thing to check in step 6's output** is the printed positive rate.
JAAD gave 25.5% positive on the first 20 videos; PIE gave 4.6%. If it comes out
near 0 or near 1 on the full set, the label or the split is wrong, and no
metric below that line means anything.

---

## Known issues and rough edges

**No pose on real data yet.** The single biggest gap. Until the clips are
processed, `use_context=True` fusion is really context + gesture, and the
ablation against a pose-only baseline compares something against near-nothing.

**`transportive` is barely supervised.** JAAD annotates carried objects only in
the appearance file, which covers the high-visibility subset. PIE does not
annotate it at all. The head exists so the taxonomy stays faithful to TITAN and
the loss ignores unlabelled groups — but on PIE it trains on zero examples.
Options: drop the head for PIE runs, or pretrain it on TITAN. Not decided.

**The ego corridor is a fixed wedge**, not a calibrated projection: about one
lane wide at the bottom of the frame, narrowing towards a fixed horizon row.
Both datasets are forward-facing dashcam with a similar mount so it is a
reasonable stand-in, but PIE ships real camera parameters in
`external/pie/camera_params` and the curb-distance feature should use them.

**`PHONE_DISTRACTED` cannot be derived from the behaviour columns.** It comes
only from JAAD's appearance file. On PIE that class will never fire.

**PIE's `contextual` labels are lopsided.** Because windows truncate at the
crossing point, almost no observed frame carries `cross="crossing"`, so the
`CROSSING` class is nearly empty inside windows (12 occurrences in the 5-video
sample against 132,931 `WAITING_TO_CROSS`). That is correct behaviour for a
forecasting task, but the gesture loss on that group is close to degenerate and
its accuracy number will look impressive for the wrong reason.

**The collector has only been run on Town10HD_Opt.** One collection of 20
episodes worked there, 19 of which were usable. Other towns lay their pavements
out differently and the staging may need another pass. Run
`--episodes 1 --scenario straight` in a new town and look at the JSON before
launching forty of them.

**About one episode in twenty comes out unusable and is discarded.** Both known
causes are about where in the town the episode landed: the walker is staged
behind a row of parked cars and never gets a clear frame, or it cannot move.
`unusable()` catches both and says which. If the discard rate climbs, that is
the thing to look at, not the labels.

**`summarise()` reports the lead time in seconds using `FPS = 30`.** On a CARLA
recording at 20 Hz that number is a third too small. The windows themselves are
correct - only that one printed field is wrong. It is shared with the JAAD and
PIE path, so it was left alone rather than special-cased.

**The occlusion measurement depends on `World.cast_ray`.** If the CARLA build
in use does not have it, the collector says so and records
`occlusion_measured: false`, and every walker is written as fully visible. Do
not read the occlusion column off a recording with that flag set to false.

**No CI.** Tests are run by hand.

**`pixi.toml` lists `scipy` and `tqdm`, which nothing imports.** Harmless, but
they are not in `requirements.txt` because that file was generated from the
actual imports. Worth pruning next time the lock file is regenerated.

**`OCCLUSION` in `src/pif/data/schema.py` was unused** — the JAAD and PIE
parsers inline the same mapping. The CARLA parser uses the constant, so it is
no longer dead, but the other two were left alone rather than churned.

---

## Design decisions, and why

**Why a look-ahead label instead of a per-track "did they cross" label.**
A track-level label makes every window in the track a positive, including
windows from ten seconds earlier where nothing was decided yet. That trains a
model to recognise pedestrians who eventually cross, not pedestrians about to
cross. The look-ahead formulation asks a question with a definite answer at a
definite moment.

**Why windows stop at the crossing point.** Once the person is in the road, the
answer is visible, and a "forecast" that includes those frames is a detector
with a good score. JAAD's own reference pipeline truncates at the same frame,
so this also matches how the dataset authors intended it to be used.

**Why the split is by track, never by window.** Consecutive windows from one
track overlap by 12 of 16 frames. Splitting at window level puts near-copies of
the same sample on both sides, and every number that comes out is inflated.
`split_windows` splits on track id, and the JAAD and PIE builders use the
datasets' own video-level and set-level splits.

**Why the CARLA parser fills the same schema instead of its own format.**
A simulator recording and a JAAD video have nothing in common at the file
level, but they answer the same questions: where was the person, were they on
the road, how much of them could you see. Normalising CARLA into `PedTrack` and
`VideoAnnotation` means windowing, the feature encoders, the model and the
training loop are untouched, and a CARLA-trained model can be evaluated on JAAD
without a second code path. A parallel format would have been quicker to write
and would have doubled everything downstream.

**Why the CARLA parser leaves the gesture columns empty.**
It would have been easy to write `action="walking"` whenever the walker's speed
is above zero. That is a real fact about the simulation, but it is not the
label the gesture head is supposed to learn, and once it is in the file nobody
downstream can tell a measured label from a filled-in one. The columns stay
empty and `mask_unlabelled_gestures` makes the loss skip them. The cost is that
the gesture branch cannot be trained on CARLA at all, which is exactly the
thing to say out loud: it is why real JAAD and PIE access still matters.

**Why the walkers are not driven by `WalkerAIController`.**
That is the normal way to move a CARLA pedestrian, and it was the first thing
tried. It navigates the pedestrian navmesh, and the navmesh does not cover the
carriageway except where the map has drawn a crosswalk, so a walker told to go
to the far pavement routes to the nearest crossing instead of stepping into the
road in front of the ego. In testing it produced zero crossings: the walker
strolled off down the pavement every time. Setting the walking direction
directly each frame puts the crossing exactly where the script says it is, and
makes the hesitation exact - speed to zero on one frame, back up on another.
The cost is that the walkers do not avoid obstacles, which is why `unusable()`
throws away episodes where one walked into something.

**Why occlusion comes from ray casts and not from pixel counting.**
Counting how many pixels of the walker are visible needs a second render and a
reference for how many pixels an unoccluded walker of that size at that
distance would have covered, and that reference is a guess. Casting nine rays
from the camera at the walker's bounding box and counting how many arrive
unblocked is a direct geometric measurement, so the partial-occlusion code is
real. If the build has no `cast_ray`, the collector records that it did not
measure rather than emitting a plausible middle value.

**Why the transformer is hand-written.** `nn.TransformerEncoder` carries mask
and nested-tensor branching that ONNX export has historically handled badly,
and clean export is a hard requirement here. At T=16 the whole attention is a
16x16 matrix, so there is nothing to gain from the library version.

**Why pre-norm.** At 2-4 layers either trains, but pre-norm does not need a
warmup schedule to stay stable. One less thing to tune.

**Why learned positional embeddings, not sinusoids.** T is fixed at 16 and the
steps are 33 ms apart. There is no length generalisation to preserve, so a
learned table fits the fixed geometry better.

**Why the gesture logits are detached before the temporal branch.** The gesture
head is supervised directly against the taxonomy. If the intent loss could also
push it, it would drift into being an unconstrained feature extractor whose
outputs no longer mean what the class names say — and the gesture accuracy
number would become meaningless while still looking fine.

**Why four parallel softmaxes instead of one.** That is TITAN's point: the
groups are independent, a person can be walking and waving and pushing a pram
at once. One joint class list would be 3x5x4x4 = 240 mostly-empty classes.

**Why the trajectory head is conditioned on the intent summary rather than
concatenated per step.** Initialising the GRU's hidden state gives the
recurrence a starting belief about what the person is going to do, and leaves
the per-step inputs purely kinematic. Concatenating would mix the two at every
step and make it harder to say what the conditioning contributed.

**Why the trajectory head sees velocities, not positions.** Absolute image
position is the context branch's job; feeding it again here just teaches the
GRU the camera framing.

**Why metrics are hand-implemented instead of sklearn.** The package already
depends on numpy and torch and nothing else numeric. AP, AUC and displacement
error are about sixty lines, and the tests check them against worked examples.
One fewer dependency to pin across two install paths.

**Why average precision is the headline.** The positive class is a minority —
4.6% on the PIE sample. Accuracy on an imbalanced binary problem mostly reports
the class prior.

**Why `pif.data` does not import torch.** Parsing annotations and building
windows is pure numpy, so `scripts/parse_annotations.py` works in a minimal
environment. Import `pif.data.dataset` explicitly when you want tensors.

**Why `pif.features` does not import `pif.data` at runtime.** It used to, and
that closed an import cycle: `pif.data.__init__` imports `pif.data.windows`,
which imports `pif.features.context`, which imported `pif.data.schema`. Any
program that reached `pif.features.*` first — including
`python -m scripts.train` and `python -m scripts.export_onnx` — crashed with a
partially-initialised-module error. Fixed by moving the JAAD ego-speed proxy
table into `pif.features.context` (it is a feature-encoding choice, not a fact
about the file format) and putting the type-only imports under `TYPE_CHECKING`.
The dependency direction is now one way:
`taxonomy <- features <- data <- models / train / export`. Keep it that way.

**Why torch comes from PyPI and not the pytorch conda channel.** The cu121
wheel bundles its own CUDA runtime, so the environment does not have to agree
with whatever CUDA is installed on the host. That removes the most common
"works on my machine" failure. The earlier version of `pixi.toml` used the
conda channel and needed the system CUDA to match.

**Why synthetic data exists at all, and why it is not a shortcut.** Without it
nothing could be run, tested or profiled until a 74 GB download completed. The
generator drives pose and context from one latent state, with weak noisy cues
in pose and stronger ones in context — which is the project's hypothesis
written into the data. That makes it useful for checking wiring and
*worthless* as evidence. Every script that touches it prints a warning.
