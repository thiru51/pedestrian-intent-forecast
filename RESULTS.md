# Results

Run 3 September 2026. Every number here comes from `artifacts/ablation.json` and
`results/ablation_carla.log`, produced by `scripts/run_ablation.py`. Nothing is estimated.

## Read this first

**These are numbers on simulated pedestrians.** The labels are exact, which is the whole
point of using a simulator, but the people are not real: they walk on a navmesh, they are
lit by a game engine, and they never do the things a gesture head exists to catch. This is
a measurement of the pipeline on synthetic video with ground-truth labels. It is **not**
accuracy on real pedestrians and must never be quoted as one. The run that would mean that
is `--source jaad` with a pose cache, and it needs the clips, which are still access-gated.

## Real pedestrians, with pose: the headline result

The clips were downloaded (2.9 GB) and MediaPipe run over every annotated pedestrian
box: **686 tracks, 132,700 pedestrian-frames**, cached in `data/pose_jaad.npz`. So the
architecture this project is actually about -- pose fused with gesture and scene
context -- has now been evaluated on real pedestrians. 3 seeds, 30 epochs, split by
video.

| arm | AP | AUC |
|---|---|---|
| pose_only | 0.3849 +-0.0095 | 0.5461 |
| **fusion** | **0.6326** +-0.0183 | **0.7655** |

**+0.2477 AP, a 64% relative gain**, on real data. Per-seed AP was [0.654, 0.621, 0.623]
for fusion against [0.394, 0.385, 0.375] for pose_only; the ranges do not overlap.

### The CARLA occlusion result does not reproduce

This is the part worth reading. The CARLA section below reports that the fusion gain is
**2.5x larger on occluded pedestrians** than on clear ones, and calls that "the result
worth keeping". On real data it is not there:

| | occluded | clear | occluded / clear |
|---|---|---|---|
| CARLA (synthetic) | +0.2086 | +0.0831 | **2.5x** |
| JAAD (real) | +0.2001 +-0.105 | +0.2574 +-0.017 | **0.8x** |

On JAAD the occluded gain is if anything *smaller* than the clear gain, and the
difference (+0.057) sits well inside the seed spread on that subset (+-0.105 on n=126).
So the honest statement is not "the opposite is true" -- it is:

> **On real pedestrians there is no measurable occlusion-specific advantage.** Fusion
> helps a great deal overall, and the claim that it helps *specifically* when the
> pedestrian is occluded is a property of the CARLA data, not a property of the method.

Two candidate reasons, neither tested here. CARLA's occlusion labels are exact and
CARLA's occluders are large scripted vehicles, so "occluded" there is a cleaner and more
extreme condition than JAAD's annotation. And the CARLA pose came from rendered frames
where MediaPipe either works or fails cleanly; on real video it degrades gradually, so
the pose_only baseline may already be partly broken on the clear subset too.

**What this changes.** The occlusion claim should not be quoted without the dataset
attached to it. The overall fusion result is stronger than it was -- it now holds on
real pedestrians, not only in simulation -- but the mechanism story does not survive the
move.

---

## Real pedestrians: the first JAAD numbers

Everything below this section is CARLA. This part is not.

**What made it possible.** RESULTS.md previously said the real-data run was "blocked
on JAAD/PIE clip access". That was half right. The 2.9 GB of video clips are indeed a
separate download and are still not here -- but the **annotations are not gated at
all**, and `pif.data.jaad` says so in its own docstring: "Only the XML is needed here."
346 annotated videos with JAAD's own train/val split ids were already on disk.

**What that buys, and what it does not.** No clips means no pose channel. So this is
not the pose-plus-gesture-plus-context experiment; it is everything except pose. The
existing `fusion` vs `pose_only` ablation is therefore **meaningless on this data** --
run it here and `pose_only` returns AUC exactly 0.5000 with recall 1.0000, a
null-input model that predicts every window positive. Quoting its +0.145 AP gap as a
real-data result would be reporting the difference between a model and an empty one.

`scripts/gesture_ablation.py` runs the comparison that is actually available: both arms
without pose, differing in `use_gesture` and nothing else.

### Does the gesture head help?

This is the question the project has never been able to ask. CARLA supplies no gesture
annotations, so on synthetic data those label groups are masked and the head is never
scored. JAAD annotates `hand_gesture`, `nod` and `reaction`.

5 seeds, 30 epochs, 6,285 train / 775 val windows, split by video, val positive rate
0.341:

| arm | AP | AUC |
|---|---|---|
| context | 0.6343 +-0.0536 | 0.7677 |
| context+gesture | 0.6142 +-0.0316 | 0.7550 |

**dAP -0.0201.** And the per-seed deltas are what matter:

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| gesture - context | **+0.1160** | **-0.0988** | +0.0199 | -0.0700 | -0.0676 |

The sign flips across seeds and the spread (+-0.054 on the context arm alone) is more
than twice the mean difference. **The gesture head has no measurable effect here.**
That is a null result, not a negative one -- there is no evidence it helps and none
that it hurts.

### What this does and does not establish

**It does** give the repository its first number on real pedestrians. Context plus
trajectory reaches **AP 0.634 against a positive rate of 0.341**, so the model learns
real crossing intent from real annotations, on a by-video split, with no pose at all.

**It does not** test the gesture head as designed. The head consumes
`POSE_DIM + CONTEXT_DIM` per frame, and the pose half is zeros here. A head starved of
its main input showing no effect is weak evidence about the head. Settling it needs the
clips and a pose cache; that run is still not done.

**It also does not** transfer to the CARLA numbers below. Different dataset, different
crossing definition, different label noise, and pose present there but absent here.
The two sets of numbers should never be put in the same table.

---

## What was run

| | |
|---|---|
| Data | 222 CARLA episodes, Town10HD_Opt, 20 Hz |
| Split | **by episode** -- 141 train / 50 val, asserted non-overlapping |
| Windows | 1,691 train / 684 val (positive rate 0.351 / 0.542) |
| Prediction lead | 45 frames = 2.25 s ahead |
| Arms | `fusion` (pose + gesture + scene context) vs `pose_only` |
| Seeds | 3 per arm, identical data and schedule |
| Metrics from | **final epoch, no best-epoch selection** |
| Pose | MediaPipe BlazePose on saved RGB frames |
| Hardware | RTX 4080 Laptop, bf16, 12.9 s total for all six runs |

The split is by episode, not by window. Windows drawn from the same episode overlap in
time and share frames; splitting by window would leak them across train and validation and
inflate every number below. `build_carla` asserts the episode sets do not intersect.

## The headline

| arm | params | AP | AUC | precision | recall | F1 | miss rate |
|---|---|---|---|---|---|---|---|
| pose_only | 435,424 | 0.8298 +-0.0126 | 0.7589 | 0.7958 | 0.6532 | 0.7174 | 0.3468 |
| **fusion** | 458,192 | **0.9432 +-0.0058** | **0.9296** | **0.8795** | **0.8122** | **0.8444** | **0.1878** |

**Fusion beats pose-only by +0.1134 AP, a 13.7% relative gain**, for 5% more parameters.
Per-seed AP was [0.9455, 0.9366, 0.9474] for fusion against [0.8421, 0.8304, 0.8169] for
pose-only -- the two ranges do not overlap, so the effect is not seed noise.

## The interesting part: where the gain comes from

Splitting the validation set by whether the pedestrian was ever occluded:

| subset | n | pose_only AP | fusion AP | gain |
|---|---|---|---|---|
| occluded | 211 | 0.7395 +-0.0566 | 0.9481 +-0.0055 | **+0.2086** |
| clear | 473 | 0.8587 +-0.0064 | 0.9418 +-0.0061 | +0.0831 |

**The gain on occluded pedestrians is 2.5x the gain on clear ones.** That is the
hypothesis this project was built on, stated in the README before any of it was measured:
pose alone degrades when the body is hidden, and scene context plus gesture carry the
prediction through. The data agrees.

The safety-relevant version of the same fact is the miss rate. On occluded pedestrians
pose-only misses **44.1%** of crossings; fusion misses **19.8%**. Those are the cases where
a planner has least warning and most need of it.

Note also the variance. On the occluded subset pose-only has a standard deviation of
0.0566 across seeds against fusion's 0.0055 -- ten times wider. Pose-only is not just worse
when the view is blocked, it is erratic.

## Verifying the results

```bash
python -m scripts.run_ablation \
  --source carla --data-root data/carla_run --pose-cache data/pose_carla.npz \
  --repeats 3 --device auto
```

Twelve seconds on a 12 GB GPU. `artifacts/ablation.json` records the per-seed AP for every
arm and subset, so the means above can be recomputed rather than taken on trust:

```python
import json, statistics
d = json.load(open("artifacts/ablation.json"))
statistics.mean(d["arms"]["fusion"]["ap_all"])   # 0.9432
```

The 222 episode recordings are committed under `data/carla_run/` (12 MB, annotations
only), along with `artifacts/ablation.json`. The RGB frames are not: pose is cached in
`data/pose_carla.npz` and the frames were deleted once extracted, which is why 222 episodes
occupy 12 MB rather than 55 GB. The pose cache (2.8 MB) and the MediaPipe model (9 MB) are
also left out, since both are regenerable -- the model via `scripts/fetch_pose_model.sh`,
the cache by re-running the collector and `scripts/extract_pose.py`. Reproducing the
ablation from a fresh clone therefore means re-collecting; reading the result does not.

## Honest limits

**Synthetic, and the gesture head is untested.** CARLA supplies no gesture annotations, so
those label groups are masked during training on this data. The fusion arm's advantage here
comes from scene context and the extra capacity, not from gesture recognition. The gesture
half of the architecture has still never been evaluated against a real label.

**One simulator, one town.** All 222 episodes are Town10HD_Opt with the same camera
intrinsics and four scripted scenario types. Generalisation across towns, weather, and
camera setups is unmeasured.

**50 validation episodes is a small held-out set.** The per-seed spread is tight and the
arms are cleanly separated, but the third decimal place should not be leaned on.

**The real-data number is partial.** There is now a JAAD result (see the section at the
top): AP 0.634 on real pedestrians, by-video split. But it runs without pose, because
the clips are a separate 2.9 GB download that has not been made. The full
pose-plus-gesture-plus-context architecture has still never been evaluated on real
data, and the gesture head has still never been tested with its pose input present.

## Summary

On synthetic CARLA data with ground-truth labels, fusing pose with gesture and scene
context predicts crossing intent 2.25 s ahead at AP 0.943 against 0.830 for pose alone, and
the advantage is 2.5x larger when the pedestrian is occluded -- 0.209 AP against 0.083. The
occlusion result is the one worth keeping: it is the case the architecture was designed
for, it is the case a planner most needs, and it is now measured rather than argued.
