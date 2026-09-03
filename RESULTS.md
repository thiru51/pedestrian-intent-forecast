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

**No real-data number exists.** Nothing in this repository has been trained or evaluated on
real pedestrians. That remains blocked on JAAD/PIE clip access.

## Summary

On synthetic CARLA data with ground-truth labels, fusing pose with gesture and scene
context predicts crossing intent 2.25 s ahead at AP 0.943 against 0.830 for pose alone, and
the advantage is 2.5x larger when the pedestrian is occluded -- 0.209 AP against 0.083. The
occlusion result is the one worth keeping: it is the case the architecture was designed
for, it is the case a planner most needs, and it is now measured rather than argued.
