# End goal

## The one sentence version

A model that tells a vehicle's planner "this person is about to step into the
road" about a second and a half before they do, running on the vehicle, from
the camera it already has.

## What "done" looks like

A single ONNX file that takes 16 frames of one tracked pedestrian — their pose,
their scene context, their recent box motion — and returns a crossing
probability and a short future trajectory, with a measured precision/recall
curve on real JAAD and PIE video, a measured latency on a Jetson-class board,
and an ablation showing that the pose and context and gesture branches each
earn their place.

Nothing in that sentence is achieved yet except the file format.

## Why this problem

Trajectory prediction is a solved-enough problem that its remaining weakness is
structural rather than technical: it cannot tell you about a decision until the
decision has already changed the motion. A pedestrian who has decided to cross
looks, for perhaps a second, exactly like a pedestrian who has not — same
position, same velocity, same box. The difference in that second is in the
body: shoulders rotating towards the road, a glance over the shoulder for
traffic, a shortened stride, a raised hand.

A vehicle at 50 km/h covers roughly 14 metres a second. Recovering that second
is the difference between a comfortable deceleration and a hard stop. That is
the whole thesis.

## The specific claim to be tested

> Fusing scene context and a gesture/action taxonomy with body pose predicts a
> crossing decision earlier and more accurately than body pose alone.

Falsifiable, and the repo is built to test it: the pose-only baseline
(`PoseOnlyBaseline`) is the same backbone with the branches switched off, so a
null result is a real possible outcome and would be worth reporting.

The comparison arm stands in for the family of pose-only crossing classifiers,
of which Fang & López (IV 2018) is the clearest example.

## Success criteria

These are targets, not results. Nothing here has been measured.

1. **The forecast is genuinely early.** Average precision reported at 1.0 s,
   1.5 s and 2.0 s lead, not just at one horizon, with the windowing rule that
   no observation extends past the annotated crossing point.
2. **Fusion beats pose alone by a margin bigger than seed noise.** Three seeds
   per arm minimum, AP reported as mean and standard deviation, on the
   published JAAD and PIE splits.
3. **The comparison is fair.** Same windows, same split, same optimiser, same
   schedule, same parameter budget where possible. If the fused model is bigger,
   that is reported and a width-matched baseline is run too.
4. **Miss rate is reported, not hidden inside recall.** A planner cares about
   the crossing it did not see. The operating curve goes in the write-up, not a
   single F1 at threshold 0.5.
5. **It exports and it runs on the target.** ONNX round trip within tolerance
   (done), then a measured latency and memory footprint on a real Jetson Orin
   through TensorRT (not done — no device).
6. **The gesture head means something.** Per-group accuracy on the groups JAAD
   and PIE actually annotate, and honesty about the groups they do not.

## Explicit non-goals

- **Not a detector or a tracker.** Both datasets ship annotated boxes and track
  ids. Perception upstream is assumed.
- **Not a planner.** The output is a probability and a short trajectory. What a
  vehicle does with it is somebody else's module.
- **Not a new pose estimator.** MediaPipe now, HRNet later; both are somebody
  else's work and the code is structured so either drops in.
- **Not multi-agent or social.** One pedestrian at a time. Groups matter to
  crossing behaviour and are a natural extension, but they are not in scope.
- **Not a leaderboard entry.** The point is the early-warning margin and
  whether context earns its keep, not a state-of-the-art number.

## What would make this fail honestly

Worth writing down in advance so it does not get quietly rationalised later:

- The gesture labels derivable from JAAD and PIE may be too coarse. Their
  behaviour columns were designed for a different question, and mapping them
  into a TITAN-shaped taxonomy is an approximation.
- MediaPipe on a 60-pixel-tall pedestrian may simply not resolve the cues the
  hypothesis depends on. If the pose branch is noise, fusion wins for the wrong
  reason — because context alone is good — and the ablation must separate
  pose-only, context-only and fused to catch that.
- Both datasets are forward-facing dashcam from one group in two Canadian
  cities. Anything learned here transfers on faith until tested elsewhere.

## Sequence to get there

1. Obtain the JAAD clips (3.1 GB) and the PIE clips (74 GB).
2. Run pose extraction, cache the features.
3. Train the fused model on JAAD, then PIE. Compare against the pose-only arm.
4. Sweep the lead time; report the AP-versus-lead curve. That curve *is* the
   result.
5. Add a context-only arm to separate "context is doing the work" from "fusion
   is doing the work".
6. Get a Jetson, export, benchmark, report latency honestly.
