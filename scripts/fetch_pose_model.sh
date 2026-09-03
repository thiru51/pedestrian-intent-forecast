#!/usr/bin/env bash
# Download the MediaPipe pose landmarker weights.
#
# Why this script exists at all. Up to mediapipe 0.10.x the BlazePose weights
# were bundled inside the python wheel and `mp.solutions.pose` just worked.
# From mediapipe 1.0 that module was removed, the replacement is
# `mp.tasks.vision.PoseLandmarker`, and the weights are no longer in the wheel:
# they are a separate .task bundle that has to be on disk. The environment here
# has mediapipe 1.0.1, so pose extraction needs this file and will refuse to
# run without it, with a message pointing at this script.
#
# What it fetches: one file from Google's official MediaPipe model host.
#
#   pose_landmarker_full.task    about 9 MB   (the default, model_complexity 1)
#   pose_landmarker_lite.task    about 6 MB   (model_complexity 0, faster)
#   pose_landmarker_heavy.task   about 29 MB  (model_complexity 2, slowest)
#
# It lands in data/, which is git-ignored, so nothing here is ever committed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${ROOT}/data"
VARIANT="${1:-full}"

case "${VARIANT}" in
  lite|full|heavy) ;;
  *)
    echo "usage: $0 [lite|full|heavy]" >&2
    exit 2
    ;;
esac

BASE="https://storage.googleapis.com/mediapipe-models/pose_landmarker"
URL="${BASE}/pose_landmarker_${VARIANT}/float16/latest/pose_landmarker_${VARIANT}.task"
DEST="${DEST_DIR}/pose_landmarker.task"

mkdir -p "${DEST_DIR}"

if [ -s "${DEST}" ]; then
  echo "==> already present: ${DEST} ($(du -h "${DEST}" | cut -f1))"
else
  echo "==> downloading the ${VARIANT} pose landmarker"
  echo "    from ${URL}"
  curl -fL --retry 3 -o "${DEST}.part" "${URL}"
  mv "${DEST}.part" "${DEST}"
  echo "==> wrote ${DEST} ($(du -h "${DEST}" | cut -f1))"
fi

cat <<EOF

The extractor looks for this file at data/pose_landmarker.task by default.
Override with \$PIF_POSE_MODEL or --pose-model if you keep it elsewhere.

Next, turn collected CARLA frames into pose features:

    .pixi/envs/default/bin/python -m scripts.extract_pose --dataset carla \\
        --root data/carla_run --out data/pose_carla.npz --workers 5
EOF
