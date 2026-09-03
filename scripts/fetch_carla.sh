#!/usr/bin/env bash
# Download and unpack the CARLA 0.9.16 simulator, then say how to launch it.
#
# Why this project wants CARLA at all: JAAD and PIE both gate their video clips
# behind an access request, so there is no way to train on real crossing footage
# until that lands. CARLA is a free driving simulator, and a scripted crossing
# in it comes with exact ground truth - the walker's 3D box, whether it is on
# the road, and what is blocking the view - which is better labelling than any
# hand-annotated dataset can give. What it cannot give is gestures. See the
# README for the honest version of that trade.
#
# What this fetches: the packaged Linux build, 8.3 GB compressed and about
# 20 GB unpacked. It is a prebuilt binary, so there is nothing to compile.
#
# Nothing here goes into git. .gitignore has /carla/ and /CARLA_*.tar.gz for
# the case where someone points this at the repo directory by mistake.

set -euo pipefail

URL="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz"
TARBALL_NAME="CARLA_0.9.16.tar.gz"

# The exact size of the release, so a truncated download is caught here rather
# than as a confusing tar error.
EXPECTED_BYTES=8346095504

# Where it lands. Override with the first argument or the CARLA_DIR variable.
CARLA_DIR="${1:-${CARLA_DIR:-$HOME/carla}}"
TARBALL="${CARLA_DIR}/${TARBALL_NAME}"
SERVER_DIR="${CARLA_DIR}/CARLA_0.9.16"

mkdir -p "${CARLA_DIR}"

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 bytes"; }

size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }

echo "==> target directory: ${CARLA_DIR}"

if [ -x "${SERVER_DIR}/CarlaUE4.sh" ]; then
  echo "==> already unpacked at ${SERVER_DIR}, nothing to download"
else
  have=$(size_of "${TARBALL}")
  if [ "${have}" -eq "${EXPECTED_BYTES}" ]; then
    echo "==> tarball already complete ($(human "${have}"))"
  else
    if [ "${have}" -gt 0 ]; then
      echo "==> resuming: have $(human "${have}") of $(human "${EXPECTED_BYTES}")"
    else
      echo "==> downloading $(human "${EXPECTED_BYTES}"). This takes a while."
    fi
    # -C - resumes a part-finished file instead of starting over, which matters
    # a lot at this size. --retry rides out a dropped connection.
    curl -L -C - --retry 5 --retry-delay 5 -o "${TARBALL}" "${URL}"

    have=$(size_of "${TARBALL}")
    if [ "${have}" -ne "${EXPECTED_BYTES}" ]; then
      echo "ERROR: got $(human "${have}"), expected $(human "${EXPECTED_BYTES}")." >&2
      echo "       Run this script again to resume." >&2
      exit 1
    fi
  fi

  echo "==> unpacking into ${SERVER_DIR}"
  # The tarball has no top-level directory: it drops CarlaUE4.sh and friends
  # straight into the working directory, so give it one of its own.
  mkdir -p "${SERVER_DIR}"
  tar -xzf "${TARBALL}" -C "${SERVER_DIR}"
fi

if [ ! -x "${SERVER_DIR}/CarlaUE4.sh" ]; then
  echo "ERROR: ${SERVER_DIR}/CarlaUE4.sh is missing after unpacking." >&2
  exit 1
fi

cat <<EOF

==> done. ${SERVER_DIR}

Launch the server. -RenderOffScreen renders on the GPU but opens no window,
which is what you want on a headless box or over SSH:

    ${SERVER_DIR}/CarlaUE4.sh -RenderOffScreen

It listens on port 2000 and prints nothing much once it is up. Leave it running
in its own terminal, or a tmux session, and collect from a second one:

    .venv-carla/bin/python scripts/collect_carla.py --episodes 20 --out-dir data/carla

Use .venv-carla, not the pixi python. The pixi environment cannot install the
carla package at all; README, "Two environments", explains why.

Then read the episodes back with the pixi python, which is the one that has the
rest of the project:

    .pixi/envs/default/bin/python -m pytest tests/test_carla_parser.py -q

Notes:
  * The first launch takes a minute or two while shaders compile.
  * The server needs a GPU with about 6 GB free. It will start without one but
    will be extremely slow.
  * The tarball can be deleted once unpacked; it is only kept so a re-run of
    this script does not download it again.
EOF
