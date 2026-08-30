#!/usr/bin/env bash
# Clone the JAAD and PIE reference repositories into external/.
#
# These are the official annotation releases from the group that built both
# datasets. They carry the XML annotations, the split files, and the authors'
# own python interfaces (jaad_data.py, pie_data.py), which are what the parsers
# in src/pif/data were written against.
#
# What this does NOT fetch: the video clips. JAAD's are 3.1 GB and PIE's are
# 74 GB, they live on a separate server, and nothing in this repo needs them
# until you want real pose features. See the README section "Getting the data".
#
# external/ is git-ignored on purpose. It is ~620 MB of somebody else's history
# and it must never end up in a commit here.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/external"

JAAD_URL="https://github.com/ykotseruba/JAAD.git"
PIE_URL="https://github.com/aras62/PIE.git"

mkdir -p "${DEST}"

clone_or_update() {
  local url="$1" dir="$2" name="$3"
  if [ -d "${dir}/.git" ]; then
    echo "==> ${name}: already at ${dir}, pulling"
    git -C "${dir}" pull --ff-only
  else
    echo "==> ${name}: cloning ${url}"
    # --depth 1: we want the files, not the history. Saves a few hundred MB.
    git clone --depth 1 "${url}" "${dir}"
  fi
}

clone_or_update "${JAAD_URL}" "${DEST}/jaad" "JAAD"
clone_or_update "${PIE_URL}" "${DEST}/pie" "PIE"

echo
echo "==> checking what landed"

# JAAD ships its XML unpacked, one file per video per annotation type.
jaad_videos=$(find "${DEST}/jaad/annotations" -name 'video_*.xml' 2>/dev/null | wc -l)
echo "JAAD  ${jaad_videos} annotation files in external/jaad/annotations"
if [ "${jaad_videos}" -lt 300 ]; then
  echo "      WARNING: expected 346. The clone may be incomplete."
fi

# PIE keeps its XML packed in three zips; the parser reads them without
# unpacking, so leaving them as-is is correct.
for z in annotations.zip annotations_attributes.zip annotations_vehicle.zip; do
  if [ -f "${DEST}/pie/annotations/${z}" ]; then
    echo "PIE   annotations/${z} present ($(du -h "${DEST}/pie/annotations/${z}" | cut -f1))"
  else
    echo "PIE   annotations/${z} MISSING"
  fi
done

echo
echo "total: $(du -sh "${DEST}" | cut -f1) in ${DEST}"
echo
echo "Next: python -m scripts.parse_annotations --dataset jaad --limit 20"
echo "The video clips are a separate download and are not needed for that."
