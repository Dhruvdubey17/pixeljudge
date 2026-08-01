#!/usr/bin/env bash
# Fetch open, freely licensed master clips and turn them into local references.
#
# Why these clips: they are short (10 s), small (about 10 MB each), openly
# licensed, and hosted specifically for testing. That keeps a full checkout in the
# tens of megabytes instead of the tens of gigabytes a real studio master would
# cost, while still being genuine photographic and animated content rather than
# a synthetic pattern.
#
# Honest caveat, repeated in DATA_CARD.md: these downloads are already H.264
# encoded, so they are *mezzanine* files, not pristine masters. We transcode each
# one to a lossless local reference so that our own encodes are measured against
# a fixed, un-degrading target, but the reference itself carries whatever the
# original encode already threw away. Absolute VMAF numbers are therefore slightly
# optimistic; comparisons between our ladders on the same reference are unaffected,
# which is what the report actually claims.
#
# For a truly uncompressed reference, use Xiph's y4m sequences
# (https://media.xiph.org/video/derf/) - correct, and roughly 200x larger.
#
# Usage:
#   bash scripts/download_media.sh              # into data/raw
#   OUT=/tmp/masters bash scripts/download_media.sh

set -euo pipefail

OUT="${OUT:-data/raw}"
FFMPEG="${PIXELJUDGE_FFMPEG:-ffmpeg}"

mkdir -p "$OUT"

# name|url  - Blender Foundation open movies (CC BY) plus a high-motion clip.
CLIPS=(
  "big_buck_bunny|https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4"
  "sintel|https://test-videos.co.uk/vids/sintel/mp4/h264/720/Sintel_720_10s_10MB.mp4"
  "jellyfish|https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_10MB.mp4"
)

# Seconds of each clip to keep. Short on purpose: an encoding sweep is
# quality-per-bit, not endurance, and every extra second multiplies across every
# rung of every ladder of every codec.
DURATION="${DURATION:-6}"

for entry in "${CLIPS[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  download="$OUT/.${name}_source.mp4"
  master="$OUT/${name}.mp4"

  if [ -f "$master" ]; then
    echo "skip (exists): $master"
    continue
  fi

  echo "downloading $name"
  curl -fL --retry 3 --progress-bar -o "$download" "$url"

  # Lossless (-qp 0) so the reference stops degrading here, and -an because
  # nothing downstream looks at audio.
  echo "building lossless reference $master"
  "$FFMPEG" -hide_banner -nostdin -y -i "$download" -t "$DURATION" \
    -c:v libx264 -qp 0 -preset veryfast -pix_fmt yuv420p -an "$master"
  rm -f "$download"
done

echo
echo "references in $OUT:"
ls -la "$OUT" | grep -v '^total' || true
echo
echo "Licences: Big Buck Bunny and Sintel are (c) Blender Foundation, CC BY 3.0."
echo "See DATA_CARD.md for provenance and the mezzanine caveat."
