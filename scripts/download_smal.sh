#!/usr/bin/env bash
# Helper to drop the SMAL parametric model into ``data/smal/``.
#
# SMAL is gated: register at https://smal.is.tue.mpg.de/ and download
# ``smal_online_V1.0.zip``.  This script lays out ``data/smal/`` and (if
# you point ``SMAL_ZIP=`` at a local archive) extracts it there.
#
# Usage:
#   ./scripts/download_smal.sh                  # just create the dir
#   SMAL_ZIP=/path/to/smal_online_V1.0.zip \
#       ./scripts/download_smal.sh              # extract from local zip
set -euo pipefail

ROOT_DIR="${BENDING_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"
SMAL_OUT="${SMAL_DIR:-$ROOT_DIR/smal}"
mkdir -p "$SMAL_OUT"

if [[ -n "${SMAL_ZIP:-}" ]]; then
    if [[ ! -f "$SMAL_ZIP" ]]; then
        echo "SMAL_ZIP=$SMAL_ZIP does not exist." >&2
        exit 1
    fi
    echo "==> Extracting $SMAL_ZIP into $SMAL_OUT ..."
    unzip -q -o "$SMAL_ZIP" -d "$SMAL_OUT"
    # The archive nests the data under ``smal_online_V1.0/`` — flatten it.
    if [[ -d "$SMAL_OUT/smal_online_V1.0" ]]; then
        mv "$SMAL_OUT/smal_online_V1.0/"* "$SMAL_OUT/"
        rmdir "$SMAL_OUT/smal_online_V1.0"
    fi
fi

if [[ -f "$SMAL_OUT/smal_CVPR2017.pkl" && -f "$SMAL_OUT/smal_CVPR2017_data.pkl" ]]; then
    echo "==> SMAL ready at $SMAL_OUT"
else
    cat <<EOF
==> SMAL parametric model not yet in $SMAL_OUT.

   1. Register at https://smal.is.tue.mpg.de/ and download smal_online_V1.0.zip
   2. Re-run with:  SMAL_ZIP=/path/to/smal_online_V1.0.zip ./scripts/download_smal.sh

   The expected files are:
      $SMAL_OUT/smal_CVPR2017.pkl
      $SMAL_OUT/smal_CVPR2017_data.pkl
EOF
fi
