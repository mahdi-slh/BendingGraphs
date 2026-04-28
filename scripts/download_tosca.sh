#!/usr/bin/env bash
# Helper to drop the TOSCA high-resolution dataset into ``data/tosca/``.
#
# TOSCA is gated: register at http://tosca.cs.technion.ac.il/ and download
# ``toscahires-mat.zip``.  This script lays out ``data/tosca/`` and (if
# you point ``TOSCA_ZIP=`` at a local archive) extracts it there.
#
# Usage:
#   ./scripts/download_tosca.sh                       # just create dirs
#   TOSCA_ZIP=/path/to/toscahires-mat.zip \
#       ./scripts/download_tosca.sh                   # extract from local zip
set -euo pipefail

ROOT_DIR="${BENDING_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"
TOSCA_OUT="${TOSCA_DIR:-$ROOT_DIR/tosca}"
TARGET="$TOSCA_OUT/toscahires-mat"
mkdir -p "$TARGET"

if [[ -n "${TOSCA_ZIP:-}" ]]; then
    if [[ ! -f "$TOSCA_ZIP" ]]; then
        echo "TOSCA_ZIP=$TOSCA_ZIP does not exist." >&2
        exit 1
    fi
    echo "==> Extracting $TOSCA_ZIP into $TARGET ..."
    unzip -q -o "$TOSCA_ZIP" -d "$TARGET"
fi

if compgen -G "$TARGET/*.mat" > /dev/null; then
    echo "==> TOSCA ready at $TARGET ($(ls $TARGET | wc -l) files)"
else
    cat <<EOF
==> TOSCA .mat files not yet in $TARGET.

   1. Register at http://tosca.cs.technion.ac.il/ and download toscahires-mat.zip
   2. Re-run with:  TOSCA_ZIP=/path/to/toscahires-mat.zip ./scripts/download_tosca.sh

   The expected layout is:
      $TARGET/cat0.mat ... horse19.mat ... wolfXX.mat
EOF
fi
