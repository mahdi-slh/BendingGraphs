#!/usr/bin/env bash
# Helper to drop the SURREAL training data + SMPL parametric model into
# ``data/surreal/``.  Both are gated:
#
# * SURREAL: register at https://www.di.ens.fr/willow/research/surreal/data/
# * SMPL:    register at https://smpl.is.tue.mpg.de/ and download
#            ``SMPL_python_v.1.0.0.zip``.
#
# Usage:
#   ./scripts/download_surreal.sh                          # just create dirs
#
#   SURREAL_USER=foo SURREAL_PASS=bar \
#       ./scripts/download_surreal.sh                       # fetch smpl_data.npz
#
#   SMPL_ZIP=/path/to/SMPL_python_v.1.0.0.zip \
#       ./scripts/download_surreal.sh                       # install SMPL .pkl files
set -euo pipefail

ROOT_DIR="${BENDING_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"
SURREAL_DATA="${SURREAL_DATA_DIR:-$ROOT_DIR/surreal/smpl_data}"
SURREAL_MODELS="${SURREAL_MODEL_DIR:-$ROOT_DIR/surreal/smpl_models}"
mkdir -p "$SURREAL_DATA" "$SURREAL_MODELS"

# ---- 1. SURREAL smpl_data.npz ----------------------------------------------
if [[ -n "${SURREAL_USER:-}" && -n "${SURREAL_PASS:-}" ]]; then
    URL="https://lsh.paris.inria.fr/SURREAL/smpl_data/smpl_data.npz"
    OUT="$SURREAL_DATA/smpl_data.npz"
    if [[ ! -f "$OUT" ]]; then
        echo "==> Fetching SURREAL smpl_data.npz ..."
        curl --user "$SURREAL_USER:$SURREAL_PASS" -L -o "$OUT" "$URL"
    fi
fi

# ---- 2. SMPL .pkl files ----------------------------------------------------
if [[ -n "${SMPL_ZIP:-}" ]]; then
    if [[ ! -f "$SMPL_ZIP" ]]; then
        echo "SMPL_ZIP=$SMPL_ZIP does not exist." >&2
        exit 1
    fi
    TMP=$(mktemp -d)
    echo "==> Extracting $SMPL_ZIP -> $TMP ..."
    unzip -q -o "$SMPL_ZIP" -d "$TMP"
    # Move all basicModel_* / basicmodel_* .pkl files to SURREAL_MODELS.
    find "$TMP" -name "basic*model_*.pkl" -exec cp {} "$SURREAL_MODELS/" \;
    rm -rf "$TMP"
fi

echo
if [[ -f "$SURREAL_DATA/smpl_data.npz" ]]; then
    echo "==> SURREAL data ready at $SURREAL_DATA"
else
    echo "==> SURREAL smpl_data.npz still missing — set SURREAL_USER + SURREAL_PASS,"
    echo "    or download manually and place it at $SURREAL_DATA/smpl_data.npz"
fi
if compgen -G "$SURREAL_MODELS/basic*model_*.pkl" > /dev/null; then
    echo "==> SMPL models ready at $SURREAL_MODELS"
else
    echo "==> SMPL .pkl models still missing — set SMPL_ZIP=... or copy them"
    echo "    manually into $SURREAL_MODELS/"
fi
