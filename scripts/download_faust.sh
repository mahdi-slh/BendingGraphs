#!/usr/bin/env bash
# Helper to drop the MPI-FAUST registrations into ``data/MPI-FAUST/``.
#
# The dataset is gated: you must register at http://faust.is.tue.mpg.de/
# and accept the licence before you can download.  This script does
# *not* fetch the archive itself; it lays out the expected directory
# structure and (optionally) extracts a zip you point it at.
#
# Usage:
#   ./scripts/download_faust.sh                       # just create dirs
#   FAUST_ZIP=/path/to/MPI-FAUST.zip \
#       ./scripts/download_faust.sh                   # extract from local zip
set -euo pipefail

ROOT_DIR="${BENDING_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"
FAUST_DIR="${MPIFAUST_DIR:-$ROOT_DIR/MPI-FAUST}"

mkdir -p "$FAUST_DIR/training/registrations"
mkdir -p "$FAUST_DIR/training/scans"
mkdir -p "$FAUST_DIR/training/scans_off"
mkdir -p "$FAUST_DIR/test/scans"

if [[ -n "${FAUST_ZIP:-}" ]]; then
    if [[ ! -f "$FAUST_ZIP" ]]; then
        echo "FAUST_ZIP set to $FAUST_ZIP but the file does not exist." >&2
        exit 1
    fi
    echo "==> Extracting $FAUST_ZIP into $FAUST_DIR ..."
    unzip -q -o "$FAUST_ZIP" -d "$FAUST_DIR"
fi

# The FAUST archive ships ``training/registrations/tr_reg_XXX.ply`` files;
# the loader also reads off-mesh copies under ``training/scans_off/`` for the
# patch-mode generators.  Build them lazily if they're missing.
if compgen -G "$FAUST_DIR/training/registrations/tr_reg_*.ply" > /dev/null; then
    if ! compgen -G "$FAUST_DIR/training/scans_off/tr_reg_*.off" > /dev/null; then
        echo "==> Converting registrations to .off (lazy, one-time)..."
        python - <<PYEOF
import glob, os
import open3d as o3d
src = "$FAUST_DIR/training/registrations"
dst = "$FAUST_DIR/training/scans_off"
os.makedirs(dst, exist_ok=True)
for path in sorted(glob.glob(os.path.join(src, "tr_reg_*.ply"))):
    out = os.path.join(dst, os.path.splitext(os.path.basename(path))[0] + ".off")
    if os.path.exists(out):
        continue
    mesh = o3d.io.read_triangle_mesh(path)
    o3d.io.write_triangle_mesh(out, mesh)
print("done")
PYEOF
    fi
else
    cat <<'EOF'
==> No FAUST .ply files found yet.

   1. Register at http://faust.is.tue.mpg.de/ and download MPI-FAUST.zip
   2. Place the zip somewhere local.
   3. Re-run with FAUST_ZIP=/path/to/MPI-FAUST.zip ./scripts/download_faust.sh
   (or just unzip the archive into data/MPI-FAUST/ manually).

   The expected structure is:

      data/MPI-FAUST/
        training/registrations/tr_reg_000.ply ... tr_reg_099.ply
        training/scans/scan_000.ply ...
        training/pairs.json
        test/scans/...
EOF
fi

echo "==> FAUST root: $FAUST_DIR"
