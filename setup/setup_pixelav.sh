#!/usr/bin/env bash
# ============================================================================
# setup_pixelav.sh — reproducibly (re)build the PIXELAV working tree on EAF.
#
# WHY THIS EXISTS: PIXELAV is built under /tmp on EAF, which is OFF-QUOTA
# (the per-user /home quota is only 23 GB) but EPHEMERAL — /tmp is wiped when
# the EAF container restarts. Re-run this script to recreate the build from
# scratch in a fresh session.
#
# Usage (from an EAF JupyterLab terminal, in the CALOMAPS repo root):
#     bash setup/setup_pixelav.sh
#
# Result: a working binary at  $PIXELAV_WORK/pixelav/ppixelav2_list_trkpy_n_2f
# (default $PIXELAV_WORK = /tmp/pixelav_journey).
# ============================================================================
set -euo pipefail

WORK="${PIXELAV_WORK:-/tmp/pixelav_journey}"
SRC="$WORK/pixelav"
# Our additions live in the repo and are copied into the upstream tree:
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PXDIR="$REPO_DIR/analysis/pixelav"

DRIVER_UPSTREAM="ppixelav2_list_trkpy_n_2f"          # 7-col list driver (baseline)
DRIVER_REAL="ppixelav2_list_trkpy_real_entry"        # our patched driver (real entry pt)

echo "[setup] PIXELAV_WORK=$WORK"
mkdir -p "$WORK"
cd "$WORK"

# 1) Clone Swartz's PIXELAV (public mirror, no CERN login). ~25 MB.
if [ ! -d "$SRC/.git" ]; then
  echo "[setup] cloning badeaa3/pixelav ..."
  git clone https://github.com/badeaa3/pixelav.git
else
  echo "[setup] upstream source already present: $SRC"
fi
cd "$SRC"

# 2) Build the upstream baseline driver (single-file compile; sse2neon.h is ARM-only,
#    x86-64 uses <xmmintrin.h>). gcc 11.5 on EAF builds it clean.
echo "[setup] building $DRIVER_UPSTREAM (gcc -O2 -msse) ..."
gcc -O2 "${DRIVER_UPSTREAM}.c" -msse -lm -o "$DRIVER_UPSTREAM"
echo "[setup]   -> $(ls -la "$DRIVER_UPSTREAM" | awk '{print $5, $NF}')"

# 3) If our patched driver / Stage-A config exist in the repo, install + build them.
if [ -f "$PXDIR/${DRIVER_REAL}.c" ]; then
  echo "[setup] installing patched driver ${DRIVER_REAL}.c from repo ..."
  cp "$PXDIR/${DRIVER_REAL}.c" .
  gcc -O2 "${DRIVER_REAL}.c" -msse -lm -o "$DRIVER_REAL"
  echo "[setup]   -> $(ls -la "$DRIVER_REAL" | awk '{print $5, $NF}')"
fi
if [ -f "$PXDIR/ppixel2.decal.init" ]; then
  echo "[setup] installing DECAL Stage-A config (ppixel2.init + wgt_pot.init + weighting) ..."
  cp "$PXDIR/ppixel2.decal.init" ppixel2.init
  [ -f "$PXDIR/wgt_pot.decal.init" ]    && cp "$PXDIR/wgt_pot.decal.init"    wgt_pot.init
  [ -f "$PXDIR/weighting.decal.init" ]  && cp "$PXDIR/weighting.decal.init"  weighting.decal.init
fi

# 4) Smoke test: one perpendicular 2 GeV track through whatever config is in place.
echo "[setup] smoke test ..."
printf "0.0 0.0 2.0 1 0.0 0.0 2.0\n" > _smoke_deck.txt
rm -f _smoke_out.out _smoke_seed.txt
./"$DRIVER_UPSTREAM" 1 _smoke_deck.txt _smoke_out.out _smoke_seed.txt >/dev/null 2>&1 || true
if grep -q "<cluster>" _smoke_out.out 2>/dev/null; then
  echo "[setup] smoke test PASS ($(grep -c '<cluster>' _smoke_out.out) cluster; param line: $(sed -n 2p _smoke_out.out))"
  rm -f _smoke_deck.txt _smoke_out.out _smoke_seed.txt
else
  echo "[setup] smoke test FAIL — check $SRC"; exit 1
fi

echo "[setup] DONE. Run dir: $SRC"
echo "[setup] A run dir needs: SIRUTH.SPR, ppixel2.init, wgt_pot.init, <weighting>.init, track_list.txt"
echo "[setup] Invoke:  ./$DRIVER_UPSTREAM 1 track_list.txt clusters.out seedfile.txt"
