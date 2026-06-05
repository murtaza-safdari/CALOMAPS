#!/usr/bin/env bash
# ============================================================================
# setup_pixelav.sh — reproducibly (re)build PIXELAV + the DECAL sensor model on EAF.
#
# WHY: PIXELAV is built under /tmp on EAF (OFF-QUOTA; the per-user /home quota is only
# 23 GB) but /tmp is EPHEMERAL — wiped on container restart. Re-run this to recreate
# everything from scratch in a fresh session.
#
# Usage (EAF JupyterLab terminal, in the CALOMAPS repo root):
#     bash setup/setup_pixelav.sh
#
# Produces, under $PIXELAV_WORK (default /tmp/pixelav_journey):
#   pixelav/ppixelav2_list_trkpy_n_2f          baseline 7-col list driver
#   pixelav/ppixelav2_list_trkpy_real_entry    our patched driver (real entry injection)
#   pixelav/decal_run/                         ready-to-run DECAL sensor dir (320um/100um)
# ============================================================================
set -euo pipefail
ulimit -s unlimited 2>/dev/null || true   # NEHSTORE=1,000,000 static arrays overflow the 8 MB default stack

WORK="${PIXELAV_WORK:-/tmp/pixelav_journey}"
SRC="$WORK/pixelav"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PXDIR="$REPO_DIR/analysis/pixelav"
PY="${PIXELAV_PY:-/opt/conda/bin/python3}"     # needs numpy (the Stage-A generator)

echo "[setup] PIXELAV_WORK=$WORK ; repo=$REPO_DIR"
mkdir -p "$WORK"; cd "$WORK"

# 1) Clone Swartz's PIXELAV (public mirror, no CERN login). ~25 MB.
if [ ! -d "$SRC/.git" ]; then
  echo "[setup] cloning badeaa3/pixelav ..."; git clone https://github.com/badeaa3/pixelav.git
else
  echo "[setup] upstream already present: $SRC"
fi
cd "$SRC"

# 2) Build the baseline driver and our real-entry driver (single-file compile; sse2neon.h
#    is ARM-only, x86-64 uses <xmmintrin.h>).
echo "[setup] building baseline + patched drivers ..."
gcc -O2 ppixelav2_list_trkpy_n_2f.c -msse -lm -o ppixelav2_list_trkpy_n_2f
cp "$PXDIR/ppixelav2_list_trkpy_real_entry.c" .
gcc -O2 ppixelav2_list_trkpy_real_entry.c -msse -lm -o ppixelav2_list_trkpy_real_entry

# 3) Fix the bundled weighting potential: upstream wgt_pot.init holds only the PATH
#    "./weighting_BPix_50x13x100.init", but pixinit() reads the weighting GRID from
#    wgt_pot.init itself -> it must BE the data file (its header already matches).
cp weighting_BPix_50x13x100.init wgt_pot.init

# 4) Generate the DECAL Stage-A sensor model (320 um, 100 um pitch, simple uniform E-field,
#    B=0, FFT Ramo weighting potential).
echo "[setup] generating DECAL Stage-A (ppixel2.decal.init + wgt_pot.decal.init) ..."
"$PY" "$PXDIR/make_decal_stagea.py" .

# 5) Assemble a ready-to-run DECAL run directory.
RUN="$SRC/decal_run"; mkdir -p "$RUN"
cp SIRUTH.SPR ppixelav2_list_trkpy_real_entry ppixel2.decal.init wgt_pot.decal.init "$RUN"/
cp "$RUN/ppixel2.decal.init" "$RUN/ppixel2.init"
cp "$RUN/wgt_pot.decal.init" "$RUN/wgt_pot.init"

# 6) Smoke test on the DECAL geometry.
cd "$RUN"
printf "0.0 0.0 2.0 1 0.0 0.0 2.0\n" > _smoke.txt; rm -f _s.out _s.seed
./ppixelav2_list_trkpy_real_entry 1 _smoke.txt _s.out _s.seed >/dev/null 2>&1 || true
if grep -q "<cluster>" _s.out 2>/dev/null; then
  echo "[setup] DECAL smoke PASS (param line: $(sed -n 2p _s.out))"
  rm -f _smoke.txt _s.out _s.seed
else
  echo "[setup] DECAL smoke FAIL — check $RUN"; exit 1
fi

echo "[setup] DONE."
echo "[setup] DECAL run dir: $RUN"
echo "[setup] Run our event:"
echo "        cp $REPO_DIR/models/pixelav_segments_gamma50_1evt.pixelav.txt $RUN/track_list.txt"
echo "        cd $RUN && ulimit -s unlimited && ./ppixelav2_list_trkpy_real_entry 1 track_list.txt clusters.out seedfile.txt"
echo "[setup] (baseline BPix sensor instead: use ppixel2.init/wgt_pot.init in $SRC + ppixelav2_list_trkpy_n_2f)"
