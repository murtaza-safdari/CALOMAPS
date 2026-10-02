#!/usr/bin/env bash
# ============================================================================
# setup_pixelav.sh — reproducibly (re)build PIXELAV + a sensor-model run directory on EAF.
#
# WHY: PIXELAV is built under /tmp on EAF (OFF-QUOTA; the per-user /home quota is only
# 23 GB) but /tmp is EPHEMERAL — wiped on container restart. Re-run this to recreate
# everything from scratch in a fresh session.
#
# Usage (EAF JupyterLab terminal, in the CALOMAPS repo root):
#     bash setup/setup_pixelav.sh                                     # decal320 sensor model
#     PIXELAV_SENSOR=bpix100_50x25 PIXELAV_SENSOR_DIR=<dir> bash setup/setup_pixelav.sh
#
# Environment:
#   PIXELAV_WORK         work area (default /tmp/pixelav_journey)
#   PIXELAV_SENSOR       sensor model installed in the run directory:
#     decal320           (default) generated here by analysis/pixelav/make_decal_stagea.py:
#                        320 um thick, 100 x 100 um pixels, uniform drift field, B = 0, 300 K.
#                        Pairs with the 320 um decks.
#     bpix100_50x25      a TCAD-derived PIXELAV field map of a 100 um thick planar sensor with
#                        50 x 25 um pixels, obtained separately and not redistributed here.
#                        Its ppixel2.init title reads
#                          dot1_50x25_phase3:BPix@-100V,3.8T@90deg,253K,rhe=1.10,rhh=0.7
#                        and its header gives 253 K, a 26 x 26 x 51 grid and a zeroed B-field
#                        line (the title's 3.8 T is not applied). Pairs with the 100 um decks.
#                        Both files must match the sha256 sums below.
#   PIXELAV_SENSOR_DIR   directory holding that model's ppixel2.init and wgt_pot.init
#                        (required for bpix100_50x25)
#   PIXELAV_ALLOW_BFIELD=1  accept a sensor model whose B field is not zero (refused by
#                        default: the CALOMAPS simulation has no magnetic field)
#   PIXELAV_PY           python3 with numpy for the decal320 generator, run with PYTHONPATH
#                        cleared (e.g. /opt/conda/bin/python3 from a Key4hep-sourced shell, whose
#                        python-3.13 PYTHONPATH breaks other interpreters). Default: python3 on
#                        PATH, in the current environment.
#   PIXELAV_UPSTREAM     where to clone PIXELAV from (default the public mirror below; a local
#                        clone works too). The pinned commit is checked out either way.
#
# Produces, under $PIXELAV_WORK:
#   pixelav/                                   upstream PIXELAV at $PIXELAV_COMMIT
#   pixelav/ppixelav2_list_trkpy_n_2f          baseline 7-col list driver (bundled BPix model)
#   pixelav/ppixelav2_list_trkpy_real_entry    our patched driver (real entry injection)
#   pixelav/decal_run/ or pixelav/bpix100_50x25_run/   ready-to-run sensor dir
# ============================================================================
set -euo pipefail
# NEHSTORE=1,000,000 arrays (some on propag()'s stack) overflow the 8 MB default stack;
# macOS caps the stack at its hard limit (64 MB), which is enough.
ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null || true

PIXELAV_COMMIT=8a774a33519e6e303d8a0eddc84c74a93efec6b5     # upstream HEAD of 2024-05-13
UPSTREAM="${PIXELAV_UPSTREAM:-https://github.com/badeaa3/pixelav.git}"
BPIX100_PPIXEL2_SHA256=edb21963717648ffd032215308bd1424f732386a41e4d42b32ecd09a3b4d0654
BPIX100_WGTPOT_SHA256=e641391b1ac33324de5d5691257df851bd7ed9cd31bbed787d3a7420b59051ff

WORK="${PIXELAV_WORK:-/tmp/pixelav_journey}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PXDIR="$REPO_DIR/analysis/pixelav"
SENSOR="${PIXELAV_SENSOR:-decal320}"
if [ -n "${PIXELAV_PY:-}" ]; then
  PYRUN=(env PYTHONPATH= "$PIXELAV_PY")
else
  PYRUN=(python3)
fi

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
verify_sha256() {    # file, expected sum
  if [ ! -f "$1" ]; then echo "[setup] ERROR: missing $1"; exit 1; fi
  local got; got="$(sha256_of "$1")"
  if [ "$got" != "$2" ]; then
    echo "[setup] ERROR: $1 has sha256 $got, expected $2 -- not the $SENSOR model"; exit 1
  fi
}

echo "[setup] PIXELAV_WORK=$WORK ; repo=$REPO_DIR ; sensor=$SENSOR"

# 0) Check the sensor-model choice before building anything.
case "$SENSOR" in
  decal320)
    DECK_TAG=""; DECK_MAKER="bash sim/make_pixelav_inputs.sh" ;;
  bpix100_50x25)
    SDIR="${PIXELAV_SENSOR_DIR:-}"
    if [ -z "$SDIR" ] || [ ! -d "$SDIR" ]; then
      echo "[setup] ERROR: PIXELAV_SENSOR=bpix100_50x25 needs PIXELAV_SENSOR_DIR, a directory"
      echo "        holding that model's ppixel2.init and wgt_pot.init (got '${SDIR}')"; exit 1
    fi
    SDIR="$(cd "$SDIR" && pwd)"
    verify_sha256 "$SDIR/ppixel2.init" "$BPIX100_PPIXEL2_SHA256"
    verify_sha256 "$SDIR/wgt_pot.init" "$BPIX100_WGTPOT_SHA256"
    echo "[setup] $SENSOR model files in $SDIR match their sha256 sums"
    DECK_TAG="_si100um"; DECK_MAKER="CALOMAPS_SI_THICKNESS_UM=100 bash sim/make_pixelav_inputs.sh" ;;
  *)
    echo "[setup] ERROR: PIXELAV_SENSOR=$SENSOR; choose decal320 or bpix100_50x25"; exit 1 ;;
esac

if [ -d "$UPSTREAM" ]; then UPSTREAM="$(cd "$UPSTREAM" && pwd)"; fi
mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"; cd "$WORK"
SRC="$WORK/pixelav"

# 1) Clone upstream PIXELAV (public mirror, no CERN login; ~25 MB) at the pinned commit.
if [ ! -d "$SRC/.git" ]; then
  echo "[setup] cloning $UPSTREAM ..."; git clone "$UPSTREAM" "$SRC"
  if ! git -C "$SRC" -c advice.detachedHead=false checkout -q "$PIXELAV_COMMIT"; then
    echo "[setup] ERROR: cannot check out the pinned commit $PIXELAV_COMMIT from $UPSTREAM"; exit 1
  fi
else
  echo "[setup] upstream already present: $SRC"
fi
HAVE="$(git -C "$SRC" rev-parse HEAD)"
if [ "$HAVE" != "$PIXELAV_COMMIT" ]; then
  echo "[setup] ERROR: $SRC is at commit $HAVE, not the pinned $PIXELAV_COMMIT."
  echo "        Check out the pin (git -C $SRC checkout $PIXELAV_COMMIT) or remove $SRC."; exit 1
fi
if ! git -C "$SRC" diff --quiet HEAD -- ppixelav2.c ppixelav2_prototypes.h \
       ppixelav2_list_trkpy_n_2f.c sse2neon.h SIRUTH.SPR; then
  echo "[setup] ERROR: $SRC has local changes to the sources or tables this build uses:"
  git -C "$SRC" diff --stat HEAD -- ppixelav2.c ppixelav2_prototypes.h \
      ppixelav2_list_trkpy_n_2f.c sse2neon.h SIRUTH.SPR; exit 1
fi
cd "$SRC"

# 2) Build the baseline driver and our real-entry driver (single-file compile). x86-64 uses
#    <xmmintrin.h> with -msse; ARM includes sse2neon.h and takes upstream's linkarm_gcc flags.
case "$(uname -m)" in
  x86_64|amd64)  ARCH_FLAGS=(-msse) ;;
  arm64|aarch64) ARCH_FLAGS=(-march=armv8-a+fp+simd+crypto+crc) ;;
  *) echo "[setup] ERROR: no PIXELAV build flags for $(uname -m)"; exit 1 ;;
esac
echo "[setup] building baseline + patched drivers ..."
gcc -O2 ppixelav2_list_trkpy_n_2f.c "${ARCH_FLAGS[@]}" -lm -o ppixelav2_list_trkpy_n_2f
cp "$PXDIR/ppixelav2_list_trkpy_real_entry.c" .
gcc -O2 ppixelav2_list_trkpy_real_entry.c "${ARCH_FLAGS[@]}" -lm -o ppixelav2_list_trkpy_real_entry

# 3) Fix the bundled weighting potential: upstream wgt_pot.init holds only the PATH
#    "./weighting_BPix_50x13x100.init", but pixinit() reads the weighting GRID from
#    wgt_pot.init itself -> it must BE the data file (its header already matches).
cp weighting_BPix_50x13x100.init wgt_pot.init

# 4) Assemble the run directory with the chosen sensor model.
if [ "$SENSOR" = decal320 ]; then
  # DECAL Stage-A: 320 um, 100 um pitch, simple uniform E-field, B=0, FFT Ramo weighting.
  echo "[setup] generating DECAL Stage-A (ppixel2.decal.init + wgt_pot.decal.init) ..."
  "${PYRUN[@]}" "$PXDIR/make_decal_stagea.py" .
  RUN="$SRC/decal_run"; mkdir -p "$RUN"
  cp SIRUTH.SPR ppixelav2_list_trkpy_real_entry ppixel2.decal.init wgt_pot.decal.init "$RUN"/
  cp "$RUN/ppixel2.decal.init" "$RUN/ppixel2.init"
  cp "$RUN/wgt_pot.decal.init" "$RUN/wgt_pot.init"
else
  RUN="$SRC/${SENSOR}_run"; mkdir -p "$RUN"
  cp SIRUTH.SPR ppixelav2_list_trkpy_real_entry "$RUN"/
  cp "$SDIR/ppixel2.init" "$SDIR/wgt_pot.init" "$RUN"/
fi

# 5) Describe the installed model; refuse an inconsistent one, or a B field unless allowed.
BCHECK=--require-zero-bfield
if [ "${PIXELAV_ALLOW_BFIELD:-0}" = 1 ]; then BCHECK=""; fi
rc=0; SUMMARY="$("${PYRUN[@]}" "$PXDIR/sensor_model.py" --summary "$RUN" $BCHECK)" || rc=$?
echo "[setup] sensor model $SUMMARY"
if [ "$rc" -ne 0 ]; then
  echo "[setup] ERROR: refusing the sensor model in $RUN (PIXELAV_ALLOW_BFIELD=1 accepts a B field)"
  exit 1
fi

# 6) Smoke test on the installed sensor model.
cd "$RUN"
printf "0.0 0.0 2.0 1 0.0 0.0 2.0\n" > _smoke.txt; rm -f _s.out _s.seed
./ppixelav2_list_trkpy_real_entry 1 _smoke.txt _s.out _s.seed >/dev/null 2>&1 || true
if grep -q "<cluster>" _s.out 2>/dev/null; then
  echo "[setup] $SENSOR smoke PASS (param line: $(sed -n 2p _s.out))"
  rm -f _smoke.txt _s.out _s.seed
else
  echo "[setup] $SENSOR smoke FAIL — check $RUN"; exit 1
fi

DECK="${CALOMAPS_HOME:-$REPO_DIR}/models/pixelav_segments_gamma50_1evt${DECK_TAG}.pixelav.txt"
echo "[setup] DONE."
echo "[setup] $SENSOR run dir: $RUN"
echo "[setup] Run the matching deck (made by '$DECK_MAKER'): check that it pairs with this"
echo "        sensor model, then copy and run it:"
echo "        ${PIXELAV_PY:-python3} $PXDIR/sensor_model.py --check-deck $DECK $RUN"
echo "        cp $DECK $RUN/track_list.txt"
echo "        cd $RUN && ulimit -s unlimited && ./ppixelav2_list_trkpy_real_entry 1 track_list.txt clusters.out seedfile.txt"
echo "[setup] (baseline BPix sensor instead: use ppixel2.init/wgt_pot.init in $SRC + ppixelav2_list_trkpy_n_2f)"
