#!/bin/bash
# make_pixelav_inputs.sh — one command: cascade simulation -> .npz -> PIXELAV deck.
#
# Replaces the five manual steps (handbook §10.1: two ddsim runs + two extractors +
# the converter). Particle + energy come from the same env vars as the rest of the
# pipeline (no file edits):
#   CALOMAPS_GUN_PARTICLE     particle name (default gamma; e.g. pi+, pi-, proton)
#   CALOMAPS_GUN_ENERGY_GEV   beam energy in GeV (default 50; legacy alias CALOMAPS_GUN_GEV)
#
# Usage (EAF JupyterLab terminal, after `source ~/setup_calomaps.sh`):
#   bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh                 # gamma 50 GeV
#   CALOMAPS_GUN_PARTICLE=pi+ bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh
#   CALOMAPS_GUN_PARTICLE=pi- CALOMAPS_GUN_ENERGY_GEV=80 bash .../make_pixelav_inputs.sh --fullcascade
#
# Produces (ROOT under $CALOMAPS_DATA_BASE, the rest under $CALOMAPS_HOME/models):
#   trackermom_<tag><E>_1evt.root        tracker-SD readout: per-crossing momentum (the PIXELAV source)
#   trackermom_<tag><E>_1evt.npz         MCParticle cascade + tracker-hit arrays
#   pixelav_segments_<tag><E>_1evt.pixelav.txt        the 7-column badeaa3 PIXELAV deck (the hand-off file)
#   pixelav_segments_<tag><E>_1evt.pixelav.txt.columns.txt   column legend
#   pixelav_segments_<tag><E>_1evt.{json,csv}         per-crossing records (16 fields each)
# With --fullcascade it ALSO makes the calorimeter cascade (for nb04 / nb05b calo-SD view):
#   fullcascade_<tag><E>_1evt.root + .npz
#
# Assumes the Key4hep env is sourced (ddsim + python on PATH).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPACT="$REPO_DIR/geometry/SiD_TestBeam.xml"
HOME_DIR="${CALOMAPS_HOME:-$REPO_DIR}"
DATA="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}"
PY="${CALOMAPS_PY:-python3}"

PARTICLE="${CALOMAPS_GUN_PARTICLE:-gamma}"
GEV="${CALOMAPS_GUN_ENERGY_GEV:-${CALOMAPS_GUN_GEV:-50}}"
case "$PARTICLE" in
  gamma) TAG="gamma" ;;                 # matches the steering files' filename convention
  pi+)   TAG="piplus" ;;
  pi-)   TAG="piminus" ;;
  *)     TAG="$(echo "$PARTICLE" | sed 's/+/plus/g; s/-/minus/g')" ;;
esac
ETAG="$(printf '%g' "$GEV")"            # 50 not 50.0, to match the steering "%g" naming
# Export so the steering files (which read the same vars) stay in lock-step with this script.
export CALOMAPS_GUN_PARTICLE="$PARTICLE" CALOMAPS_GUN_ENERGY_GEV="$GEV"

DO_FULL=0
[ "${1:-}" = "--fullcascade" ] && DO_FULL=1

mkdir -p "$DATA/trackermom" "$HOME_DIR/models"

echo "=========================================="
echo " PIXELAV inputs:  $PARTICLE @ ${ETAG} GeV"
echo "=========================================="

echo "[1/3] tracker-SD simulation (per-crossing momentum)"
TRK_ROOT="$DATA/trackermom/trackermom_${TAG}${ETAG}_1evt.root"
ddsim --compactFile "$COMPACT" --steeringFile "$REPO_DIR/sim/run_sim_trackermom.py" \
      --numberOfEvents 1 --outputFile "$TRK_ROOT"

echo "[2/3] extract -> .npz"
TRK_NPZ="$HOME_DIR/models/trackermom_${TAG}${ETAG}_1evt.npz"
"$PY" "$REPO_DIR/analysis/extract_trackermom.py" "$TRK_ROOT" "$TRK_NPZ"

echo "[3/3] convert -> PIXELAV deck (variant C tracker, badeaa3 layout)"
SEG_PREFIX="$HOME_DIR/models/pixelav_segments_${TAG}${ETAG}_1evt"
"$PY" "$REPO_DIR/analysis/pixelav_converter.py" "$TRK_NPZ" "$SEG_PREFIX"

if [ "$DO_FULL" = "1" ]; then
  echo "[+] full calorimeter cascade (for nb04 / nb05b calo-SD view)"
  mkdir -p "$DATA/fullcascade"
  FC_ROOT="$DATA/fullcascade/fullcascade_${TAG}${ETAG}_1evt.root"
  ddsim --compactFile "$COMPACT" --steeringFile "$REPO_DIR/sim/run_sim_fullcascade.py" \
        --numberOfEvents 1 --outputFile "$FC_ROOT"
  "$PY" "$REPO_DIR/analysis/extract_cascade.py" "$FC_ROOT" "$HOME_DIR/models/fullcascade_${TAG}${ETAG}_1evt.npz"
fi

echo
echo "DONE. PIXELAV hand-off package:"
echo "  deck    : $SEG_PREFIX.pixelav.txt"
echo "            (7 cols: cot_alpha cot_beta ppion flipped modx mody pT)"
echo "  legend  : $SEG_PREFIX.pixelav.txt.columns.txt"
echo "  records : $SEG_PREFIX.json / .csv   (16 fields per crossing)"
echo "  source  : $TRK_NPZ   (MCParticle cascade + tracker-hit arrays)"
