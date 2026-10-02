#!/bin/bash
# make_pixelav_inputs.sh — one command: cascade simulation -> .npz -> PIXELAV deck.
#
# Replaces the five manual steps (handbook §10.1: two ddsim runs + two extractors +
# the converter). Particle + energy come from the same env vars as the rest of the
# pipeline (no file edits):
#   CALOMAPS_GUN_PARTICLE     particle name (default gamma; e.g. pi+, pi-, proton)
#   CALOMAPS_GUN_ENERGY_GEV   beam energy in GeV (default 50; legacy alias CALOMAPS_GUN_GEV)
#   CALOMAPS_SI_THICKNESS_UM  ECal sensor thickness in um (default: the repo geometry, 320). Any
#                             other value simulates a variant geometry under $CALOMAPS_DATA_BASE
#                             (see sim/geometry_variant.sh) and appends _si<T>um to every file name below
#   CALOMAPS_N_EVENTS         events to simulate (default 1); every name below carries _<N>evt. Above
#                             1 the npz keeps only the MCParticles with a tracker hit
#                             (extract_trackermom.py --particles hit), without daughter links
#   CALOMAPS_SEED             random seed of both steering files (an integer; default: theirs, 424242)
#   CALOMAPS_OUTPUT_DIR       directory for the npz, records and deck (default $CALOMAPS_HOME/models).
#                             50 GeV photons take about 9 MB per event there with every MCParticle,
#                             3.7 MB with --particles hit (mostly the .json and .csv records)
#
# Usage, from the checkout, with the Key4hep stack sourced (on EAF: a JupyterLab terminal, after
# `source ~/setup_calomaps.sh`):
#   bash sim/make_pixelav_inputs.sh                                # gamma 50 GeV
#   CALOMAPS_GUN_PARTICLE=pi+ bash sim/make_pixelav_inputs.sh
#   CALOMAPS_GUN_PARTICLE=pi- CALOMAPS_GUN_ENERGY_GEV=80 bash sim/make_pixelav_inputs.sh --fullcascade
#   CALOMAPS_SI_THICKNESS_UM=100 bash sim/make_pixelav_inputs.sh   # 100 um silicon
#     -> trackermom_gamma50_1evt_si100um.{root,npz}, pixelav_segments_gamma50_1evt_si100um.*
#   CALOMAPS_N_EVENTS=200 CALOMAPS_SEED=7 CALOMAPS_OUTPUT_DIR=/path/to/scratch bash sim/make_pixelav_inputs.sh
#     -> trackermom_gamma50_200evt.{root,npz}, pixelav_segments_gamma50_200evt.*
# Code and geometry come from the checkout holding this script; outputs go to
# $CALOMAPS_OUTPUT_DIR, else $CALOMAPS_HOME/models (default: this checkout's models/). The script
# warns when $CALOMAPS_HOME is set to another directory.
#
# Produces (ROOT under $CALOMAPS_DATA_BASE, the rest under the output directory), N = CALOMAPS_N_EVENTS:
#   trackermom_<tag><E>_<N>evt.root      tracker-SD readout: per-crossing momentum (the PIXELAV source)
#   trackermom_<tag><E>_<N>evt.npz       MCParticle cascade + tracker-hit arrays; validated with
#                                        analysis/validate_crossings.py before the deck is written
#   pixelav_segments_<tag><E>_<N>evt.pixelav.txt      the 7-column badeaa3 PIXELAV deck (the hand-off file):
#       cot_alpha cot_beta ppion[GeV/c] flipped modx=v[um] mody=u[um] pT=|p|[GeV/c]
#   pixelav_segments_<tag><E>_<N>evt.pixelav.txt.columns.txt   column legend (units and definitions)
#   pixelav_segments_<tag><E>_<N>evt.{json,csv}       per-crossing records (18 fields each)
# With --fullcascade it ALSO makes the calorimeter cascade (for notebook 05 / notebook 07), always
# one event: the same seed makes it the first event of the tracker run, with the calorimeter readout.
#   fullcascade_<tag><E>_1evt.root + .npz
#
# Assumes the Key4hep env is sourced (ddsim + python on PATH).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HOME_DIR="${CALOMAPS_HOME:-$REPO_DIR}"
if [ -n "${CALOMAPS_HOME:-}" ] && \
   [ "$(cd "$CALOMAPS_HOME" 2>/dev/null && pwd -P)" != "$(cd "$REPO_DIR" && pwd -P)" ]; then
  echo "WARNING: CALOMAPS_HOME=$CALOMAPS_HOME is not this checkout ($REPO_DIR):" >&2
  echo "         code and geometry come from $REPO_DIR, outputs go to ${CALOMAPS_OUTPUT_DIR:-$CALOMAPS_HOME/models}" >&2
fi
DATA="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}"
PY="${CALOMAPS_PY:-python3}"
source "$SCRIPT_DIR/geometry_variant.sh" || exit 1   # COMPACT, GEO_HOME, SI_TAG

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

N_EVENTS="${CALOMAPS_N_EVENTS:-1}"
if ! [[ "$N_EVENTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CALOMAPS_N_EVENTS='$N_EVENTS' is not a positive number of events" >&2
  exit 1
fi
NTAG="_${N_EVENTS}evt"
if [ -n "${CALOMAPS_SEED:-}" ]; then
  if ! [[ "$CALOMAPS_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: CALOMAPS_SEED='$CALOMAPS_SEED' is not a non-negative integer" >&2
    exit 1
  fi
  export CALOMAPS_SEED                  # read by both steering files
fi
OUT_DIR="${CALOMAPS_OUTPUT_DIR:-$HOME_DIR/models}"

DO_FULL=0
[ "${1:-}" = "--fullcascade" ] && DO_FULL=1

mkdir -p "$DATA/trackermom" "$OUT_DIR"

echo "=========================================="
echo " PIXELAV inputs:  $PARTICLE @ ${ETAG} GeV"
echo " events: $N_EVENTS   seed: ${CALOMAPS_SEED:-default of the steering files}"
echo "=========================================="

echo "[1/3] tracker-SD simulation (per-crossing momentum)"
TRK_ROOT="$DATA/trackermom/trackermom_${TAG}${ETAG}${NTAG}${SI_TAG}.root"
ddsim --compactFile "$COMPACT" --steeringFile "$REPO_DIR/sim/run_sim_trackermom.py" \
      --numberOfEvents "$N_EVENTS" --outputFile "$TRK_ROOT"

echo "[2/3] extract -> .npz"
TRK_NPZ="$OUT_DIR/trackermom_${TAG}${ETAG}${NTAG}${SI_TAG}.npz"
PARTICLES=()
if [ "$N_EVENTS" -gt 1 ]; then
  PARTICLES=(--particles hit)
  echo "      $N_EVENTS events: the npz keeps only the MCParticles with a tracker hit (--particles hit)"
fi
"$PY" "$REPO_DIR/analysis/extract_trackermom.py" "$TRK_ROOT" "$TRK_NPZ" --home "$GEO_HOME" \
      ${PARTICLES[@]+"${PARTICLES[@]}"}
VALID=0
"$PY" "$REPO_DIR/analysis/validate_crossings.py" "$TRK_NPZ" || VALID=$?
if [ "$VALID" = "3" ]; then
  echo "WARNING: too few clean crossings to confirm the silicon thickness from the path lengths" >&2
elif [ "$VALID" != "0" ]; then
  echo "ERROR: $TRK_NPZ failed analysis/validate_crossings.py; no deck written" >&2
  exit 1
fi

echo "[3/3] convert -> PIXELAV deck (variant C tracker, badeaa3 layout)"
SEG_PREFIX="$OUT_DIR/pixelav_segments_${TAG}${ETAG}${NTAG}${SI_TAG}"
"$PY" "$REPO_DIR/analysis/pixelav_converter.py" "$TRK_NPZ" "$SEG_PREFIX"

if [ "$DO_FULL" = "1" ]; then
  echo "[+] full calorimeter cascade (for notebook 05 / notebook 07)"
  mkdir -p "$DATA/fullcascade"
  FC_ROOT="$DATA/fullcascade/fullcascade_${TAG}${ETAG}_1evt${SI_TAG}.root"
  ddsim --compactFile "$COMPACT" --steeringFile "$REPO_DIR/sim/run_sim_fullcascade.py" \
        --numberOfEvents 1 --outputFile "$FC_ROOT"
  "$PY" "$REPO_DIR/analysis/extract_cascade.py" "$FC_ROOT" \
        "$OUT_DIR/fullcascade_${TAG}${ETAG}_1evt${SI_TAG}.npz" --home "$GEO_HOME"
fi

echo
echo "DONE. PIXELAV hand-off package:"
echo "  deck    : $SEG_PREFIX.pixelav.txt"
echo "            (7 cols: cot_alpha cot_beta ppion[GeV/c] flipped modx=v[um] mody=u[um] pT=|p|[GeV/c])"
echo "  legend  : $SEG_PREFIX.pixelav.txt.columns.txt   (units and definitions)"
echo "  records : $SEG_PREFIX.json / .csv   (18 fields per crossing)"
echo "  source  : $TRK_NPZ   (MCParticle cascade + tracker-hit arrays)"
