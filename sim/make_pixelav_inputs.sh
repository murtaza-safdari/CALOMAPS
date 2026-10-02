#!/bin/bash
# make_pixelav_inputs.sh — one command: cascade simulation -> .npz -> PIXELAV deck, or PIXELAV
# deposit jobs.
#
# Replaces the five manual steps (handbook §10.1: two ddsim runs + two extractors +
# the converter). Particle + energy come from the same env vars as the rest of the
# pipeline (no file edits):
#   CALOMAPS_PIXELAV_ROUTE    track (default): one tracker hit per sensor crossing, and a deck of
#                             tracks along which PIXELAV generates its own ionisation;
#                             deposits: one tracker hit per Geant4 step (CALOMAPS_TRACKER_HITS=step
#                             in sim/run_sim_trackermom.py), and jobs of Geant4's own step deposits
#                             for PIXELAV's drift and readout (analysis/pixelav/deposit_jobs.py)
#   CALOMAPS_GUN_PARTICLE     particle name (default gamma; e.g. pi+, pi-, proton)
#   CALOMAPS_GUN_ENERGY_GEV   beam energy in GeV (default 50; legacy alias CALOMAPS_GUN_GEV)
#   CALOMAPS_SI_THICKNESS_UM  ECal sensor thickness in um (default: the repo geometry, 320). Any
#                             other value simulates a variant geometry under $CALOMAPS_DATA_BASE
#                             (see sim/geometry_variant.sh) and appends _si<T>um to every file name below
#   CALOMAPS_SI_CUT_UM        Geant4 production cut in the sensor silicon in um (track route
#                             default: none, the run's own cut; deposits route default: 1, so delta
#                             rays above about 1 um of range are tracked). A cut simulates a variant
#                             geometry whose silicon has a region of its own and appends _cut<C>um
#                             to every file name below
#   CALOMAPS_SI_STEP_UM       Geant4 maximum step length in the sensor silicon in um (track route
#                             default: none, the calorimeter's 5 mm; deposits route default: 5, so
#                             every step is close to the straight segment the jobs make of it). A
#                             step limit simulates a variant geometry whose silicon has a limitset of
#                             its own and appends _stepmax<S>um to every file name below
#   CALOMAPS_N_EVENTS         events to simulate (default 1); every name below carries _<N>evt. Above
#                             1 the npz keeps only the MCParticles with a tracker hit
#                             (extract_trackermom.py --particles hit), without daughter links
#   CALOMAPS_SEED             random seed of both steering files (an integer; default: theirs, 424242)
#   CALOMAPS_OUTPUT_DIR       directory for the npz, records and deck (default $CALOMAPS_HOME/models).
#                             50 GeV photons take about 9 MB per event there with every MCParticle,
#                             3.7 MB with --particles hit (mostly the .json and .csv records)
# track route only:
#   CALOMAPS_DECK_SELECT      which crossings the deck holds (pixelav_converter.py --select):
#                             through = only those PIXELAV's straight track through the full silicon
#                             thickness reproduces, their true path within the tolerance of that line;
#                             all = every crossing the driver simulates (default: the converter's, through)
#   CALOMAPS_PATH_TOLERANCE   largest |true path / straight chord - 1| of a `through` deck
#                             (pixelav_converter.py --path-tolerance; default: the converter's, 0.05)
# deposits route only (the npz always keeps only the MCParticles with a tracker hit):
#   PIXELAV_RUN_DIR           PIXELAV run directory whose ppixel2.init gives the pixel pitches and
#                             the thickness (within 1 um of the simulated silicon's); else
#   CALOMAPS_PIXEL_X_UM, CALOMAPS_PIXEL_Y_UM   the pitches along PIXELAV x (= v, along the barrel
#                             axis) and y (= u, across the stave), with the simulated thickness.
#                             One of the two is required; the run stops before simulating without
#   CALOMAPS_JOB_PARTS        write the jobs in this many files of about equal energy (default 1)
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
#   CALOMAPS_PIXELAV_ROUTE=deposits CALOMAPS_SI_THICKNESS_UM=100 PIXELAV_RUN_DIR=/path/to/run \
#     bash sim/make_pixelav_inputs.sh
#     -> trackermom_gamma50_1evt_step_si100um_cut1um_stepmax5um.{root,npz},
#        pixelav_jobs_gamma50_1evt_si100um_cut1um_stepmax5um.*
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
#                                        and the selection counts: crossings written and left out, why
#   pixelav_segments_<tag><E>_<N>evt.{json,csv}       per-crossing records, every crossing (20 fields
#                                        each; deck_line = its line in the deck or 0, deck_drop = why not)
# The deposits route instead (the npz validated the same way):
#   trackermom_<tag><E>_<N>evt_step.{root,npz}       one tracker hit per Geant4 step
#   pixelav_jobs_<tag><E>_<N>evt.jobs.txt (or .part000.txt ...)   the deposit jobs (the hand-off
#                                        file): "J <job_id> <n_seg>", then n_seg lines
#                                        x1 y1 z1 x2 y2 z2 [um, window frame] edep [eV]
#   pixelav_jobs_<tag><E>_<N>evt.jobs.npz      per job and per particle group: event, sensor
#                                        plane, window, energy; the frame's constants
#   pixelav_jobs_<tag><E>_<N>evt.summary.txt   counts and the energy and window checks
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

ROUTE="${CALOMAPS_PIXELAV_ROUTE:-track}"
case "$ROUTE" in
  track)    HITS="crossing"; HTAG="" ;;
  deposits) HITS="step";     HTAG="_step"
            export CALOMAPS_SI_CUT_UM="${CALOMAPS_SI_CUT_UM:-1}" CALOMAPS_SI_STEP_UM="${CALOMAPS_SI_STEP_UM:-5}" ;;
  *) echo "ERROR: CALOMAPS_PIXELAV_ROUTE='$ROUTE' is neither track nor deposits" >&2; exit 1 ;;
esac
if [ -n "${CALOMAPS_TRACKER_HITS:-}" ] && [ "$CALOMAPS_TRACKER_HITS" != "$HITS" ]; then
  echo "ERROR: CALOMAPS_TRACKER_HITS=$CALOMAPS_TRACKER_HITS, but the $ROUTE route needs $HITS hits" >&2
  exit 1
fi
export CALOMAPS_TRACKER_HITS="$HITS"    # read by sim/run_sim_trackermom.py
SENSOR=()
if [ "$ROUTE" = "deposits" ]; then
  if [ -n "${PIXELAV_RUN_DIR:-}" ]; then
    SENSOR=(--run-dir "$PIXELAV_RUN_DIR")
  elif [ -n "${CALOMAPS_PIXEL_X_UM:-}" ] && [ -n "${CALOMAPS_PIXEL_Y_UM:-}" ]; then
    SENSOR=(--pitch-x-um "$CALOMAPS_PIXEL_X_UM" --pitch-y-um "$CALOMAPS_PIXEL_Y_UM")
  else
    echo "ERROR: the deposits route needs the PIXELAV sensor: set PIXELAV_RUN_DIR to a PIXELAV run" >&2
    echo "       directory (ppixel2.init, wgt_pot.init), or CALOMAPS_PIXEL_X_UM and CALOMAPS_PIXEL_Y_UM" >&2
    echo "       to the pixel pitches along x (= v, the barrel axis) and y (= u, across the stave)" >&2
    exit 1
  fi
fi
source "$SCRIPT_DIR/geometry_variant.sh" || exit 1   # COMPACT, GEO_HOME, SI_TAG
if [ "$ROUTE" = "deposits" ]; then
  "$PY" "$REPO_DIR/analysis/pixelav/deposit_jobs.py" --check-sensor "$GEO_HOME" "${SENSOR[@]}" || exit 1
fi

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

if [ "$ROUTE" = "deposits" ]; then
  echo "[1/3] tracker-SD simulation (one hit per Geant4 step)"
else
  echo "[1/3] tracker-SD simulation (per-crossing momentum)"
fi
TRK_ROOT="$DATA/trackermom/trackermom_${TAG}${ETAG}${NTAG}${HTAG}${SI_TAG}.root"
ddsim --compactFile "$COMPACT" --steeringFile "$REPO_DIR/sim/run_sim_trackermom.py" \
      --numberOfEvents "$N_EVENTS" --outputFile "$TRK_ROOT"

echo "[2/3] extract -> .npz"
TRK_NPZ="$OUT_DIR/trackermom_${TAG}${ETAG}${NTAG}${HTAG}${SI_TAG}.npz"
PARTICLES=()
if [ "$ROUTE" = "deposits" ]; then
  PARTICLES=(--particles hit)
elif [ "$N_EVENTS" -gt 1 ]; then
  PARTICLES=(--particles hit)
  echo "      $N_EVENTS events: the npz keeps only the MCParticles with a tracker hit (--particles hit)"
fi
"$PY" "$REPO_DIR/analysis/extract_trackermom.py" "$TRK_ROOT" "$TRK_NPZ" --home "$GEO_HOME" \
      --hits "$HITS" ${PARTICLES[@]+"${PARTICLES[@]}"}
VALID=0
"$PY" "$REPO_DIR/analysis/validate_crossings.py" "$TRK_NPZ" || VALID=$?
if [ "$VALID" = "3" ] && [ "$ROUTE" = "deposits" ]; then
  echo "WARNING: too few steps to confirm the silicon thickness from their depths" >&2
elif [ "$VALID" = "3" ]; then
  echo "WARNING: too few clean crossings to confirm the silicon thickness from the path lengths" >&2
elif [ "$VALID" != "0" ]; then
  echo "ERROR: $TRK_NPZ failed analysis/validate_crossings.py; no $([ "$ROUTE" = deposits ] && echo jobs || echo deck) written" >&2
  exit 1
fi

if [ "$ROUTE" = "deposits" ]; then
  echo "[3/3] step deposits -> PIXELAV deposit jobs"
  JOB_PREFIX="$OUT_DIR/pixelav_jobs_${TAG}${ETAG}${NTAG}${SI_TAG}"
  PARTS=()
  [ -n "${CALOMAPS_JOB_PARTS:-}" ] && PARTS=(--parts "$CALOMAPS_JOB_PARTS")
  "$PY" "$REPO_DIR/analysis/pixelav/deposit_jobs.py" "$TRK_NPZ" "$JOB_PREFIX" "${SENSOR[@]}" \
        ${PARTS[@]+"${PARTS[@]}"} || exit 1
fi

if [ "$ROUTE" = "track" ]; then
  echo "[3/3] convert -> PIXELAV deck (variant C tracker, badeaa3 layout)"
  SEG_PREFIX="$OUT_DIR/pixelav_segments_${TAG}${ETAG}${NTAG}${SI_TAG}"
  SELECT=()
  [ -n "${CALOMAPS_DECK_SELECT:-}" ] && SELECT+=(--select "$CALOMAPS_DECK_SELECT")
  [ -n "${CALOMAPS_PATH_TOLERANCE:-}" ] && SELECT+=(--path-tolerance "$CALOMAPS_PATH_TOLERANCE")
  "$PY" "$REPO_DIR/analysis/pixelav_converter.py" "$TRK_NPZ" "$SEG_PREFIX" ${SELECT[@]+"${SELECT[@]}"}
fi

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
if [ "$ROUTE" = "deposits" ]; then
  echo "DONE. PIXELAV deposit jobs:"
  echo "  jobs    : $JOB_PREFIX.jobs.txt (or .partNNN.txt)"
  echo "            (J <job_id> <n_seg>, then n_seg lines x1 y1 z1 x2 y2 z2 [um, window frame] edep [eV])"
  echo "  tables  : $JOB_PREFIX.jobs.npz   (per job and per particle group; frame constants)"
  echo "  summary : $JOB_PREFIX.summary.txt"
  echo "  source  : $TRK_NPZ   (step hits + the MCParticles with one)"
  exit 0
fi
echo "DONE. PIXELAV hand-off package:"
echo "  deck    : $SEG_PREFIX.pixelav.txt"
echo "            (7 cols: cot_alpha cot_beta ppion[GeV/c] flipped modx=v[um] mody=u[um] pT=|p|[GeV/c])"
echo "  legend  : $SEG_PREFIX.pixelav.txt.columns.txt   (units, definitions, selection counts)"
echo "  records : $SEG_PREFIX.json / .csv   (every crossing, 20 fields; deck_line, deck_drop)"
echo "  source  : $TRK_NPZ   (MCParticle cascade + tracker-hit arrays)"
