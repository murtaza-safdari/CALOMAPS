#!/bin/bash

# ==========================================
# SIMULATION PARAMETERS
# ==========================================
# Sizable via env (CALOMAPS_NJOBS / CALOMAPS_NEVENTS) so you can make a quick small dataset.
NUM_JOBS="${CALOMAPS_NJOBS:-200}"          # Total number of root files you want
EVENTS_PER_JOB="${CALOMAPS_NEVENTS:-100}"  # Events per job
CONCURRENT_JOBS="${CALOMAPS_NCONCURRENT:-20}"  # How many run at once (a typical EAF pod has
                                               # a handful of cores; 200 ddsim at once thrashes)
# Job i is seeded SEED_BASE+i: every job unique, and the whole dataset reproducible.
# (bash $RANDOM is only 15-bit, so repeated draws can collide and silently duplicate
# whole files.) Set CALOMAPS_SEED_BASE to generate a statistically independent dataset.
SEED_BASE="${CALOMAPS_SEED_BASE:-0}"

# Resolve the compact geometry + steering by ABSOLUTE path (relative to this script),
# so the script runs correctly from any working directory. SiD_TestBeam.xml lives in
# geometry/ and run_sim.py in sim/, so bare relative names can't both resolve.
# CALOMAPS_SI_THICKNESS_UM=T (unset/320 = the repo geometry) simulates the thickness-T variant
# and tags the dataset dir and file names with _si<T>um, e.g. T=100:
#   data_spectrum_100um_400GeV_si100um/sim_photons_si100um_part*.root
# where "100um" is the pixel pitch and "si100um" the silicon thickness. CALOMAPS_SI_CUT_UM=C
# gives the silicon a Geant4 production cut of its own and CALOMAPS_SI_STEP_UM=S a maximum step
# length of its own (unset: the run's cut and the calorimeter's 5 mm), and append _cut<C>um and
# _stepmax<S>um to the same names. The run prints the variant's compact (see
# sim/geometry_variant.sh); extract with extract_readouts.py
# --home <the directory above that compact's geometry/> --mip-energy <MPV for T um>.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STEER="$REPO_DIR/sim/run_sim.py"
source "$SCRIPT_DIR/geometry_variant.sh" || exit 1   # COMPACT, GEO_HOME, SI_TAG

# --- PARTICLE / DATASET ---
# Particle type is read from CALOMAPS_GUN_PARTICLE (default "gamma") and passed
# through to run_sim.py. Defaults reproduce the original photon dataset exactly:
#   gamma -> data_spectrum_100um_400GeV/sim_photons_part*.root
# A non-photon run gets its own dataset dir + file prefix, e.g.:
#   CALOMAPS_GUN_PARTICLE=pi+ bash generate_dataset.sh
#     -> data_spectrum_100um_400GeV_piplus/sim_piplus_part*.root
export CALOMAPS_GUN_PARTICLE="${CALOMAPS_GUN_PARTICLE:-gamma}"
case "$CALOMAPS_GUN_PARTICLE" in
    gamma) TAG="photons" ;;
    pi+)   TAG="piplus" ;;
    pi-)   TAG="piminus" ;;
    *)     TAG="$(echo "$CALOMAPS_GUN_PARTICLE" | sed 's/+/plus/g; s/-/minus/g')" ;;
esac
if [ "$CALOMAPS_GUN_PARTICLE" = "gamma" ]; then
    DATASET_NAME="${CALOMAPS_DATASET_NAME:-data_spectrum_100um_400GeV}"
else
    DATASET_NAME="${CALOMAPS_DATASET_NAME:-data_spectrum_100um_400GeV_${TAG}}"
fi
DATASET_NAME="$(si_tagged "$DATASET_NAME")"
FILE_PREFIX="sim_${TAG}${SI_TAG}_part"
OUT_DIR="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}/${DATASET_NAME}"

TOTAL_EVENTS=$(($NUM_JOBS * $EVENTS_PER_JOB))

# Refuse to run without ddsim -- BEFORE the destructive cleanup below, so an
# unsourced environment can't delete an existing dataset and then produce nothing.
if ! command -v ddsim >/dev/null 2>&1; then
    echo "ERROR: ddsim not found in PATH. Source setup/setup_calomaps.sh first." >&2
    exit 1
fi

echo "=========================================="
echo " Starting DD4hep Simulation Pipeline"
echo " Particle: $CALOMAPS_GUN_PARTICLE | Dataset: $DATASET_NAME"
echo " Total Events: $TOTAL_EVENTS"
echo " Total Jobs: $NUM_JOBS | Concurrent: $CONCURRENT_JOBS"
echo "=========================================="

# Create output directory safely
mkdir -p "$OUT_DIR"

# Clean up old simulation files to prevent mixing datasets
echo "-> Cleaning up old ROOT and log files in $OUT_DIR/..."
rm -f "$OUT_DIR/${FILE_PREFIX}"*.root
rm -f "$OUT_DIR"/log_job*.txt
rm -f "$OUT_DIR"/.fail_job* "$OUT_DIR/SIM_COMPLETE.txt"

echo "-> Spawning background jobs ($CONCURRENT_JOBS at a time)..."

# Loop to submit jobs, at most CONCURRENT_JOBS at once
active_jobs=0
for ((i=1; i<=NUM_JOBS; i++)); do
    # A failed job leaves a .fail_job marker so the final status check can report it.
    ( ddsim --compactFile "$COMPACT" \
            --steeringFile "$STEER" \
            -N $EVENTS_PER_JOB \
            --random.seed $((SEED_BASE + i)) \
            --outputFile "$OUT_DIR/${FILE_PREFIX}${i}.root" > "$OUT_DIR/log_job${i}.txt" 2>&1 \
      || touch "$OUT_DIR/.fail_job${i}" ) &
    active_jobs=$((active_jobs + 1))
    if [[ $active_jobs -eq $CONCURRENT_JOBS ]]; then
        wait
        active_jobs=0
    fi
done

echo "-> All jobs submitted! Waiting for Geant4 to finish..."

# Wait for all background processes to complete
wait

# Only report success (and write the SIM_COMPLETE marker) if every job actually
# produced its output file and none left a failure marker.
n_failed=$(ls "$OUT_DIR"/.fail_job* 2>/dev/null | wc -l)
n_produced=$(ls "$OUT_DIR/${FILE_PREFIX}"*.root 2>/dev/null | wc -l)
if [ "$n_failed" -gt 0 ] || [ "$n_produced" -lt "$NUM_JOBS" ]; then
    echo "=========================================="
    echo " Simulation FINISHED WITH ERRORS:"
    echo "   failed jobs:  $n_failed  (see $OUT_DIR/.fail_job* and matching log_job*.txt)"
    echo "   output files: $n_produced of $NUM_JOBS expected"
    echo " NOT writing SIM_COMPLETE.txt."
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo " Simulation Complete! All $NUM_JOBS jobs produced output."
echo " Output: $OUT_DIR/${FILE_PREFIX}*.root"
echo "=========================================="

touch "$OUT_DIR/SIM_COMPLETE.txt"
