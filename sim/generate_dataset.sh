#!/bin/bash

# ==========================================
# SIMULATION PARAMETERS
# ==========================================
NUM_JOBS=200           # Number of parallel jobs (CPU cores to use)
EVENTS_PER_JOB=100     # Events per job

# --- PARTICLE / DATASET ---
# Particle type is read from CALOMAPS_GUN_PARTICLE (default "gamma") and passed
# through to run_sim.py. Defaults reproduce the original photon dataset exactly:
#   gamma -> data_spectrum_100um_500GeV/sim_photons_part*.root
# A non-photon run gets its own dataset dir + file prefix, e.g.:
#   CALOMAPS_GUN_PARTICLE=pi+ bash generate_dataset.sh
#     -> data_spectrum_100um_500GeV_piplus/sim_piplus_part*.root
export CALOMAPS_GUN_PARTICLE="${CALOMAPS_GUN_PARTICLE:-gamma}"
case "$CALOMAPS_GUN_PARTICLE" in
    gamma) TAG="photons" ;;
    pi+)   TAG="piplus" ;;
    pi-)   TAG="piminus" ;;
    *)     TAG="$(echo "$CALOMAPS_GUN_PARTICLE" | sed 's/+/plus/g; s/-/minus/g')" ;;
esac
if [ "$CALOMAPS_GUN_PARTICLE" = "gamma" ]; then
    DATASET_NAME="${CALOMAPS_DATASET_NAME:-data_spectrum_100um_500GeV}"
else
    DATASET_NAME="${CALOMAPS_DATASET_NAME:-data_spectrum_100um_500GeV_${TAG}}"
fi
FILE_PREFIX="sim_${TAG}_part"
OUT_DIR="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}/${DATASET_NAME}"

# Resolve the compact geometry + steering by ABSOLUTE path (relative to this script),
# so the script runs correctly from any working directory. SiD_TestBeam.xml lives in
# geometry/ and run_sim.py in sim/, so bare relative names can't both resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPACT="$REPO_DIR/geometry/SiD_TestBeam.xml"
STEER="$REPO_DIR/sim/run_sim.py"

TOTAL_EVENTS=$(($NUM_JOBS * $EVENTS_PER_JOB))

echo "=========================================="
echo " Starting DD4hep Simulation Pipeline"
echo " Particle: $CALOMAPS_GUN_PARTICLE | Dataset: $DATASET_NAME"
echo " Total Events: $TOTAL_EVENTS"
echo " Parallel Jobs: $NUM_JOBS"
echo "=========================================="

# Create output directory safely
mkdir -p "$OUT_DIR"

# Clean up old simulation files to prevent mixing datasets
echo "-> Cleaning up old ROOT and log files in $OUT_DIR/..."
rm -f "$OUT_DIR/${FILE_PREFIX}"*.root
rm -f "$OUT_DIR"/log_job*.txt

echo "-> Spawning background jobs..."

# Loop to submit jobs
for ((i=1; i<=NUM_JOBS; i++)); do
    ddsim --compactFile "$COMPACT" \
          --steeringFile "$STEER" \
          -N $EVENTS_PER_JOB \
          --random.seed $RANDOM \
          --outputFile "$OUT_DIR/${FILE_PREFIX}${i}.root" > "$OUT_DIR/log_job${i}.txt" 2>&1 &
done

echo "-> All jobs submitted! Waiting for Geant4 to finish..."

# Wait for all background processes to complete
wait

echo "=========================================="
echo " Simulation Complete!"
echo " Output: $OUT_DIR/${FILE_PREFIX}*.root"
echo "=========================================="

touch "$OUT_DIR/SIM_COMPLETE.txt"
