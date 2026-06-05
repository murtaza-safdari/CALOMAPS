#!/bin/bash

# --- BATCH CONFIGURATION ---
TOTAL_JOBS=1000             # Total number of root files you want
EVENTS_PER_JOB=20           # Keeps memory footprint low per file
CONCURRENT_JOBS=20          # How many jobs to run simultaneously

# --- PARTICLE / DATASET ---
# The particle type is read from CALOMAPS_GUN_PARTICLE (default "gamma") and passed
# through to run_sim.py. Defaults reproduce the original photon dataset exactly:
#   gamma -> data_spectrum_100um_400GeV/sim_photons_part*.root
# A non-photon run gets its own dataset dir + file prefix, e.g.:
#   CALOMAPS_GUN_PARTICLE=pi+ bash generate_batched.sh
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
FILE_PREFIX="sim_${TAG}_part"
OUT_DIR="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}/${DATASET_NAME}"

# Resolve the compact geometry + steering by ABSOLUTE path (relative to this script),
# so the script runs correctly from any working directory. SiD_TestBeam.xml lives in
# geometry/ and run_sim.py in sim/, so bare relative names can't both resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPACT="$REPO_DIR/geometry/SiD_TestBeam.xml"
STEER="$REPO_DIR/sim/run_sim.py"

echo "=========================================="
echo " Starting DD4hep Simulation Pipeline"
echo " Particle: $CALOMAPS_GUN_PARTICLE | Dataset: $DATASET_NAME"
echo " Total Jobs: $TOTAL_JOBS | Events/Job: $EVENTS_PER_JOB | Batch Size: $CONCURRENT_JOBS"
echo "=========================================="

# Create output directory safely
mkdir -p "$OUT_DIR"

# Clean up old simulation files to prevent mixing datasets
echo "-> Cleaning up old ROOT and log files in $OUT_DIR/..."
rm -f "$OUT_DIR/${FILE_PREFIX}"*.root
rm -f "$OUT_DIR"/log_job*.txt

# Counter to track active background jobs
active_jobs=0

for (( i=1; i<=TOTAL_JOBS; i++ ))
do
    echo "Submitting Job $i..."

    ddsim --compactFile "$COMPACT" \
          --steeringFile "$STEER" \
          -N $EVENTS_PER_JOB \
          --random.seed $RANDOM \
          --outputFile "$OUT_DIR/${FILE_PREFIX}${i}.root" \
          > "$OUT_DIR/log_job${i}.txt" 2>&1 &

    # Increment our active job counter
    active_jobs=$((active_jobs + 1))

    # If we hit our concurrency limit, stop submitting and wait
    if [[ $active_jobs -eq $CONCURRENT_JOBS ]]; then
        echo ">>> Batch limit ($CONCURRENT_JOBS) reached. Waiting for batch to finish..."
        wait  # This pauses the script until all background jobs complete
        echo ">>> Batch complete! Moving to next set."
        active_jobs=0 # Reset counter for the next batch
    fi
done

# Catch any leftover jobs if TOTAL_JOBS isn't perfectly divisible by CONCURRENT_JOBS
echo ">>> Waiting for the final partial batch to finish..."
wait

echo "=========================================="
echo " Simulation Complete! All jobs completed successfully!"
echo " Output: $OUT_DIR/${FILE_PREFIX}*.root"
echo "=========================================="

touch "$OUT_DIR/SIM_COMPLETE.txt"
