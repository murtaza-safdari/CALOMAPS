#!/bin/bash

# ==========================================
# SIMULATION PARAMETERS
# ==========================================
NUM_JOBS=200           # Number of parallel jobs (CPU cores to use)
EVENTS_PER_JOB=100    # Events per job
DATASET_NAME="data_spectrum_100um_400GeV"
OUT_DIR="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}/${DATASET_NAME}"

TOTAL_EVENTS=$(($NUM_JOBS * $EVENTS_PER_JOB))

echo "=========================================="
echo " Starting DD4hep Simulation Pipeline"
echo " Total Events: $TOTAL_EVENTS"
echo " Parallel Jobs: $NUM_JOBS"
echo "=========================================="

# Create output directory safely
mkdir -p $OUT_DIR

# Clean up old simulation files to prevent mixing datasets
echo "-> Cleaning up old ROOT and log files in $OUT_DIR/..."
rm -f $OUT_DIR/sim_photons_part*.root
rm -f $OUT_DIR/log_job*.txt

echo "-> Spawning background jobs..."

# Loop to submit jobs
for ((i=1; i<=NUM_JOBS; i++)); do
    ddsim --compactFile SiD_TestBeam.xml \
          --steeringFile run_sim.py \
          -N $EVENTS_PER_JOB \
          --random.seed $RANDOM \
          --outputFile $OUT_DIR/sim_photons_part${i}.root > $OUT_DIR/log_job${i}.txt 2>&1 &
done

echo "-> All jobs submitted! Waiting for Geant4 to finish..."

# Wait for all background processes to complete
wait

echo "=========================================="
echo " Simulation Complete!"
echo " Your ROOT files are ready in $OUT_DIR/"
echo "=========================================="

touch SIM_COMPLETE.txt