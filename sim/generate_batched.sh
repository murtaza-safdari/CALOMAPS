#!/bin/bash

# --- BATCH CONFIGURATION ---
TOTAL_JOBS=1000             # Total number of root files you want
EVENTS_PER_JOB=20           # Keeps memory footprint low per file
CONCURRENT_JOBS=20          # How many jobs to run simultaneously
DATASET_NAME="data_spectrum_100um_400GeV"
OUT_DIR="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}/${DATASET_NAME}"

echo "=========================================="
echo " Starting DD4hep Simulation Pipeline"
echo " Total Jobs: $TOTAL_JOBS | Events/Job: $EVENTS_PER_JOB | Batch Size: $CONCURRENT_JOBS"
echo "=========================================="

# Create output directory safely
mkdir -p $OUT_DIR

# Clean up old simulation files to prevent mixing datasets
echo "-> Cleaning up old ROOT and log files in $OUT_DIR/..."
rm -f $OUT_DIR/sim_photons_part*.root
rm -f $OUT_DIR/log_job*.txt

# Counter to track active background jobs
active_jobs=0

for (( i=1; i<=TOTAL_JOBS; i++ ))
do
    echo "Submitting Job $i..."

    ddsim --compactFile SiD_TestBeam.xml \
          --steeringFile run_sim.py \
          -N $EVENTS_PER_JOB \
          --random.seed $RANDOM \
          --outputFile $OUT_DIR/sim_photons_part${i}.root \
          > $OUT_DIR/log_job${i}.txt 2>&1 &
          
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
echo "=========================================="

touch SIM_COMPLETE.txt