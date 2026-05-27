#!/bin/bash
# CALOMAPS environment setup. Source this in a JupyterLab terminal:
#     source ~/setup_calomaps.sh
#
# What it does:
#   1. Loads Key4hep 2026-02-01 from CVMFS (Geant4, DD4hep, uproot, ROOT, PyTorch CPU)
#   2. Injects the OpenGL workaround library (~/lib_hack) for DD4hep visualization
#   3. Exports CALOMAPS_HOME (project root) and CALOMAPS_DATA_BASE (where ROOT data lives)
#   4. cd's into the simulation working directory

echo "Loading Key4hep environment..."
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-02-01

echo "Injecting OpenGL library hack..."
export LD_LIBRARY_PATH="$HOME/lib_hack:$LD_LIBRARY_PATH"

# Where the simulation output datasets live. Override by exporting before sourcing.
export CALOMAPS_DATA_BASE="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}"

# Project root on /nashome (cloned from github). First letter of $USER picks the
# /nashome/<X>/<username>/ bucket.
USER_LETTER="${USER:0:1}"
export CALOMAPS_HOME="${CALOMAPS_HOME:-/nashome/${USER_LETTER}/${USER}/CALOMAPS}"

echo "Navigating to simulation working directory..."
cd "${CALOMAPS_HOME}/sim"

echo
echo "Environment ready. You can now run ddsim."
echo "  CALOMAPS_HOME      = $CALOMAPS_HOME"
echo "  CALOMAPS_DATA_BASE = $CALOMAPS_DATA_BASE"
echo "  cwd                = $(pwd)"
