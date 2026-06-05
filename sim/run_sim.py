from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, mm, deg
import os

SIM = DD4hepSimulation()

# ==========================================
# PARTICLE GUN CONFIGURATION
# ==========================================
# Particle type and energy are read from environment variables so the SAME steering
# file can generate photons, pions, protons, ... without editing this file. The
# defaults reproduce the original photon spectrum (gamma, 5-400 GeV) exactly, so
# existing datasets and notebooks are unaffected.
#
#   CALOMAPS_GUN_PARTICLE   particle name           (default "gamma"; e.g. "pi+", "pi-", "proton")
#   CALOMAPS_GUN_PMIN_GEV   momentum spectrum min   (default 5.0)
#   CALOMAPS_GUN_PMAX_GEV   momentum spectrum max   (default 400.0)
#   CALOMAPS_GUN_ENERGY_GEV if set, a mono-energetic beam at this energy (overrides PMIN/PMAX)
#
# Examples:
#   CALOMAPS_GUN_PARTICLE=pi+ ddsim --compactFile SiD_TestBeam.xml --steeringFile run_sim.py -N 100 ...
#   CALOMAPS_GUN_PARTICLE=pi- CALOMAPS_GUN_ENERGY_GEV=50 ddsim ... -N 1 ...
PARTICLE = os.environ.get("CALOMAPS_GUN_PARTICLE", "gamma")
PMIN     = float(os.environ.get("CALOMAPS_GUN_PMIN_GEV", "5.0"))
PMAX     = float(os.environ.get("CALOMAPS_GUN_PMAX_GEV", "400.0"))
ENERGY   = os.environ.get("CALOMAPS_GUN_ENERGY_GEV", "")  # set for a mono-energetic beam

SIM.enableGun = True
SIM.gun.particle = PARTICLE
SIM.gun.position = (0, 0, 0)

# Energy: a fixed energy if CALOMAPS_GUN_ENERGY_GEV is set, otherwise a uniform
# momentum spectrum between PMIN and PMAX.
if ENERGY:
    SIM.gun.energy = float(ENERGY) * GeV
else:
    SIM.gun.momentumMin = PMIN * GeV
    SIM.gun.momentumMax = PMAX * GeV

# Pencil beam along the +y axis (theta=90, phi=90) so it hits the "top" barrel face.
# To target a specific z-slice instead, set SIM.gun.position = (0, 0, 100 * mm).
SIM.gun.distribution = "uniform"
SIM.gun.phiMin = 90 * deg
SIM.gun.phiMax = 90 * deg
SIM.gun.thetaMin = 90 * deg
SIM.gun.thetaMax = 90 * deg

# ==========================================
# PHYSICS & TRACKING
# ==========================================
# FTFP_BERT: standard calorimeter physics list (EM + hadronic).
SIM.physicsList = "FTFP_BERT"

# ==========================================
# OUTPUT
# ==========================================
# Standalone default name (derived from the particle). The batch scripts
# generate_dataset.sh / generate_batched.sh override this with --outputFile.
_tag = {"gamma": "photons", "pi+": "piplus", "pi-": "piminus"}.get(
    PARTICLE, PARTICLE.replace("+", "plus").replace("-", "minus"))
SIM.outputFile = "sim_data_%s.root" % _tag
