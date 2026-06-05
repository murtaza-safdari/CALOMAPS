"""DDSim steering for PER-CROSSING MOMENTUM truth (experiment "B"), EDM4hep format.

The standard calorimeter readout (run_sim_fullcascade.py) writes CaloHitContributions that carry
step position/PDG/time but NO momentum (EDM4hep CaloHitContribution has no momentum field). To
get the real per-sensor-crossing momentum PIXELAV wants, this steering maps the ECalBarrel Si to
Geant4's TRACKER sensitive action instead of the calorimeter action:

    SIM.action.mapActions['ECalBarrel'] = 'Geant4TrackerWeightedAction'

A tracker SD produces SimTrackerHits, which DO carry momentum[3], position[3], pathLength, EDep,
time and a link to the producing MCParticle -- i.e. one record per charged-track crossing of a Si
sensor, with the true Geant4 momentum at that crossing. Same gun, physics, seed (424242) as
run_sim_fullcascade.py, so the shower is identical; only the Si readout differs.

This is a SEPARATE run from the calorimeter one: nb04 (shower display) and nb05's calo view still
use the calorimeter output; the PIXELAV deck (pixelav_converter.py) uses THIS tracker output for
real per-crossing momentum + entry + direction.

Run (from geometry/, on EAF, Key4hep + lib_hack on LD_LIBRARY_PATH):
    ddsim --compactFile SiD_TestBeam.xml --steeringFile ../sim/run_sim_trackermom.py --numberOfEvents 1

Keep this file pure ASCII (non-ASCII breaks the EDM4hep RunHeader std::map conversion).
"""
import os
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, deg

SIM = DD4hepSimulation()

# Reproducibility: same seed as the calorimeter cascade so the shower is identical.
SIM.random.seed = 424242

# ---- particle gun: single fixed-energy particle, +y pencil beam ----
# CALOMAPS_GUN_PARTICLE (default gamma) + CALOMAPS_GUN_ENERGY_GEV (default 50; legacy alias
# CALOMAPS_GUN_GEV accepted) let the SAME steering make per-crossing momentum for any particle.
# (No helper functions: ddsim exec()s this file with split globals/locals.)
_particle = os.environ.get("CALOMAPS_GUN_PARTICLE", "gamma")
_gev = float(os.environ.get("CALOMAPS_GUN_ENERGY_GEV", os.environ.get("CALOMAPS_GUN_GEV", "50.0")))
_tag = {"gamma": "gamma", "pi+": "piplus", "pi-": "piminus"}.get(_particle, _particle.replace("+", "plus").replace("-", "minus"))
SIM.enableGun = True
SIM.gun.particle = _particle
SIM.gun.energy = _gev * GeV
SIM.gun.position = (0, 0, 0)
SIM.gun.distribution = "uniform"
SIM.gun.phiMin = 90 * deg
SIM.gun.phiMax = 90 * deg
SIM.gun.thetaMin = 90 * deg
SIM.gun.thetaMax = 90 * deg

SIM.physicsList = "FTFP_BERT"

# ---- keep the whole cascade (same as run_sim_fullcascade.py) ----
SIM.part.userParticleHandler = ""
SIM.part.keepAllParticles = True
SIM.part.minimalKineticEnergy = 1.0
SIM.part.printEndTracking = False
SIM.part.printStartTracking = False

# ---- THE KEY LINE: read the ECal Si out as a TRACKER -> SimTrackerHits with momentum ----
# Geant4TrackerWeightedAction combines the steps of one sensor crossing into a single hit with an
# energy-weighted position and the track momentum; that is exactly one PIXELAV crossing record.
SIM.action.mapActions['ECalBarrel'] = 'Geant4TrackerWeightedAction'

# ---- output ----
_data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
_out_dir = os.path.join(_data_base, "trackermom")
os.makedirs(_out_dir, exist_ok=True)
SIM.outputFile = os.path.join(_out_dir, "trackermom_%s%g_1evt.root" % (_tag, _gev))
