"""DDSim steering for PER-CROSSING MOMENTUM truth (experiment "B"), EDM4hep format.

The standard calorimeter readout (run_sim_fullcascade.py) writes CaloHitContributions that carry
step position/PDG/time but NO momentum (EDM4hep CaloHitContribution has no momentum field). To
get the real per-sensor-crossing momentum PIXELAV wants, this steering maps the ECalBarrel Si to
Geant4's TRACKER sensitive action instead of the calorimeter action:

    SIM.action.mapActions['ECalBarrel'] = 'Geant4TrackerWeightedAction'

A tracker SD produces SimTrackerHits, which DO carry momentum[3], position[3], pathLength, EDep,
time and a link to the producing MCParticle -- i.e. one record per charged-track crossing of a Si
sensor, with the true Geant4 momentum at that crossing. (The hit position is the action's
energy-weighted COMBINED position over the crossing, ~the sensor mid-plane -- not the
entry-face point; see docs/pixelav_reference.md.) Same gun, physics and seed (424242) as
run_sim_fullcascade.py, so the shower is identical; only the Si readout differs.

This is a SEPARATE run from the calorimeter one: nb04 (shower display) and nb05's calo view still
use the calorimeter output; the PIXELAV deck (pixelav_converter.py) uses THIS tracker output for
real per-crossing momentum + entry + direction.

Run (from geometry/, on EAF, Key4hep + lib_hack on LD_LIBRARY_PATH):
    ddsim --compactFile SiD_TestBeam.xml --steeringFile ../sim/run_sim_trackermom.py --numberOfEvents 1

Environment knobs (all optional; the defaults reproduce the canonical 50 GeV photon run):
    CALOMAPS_GUN_PARTICLE    gun particle (gamma, pi+, pi-, ...)      (default gamma)
    CALOMAPS_GUN_ENERGY_GEV  gun energy in GeV                        (default 50)
    CALOMAPS_RANGECUT_MM     Geant4 production range cut in mm        (default: DDSim's 0.7)
    CALOMAPS_KEEP_ALL        1 keeps every track; 0 prunes by min-KE  (default 1)
    CALOMAPS_MIN_KE_MEV      truth-persistency KE floor in MeV        (default 1.0)
    CALOMAPS_DATA_BASE       output directory root                    (default ~/CALOMAPS-data)
See docs/handbook.md "Controlling which secondaries are produced and saved" for the physics
(production cut) vs. output-only (persistency floor) distinction.

Keep this file pure ASCII (non-ASCII breaks the EDM4hep RunHeader std::map conversion).
"""
import os
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, MeV, mm, deg

SIM = DD4hepSimulation()

# Reproducibility: same seed as the calorimeter cascade so the shower is identical.
SIM.random.seed = 424242

# ---- particle gun: single fixed-energy particle, +y pencil beam (both overridable) ----
_particle = os.environ.get("CALOMAPS_GUN_PARTICLE", "gamma")
_energy_gev = float(os.environ.get("CALOMAPS_GUN_ENERGY_GEV", "50.0"))
_tag = {"gamma": "gamma", "pi+": "piplus", "pi-": "piminus"}.get(
    _particle, _particle.replace("+", "plus").replace("-", "minus"))
SIM.enableGun = True
SIM.gun.particle = _particle
SIM.gun.energy = _energy_gev * GeV
SIM.gun.position = (0, 0, 0)
SIM.gun.distribution = "uniform"
SIM.gun.phiMin = 90 * deg
SIM.gun.phiMax = 90 * deg
SIM.gun.thetaMin = 90 * deg
SIM.gun.thetaMax = 90 * deg

SIM.physicsList = "FTFP_BERT"

# ---- Geant4 production range cut (PHYSICS knob) ----
# Below this range Geant4 does not create soft secondaries as tracks (their energy is deposited
# along the parent step instead). Lower => more soft delta-rays => more sensor crossings for
# PIXELAV, at higher CPU cost; higher => a coarser shower. This changes the simulated physics.
# DDSim default is 0.7*mm; overridden only if CALOMAPS_RANGECUT_MM is set.
_rangecut_mm = os.environ.get("CALOMAPS_RANGECUT_MM", "")
if _rangecut_mm:
    SIM.physics.rangecut = float(_rangecut_mm) * mm

# ---- keep the whole cascade (TRUTH PERSISTENCY -- output only, not physics) ----
# userParticleHandler="" is required for ECal secondaries to persist at all (they are born
# outside the tracking region). keepAllParticles=True writes every track; the
# minimalKineticEnergy floor only bites when keepAllParticles=False (set CALOMAPS_KEEP_ALL=0).
SIM.part.userParticleHandler = ""
SIM.part.keepAllParticles = os.environ.get("CALOMAPS_KEEP_ALL", "1") != "0"
SIM.part.minimalKineticEnergy = float(os.environ.get("CALOMAPS_MIN_KE_MEV", "1.0")) * MeV
SIM.part.printEndTracking = False
SIM.part.printStartTracking = False

# ---- THE KEY LINE: read the ECal Si out as a TRACKER -> SimTrackerHits with momentum ----
# Geant4TrackerWeightedAction combines the steps of one sensor crossing into a single hit with an
# energy-weighted position (~mid-crossing, not the entry face) and the track momentum; one hit
# per crossing = one PIXELAV deck line.
SIM.action.mapActions['ECalBarrel'] = 'Geant4TrackerWeightedAction'

# ---- output ----
_data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
_out_dir = os.path.join(_data_base, "trackermom")
os.makedirs(_out_dir, exist_ok=True)
SIM.outputFile = os.path.join(_out_dir, "trackermom_%s%g_1evt.root" % (_tag, _energy_gev))
