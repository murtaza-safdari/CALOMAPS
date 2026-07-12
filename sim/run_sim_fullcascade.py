"""DDSim steering for FULL-CASCADE truth output, EDM4hep format.

Goal: persist the 4-vector of *every* shower particle (the complete Geant4 cascade) into
the output, not just the primary + a sparse subset. Output is EDM4hep ROOT -- the Key4hep
community standard, uproot-readable, and the same format notebooks 01-03 consume.

Run (from geometry/, on EAF, Key4hep sourced):
    ddsim --compactFile SiD_TestBeam.xml \
          --steeringFile ../sim/run_sim_fullcascade.py \
          --numberOfEvents 1

Environment knobs (all optional; the defaults reproduce the canonical 50 GeV photon run):
    CALOMAPS_GUN_PARTICLE    gun particle (gamma, pi+, pi-, ...)      (default gamma)
    CALOMAPS_GUN_ENERGY_GEV  gun energy in GeV                        (default 50)
    CALOMAPS_RANGECUT_MM     Geant4 production range cut in mm        (default: DDSim's 0.7)
    CALOMAPS_KEEP_ALL        1 keeps every track; 0 prunes by min-KE  (default 1)
    CALOMAPS_MIN_KE_MEV      truth-persistency KE floor in MeV        (default 1.0)
    CALOMAPS_DATA_BASE       output directory root                    (default ~/CALOMAPS-data)
See docs/handbook.md "Controlling which secondaries are produced and saved" for the
difference between the production cut (a PHYSICS knob) and the persistency floor (OUTPUT only).

The one non-obvious setting is `part.userParticleHandler = ""`:
DDSim's default Geant4TCUserParticleHandler restricts MC-truth to the inner *tracking*
region, so ECal shower secondaries (born at r>1267mm) get merged into the primary and
never stored -- even with keepAllParticles=True. Disabling the user handler removes that
region cut, so the full cascade is retained (~78k particles for a single 50 GeV photon).

NOTE: keep this file pure ASCII. A non-ASCII character (em-dash, "smart quote") in a
steering file gets copied into the run metadata and breaks the EDM4hep RunHeader
std::map<string,string> conversion -- see docs/troubleshooting.md.

Read the output with uproot (see analysis/extract_cascade.py). The per-step hit truth the
per-sensor crossing records need lives in the ECalBarrelHits
'contributions' (CaloHitContribution: PDG, energy, stepPosition, stepLength, + a link to
MCParticle). Those per-step fields are populated ONLY because enableDetailedShowerMode is set
below; without it the contributions carry energy + the particle link but zero position.
"""
import os
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, MeV, mm, deg

SIM = DD4hepSimulation()

# Reproducibility: fix the seed so the shower is stable run-to-run AND identical to
# run_sim_trackermom.py (which pins the same seed), letting the calo and tracker readouts be
# compared on the SAME shower. Comment out for statistically independent showers.
SIM.random.seed = 424242

# ==========================================
# PARTICLE GUN -- single fixed-energy particle, +y pencil beam
# ==========================================
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

# ==========================================
# PHYSICS
# ==========================================
SIM.physicsList = "FTFP_BERT"

# ---- Geant4 production range cut (PHYSICS knob) --------------------------------------------
# The production cut decides which secondaries Geant4 physically CREATES as tracks: below this
# range, soft e-/e+/gamma are not made as separate particles -- their energy is deposited
# continuously along the parent's step instead. Lowering it produces more soft delta-rays
# (=> more per-sensor crossings) at higher CPU cost; raising it coarsens the
# shower. This CHANGES the simulated physics and the energy-deposition pattern. DDSim's default
# is 0.7*mm; we keep that default unless CALOMAPS_RANGECUT_MM is set.
#   Example: CALOMAPS_RANGECUT_MM=0.05 ddsim ...   (finer delta-ray production)
_rangecut_mm = os.environ.get("CALOMAPS_RANGECUT_MM", "")
if _rangecut_mm:
    SIM.physics.rangecut = float(_rangecut_mm) * mm

# ==========================================
# MC-TRUTH PERSISTENCY -- which particles get SAVED (output only, NOT physics)
# ==========================================
# userParticleHandler="" removes the tracking-region restriction so ECal secondaries persist
# (see the docstring above). This is load-bearing for the cascade and is not env-configurable.
SIM.part.userParticleHandler = ""
# keepAllParticles / minimalKineticEnergy are the TRUTH-PERSISTENCY knobs: they decide which
# already-simulated particles are written as MCParticles. They do NOT change the physics or the
# energy deposited -- only the size/content of the truth collection. IMPORTANT: the
# minimalKineticEnergy floor is IGNORED while keepAllParticles is True, so to actually prune by
# energy you must set CALOMAPS_KEEP_ALL=0 as well.
#   Example: CALOMAPS_KEEP_ALL=0 CALOMAPS_MIN_KE_MEV=10 ddsim ...  (prune most sub-10-MeV truth;
#   NOT a pure floor -- DD4hep still keeps any track that left a hit, plus primaries/parents)
SIM.part.keepAllParticles = os.environ.get("CALOMAPS_KEEP_ALL", "1") != "0"
SIM.part.minimalKineticEnergy = float(os.environ.get("CALOMAPS_MIN_KE_MEV", "1.0")) * MeV
SIM.part.printEndTracking = False
SIM.part.printStartTracking = False

# ==========================================
# CALORIMETER HIT DETAIL -- per-step truth for the sensor-crossing records
# ==========================================
# enableDetailedShowerMode puts the calorimeter sensitive actions in DETAILED_MODE
# (HitCreationMode=2), the only mode under which the EDM4hep writer fills each
# CaloHitContribution PDG, stepPosition and stepLength (SIMPLE_MODE leaves them zero).
# NOTE: SIM.part.enableDetailedHitsAndParticleInfo is a different knob and does NOT do this.
SIM.enableDetailedShowerMode = True

# ==========================================
# OUTPUT -- EDM4hep ROOT (.root + default => EDM4hep)
# ==========================================
_data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
_out_dir = os.path.join(_data_base, "fullcascade")
os.makedirs(_out_dir, exist_ok=True)
SIM.outputFile = os.path.join(_out_dir, "fullcascade_%s%g_1evt.root" % (_tag, _energy_gev))
