"""DDSim steering for FULL-CASCADE truth output (experiment "A"), EDM4hep format.

Goal: persist the 4-vector of *every* shower particle (the complete Geant4 cascade) into
the output, not just the primary + a sparse subset. Output is EDM4hep ROOT -- the Key4hep
community standard, uproot-readable, and the same format notebooks 01-03 consume.

Run (from geometry/, on EAF, Key4hep sourced):
    ddsim --compactFile SiD_TestBeam.xml \
          --steeringFile ../sim/run_sim_fullcascade.py \
          --numberOfEvents 1

The one non-obvious setting is `part.userParticleHandler = ""`:
DDSim's default Geant4TCUserParticleHandler restricts MC-truth to the inner *tracking*
region, so ECal shower secondaries (born at r>1267mm) get merged into the primary and
never stored -- even with keepAllParticles=True. Disabling the user handler removes that
region cut, so the full cascade is retained (~75k particles for a single 50 GeV photon).

NOTE: keep this file pure ASCII. A non-ASCII character (em-dash, "smart quote") in a
steering file gets copied into the run metadata and breaks the EDM4hep RunHeader
std::map<string,string> conversion -- see docs/troubleshooting.md.

Read the output with uproot (see analysis/extract_cascade.py). The per-step hit truth a
pixel-level simulation (PIXELAV / experiment "B") needs lives in the ECalBarrelHits
'contributions' (CaloHitContribution: PDG, energy, stepPosition, stepLength, + a link to
MCParticle). Those per-step fields are populated ONLY because enableDetailedShowerMode is set
below; without it the contributions carry energy + the particle link but zero position.
"""
import os
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, deg

SIM = DD4hepSimulation()

# ==========================================
# PARTICLE GUN -- single fixed-energy particle, +y pencil beam
# ==========================================
# CALOMAPS_GUN_PARTICLE (default gamma) + CALOMAPS_GUN_ENERGY_GEV (default 50; legacy alias
# CALOMAPS_GUN_GEV accepted) so the SAME steering makes a full cascade for any particle.
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

# ==========================================
# PHYSICS
# ==========================================
SIM.physicsList = "FTFP_BERT"

# ==========================================
# MC-TRUTH PARTICLE HANDLER -- keep the WHOLE cascade
# ==========================================
SIM.part.userParticleHandler = ""      # remove the tracking-region restriction (see above)
SIM.part.keepAllParticles = True       # keep every Geant4 track as an MCParticle
SIM.part.minimalKineticEnergy = 1.0    # MeV; sane floor for when keepAllParticles=False
SIM.part.printEndTracking = False
SIM.part.printStartTracking = False

# ==========================================
# CALORIMETER HIT DETAIL -- per-step truth for PIXELAV (experiment "B")
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
SIM.outputFile = os.path.join(_out_dir, "fullcascade_%s%g_1evt.root" % (_tag, _gev))
