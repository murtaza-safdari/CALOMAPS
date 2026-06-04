"""DDSim steering for FULL-CASCADE truth output (experiment "A").

Goal: persist the 4-vector of *every* shower particle (the complete Geant4
cascade) into the output, not just the primary + a sparse subset. This truth-level
particle list is the eventual input to a high-fidelity Si pixel simulation (PIXELAV).

Run (from geometry/, on EAF, Key4hep sourced):
    ddsim --compactFile SiD_TestBeam.xml \
          --steeringFile ../sim/run_sim_fullcascade.py \
          --numberOfEvents 1

Two non-obvious settings make this work on key4hep 2026-02-01 / DD4hep 1.35:

1. part.userParticleHandler = ""   (THE key to getting the cascade)
   DDSim's default Geant4TCUserParticleHandler restricts MC-truth to the inner
   *tracking* region. ECal shower secondaries are born at r>1267mm, OUTSIDE that
   region, so they get merged into the primary and never persisted — even with
   keepAllParticles=True. Disabling the user handler removes the region cut, so
   the full cascade is retained (~75k particles for a 50 GeV photon).

2. outputConfig.forceDD4HEP = True   (EDM4hep-output workaround)
   This build's EDM4hep output path (DDSim OutputConfig._configureEDM4HEP) assigns
   Python dicts to map-typed writer properties (RunHeader, EventParameters*,
   RunParameters*). cppyy cannot convert dict -> std::map<string,string> here
   ("...map<string,string>... is not defined"); the EventParameters assignment
   aborts the process with a silent C++ exit(0) BEFORE any event runs (no traceback,
   no file). The DD4hep *native* ROOT path (setupROOTOutput) bypasses all of that.
   Output is a TTree "EVENT" with branches MCParticles (vector<Geant4Particle*>) and
   ECalBarrelHits; read it with PyROOT after `gSystem.Load("libDDG4IO")`
   (see analysis/extract_cascade.py). See docs/troubleshooting.md.

For this first look we fire a single fixed-energy photon so the cascade is
reproducible and a manageable size to inspect.
"""
import os
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, deg

SIM = DD4hepSimulation()

# ==========================================
# PARTICLE GUN — single fixed-energy photon, +Y pencil beam
# ==========================================
SIM.enableGun = True
SIM.gun.particle = "gamma"
SIM.gun.energy = 50.0 * GeV
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
# MC-TRUTH PARTICLE HANDLER — keep the WHOLE cascade
# ==========================================
SIM.part.userParticleHandler = ""      # (1) above — remove tracking-region cut
SIM.part.keepAllParticles = True       # keep every Geant4 track as an MCParticle
SIM.part.minimalKineticEnergy = 1.0    # MeV; sane floor for when keepAllParticles=False
SIM.part.printEndTracking = False
SIM.part.printStartTracking = False

# ==========================================
# OUTPUT — DD4hep native ROOT (see (2) above)
# ==========================================
SIM.outputConfig.forceDD4HEP = True
_data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
_out_dir = os.path.join(_data_base, "fullcascade")
os.makedirs(_out_dir, exist_ok=True)
SIM.outputFile = os.path.join(_out_dir, "fullcascade_gamma50_1evt.root")
