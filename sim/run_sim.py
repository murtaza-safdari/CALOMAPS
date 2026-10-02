from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, mm, deg
import os
import sys

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
#   CALOMAPS_SI_THICKNESS_UM sensor silicon thickness in um; only the sim/*.sh wrappers act on it,
#                           by passing a variant --compactFile. Here the default output name is
#                           tagged _si<T>um by the --compactFile's own thickness (untagged at the
#                           repo's 320 um); the knob only prints a warning when it disagrees.
#
# Examples:
#   CALOMAPS_GUN_PARTICLE=pi+ ddsim --compactFile SiD_TestBeam.xml --steeringFile run_sim.py -N 100 ...
#   CALOMAPS_GUN_PARTICLE=pi- CALOMAPS_GUN_ENERGY_GEV=50 ddsim ... -N 1 ...
# NOTE: ddsim exec()s this steering file with separate globals/locals, so a helper
# FUNCTION defined here cannot see the module-level `import os` (NameError at call time).
# Keep all env parsing inline at module level.
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
    # Mono-energetic beam: pin only the MOMENTUM to a degenerate window (|p| = _e). The
    # truth label downstream is E = sqrt(p^2 + m^2), so do NOT also set gun.energy: for a
    # massive particle E and |p| cannot both equal _e, and setting both makes the realized
    # beam depend on ddsim's gun apply-order. One convention (momentum) keeps it unambiguous.
    _e = float(ENERGY)
    SIM.gun.momentumMin = _e * GeV
    SIM.gun.momentumMax = _e * GeV
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
# ---- silicon thickness: the default output name is tagged by the --compactFile's geometry ----
# sim/steering_output.py reads the thickness of the compact on ddsim's command line: no tag at
# the repo's 320 um, _si<T>um otherwise, and a WARNING when CALOMAPS_SI_THICKNESS_UM disagrees.
# ddsim exec()s this file without __file__; the compiled code keeps the file's path.
_si_tag = ""
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(sys._getframe().f_code.co_filename)))
    from steering_output import si_output_tag
    _si_tag = si_output_tag()
except Exception as _err:
    print("WARNING: sim/steering_output.py could not be loaded from beside this steering file "
          "(%s): the silicon thickness of the --compactFile is unconfirmed and the default output "
          "name is untagged." % _err)

# Standalone default name (derived from the particle). The batch scripts
# generate_dataset.sh / generate_batched.sh override this with --outputFile.
_tag = {"gamma": "photons", "pi+": "piplus", "pi-": "piminus"}.get(
    PARTICLE, PARTICLE.replace("+", "plus").replace("-", "minus"))
SIM.outputFile = "sim_data_%s%s.root" % (_tag, _si_tag)
