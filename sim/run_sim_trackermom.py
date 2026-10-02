"""DDSim steering for PER-CROSSING MOMENTUM truth, EDM4hep format.

The standard calorimeter readout (run_sim_fullcascade.py) writes CaloHitContributions that carry
step position/PDG/time but NO momentum (EDM4hep CaloHitContribution has no momentum field). To
get the real per-sensor-crossing momentum the crossing records want, this steering maps the ECalBarrel Si to
Geant4's TRACKER sensitive action instead of the calorimeter action:

    SIM.action.mapActions['ECalBarrel'] = 'Geant4TrackerWeightedAction'

A tracker SD produces SimTrackerHits, which DO carry momentum[3], position[3], pathLength, EDep,
time and a link to the producing MCParticle -- i.e. one record per charged-track crossing of a Si
sensor, with the true Geant4 momentum at that crossing. (The hit position is the midpoint of
the hit's first and last step points, which for a straight through-going track is where it
crosses the sensor mid-plane -- not the entry-face point; see notebooks 06/07.) Same gun, physics and seed (CALOMAPS_SEED, default
424242) as run_sim_fullcascade.py, so the shower is identical; only the Si readout differs.

This is a SEPARATE run from the calorimeter one: notebook 05 (shower display) and notebook 07's
calo route still use the calorimeter output; the crossing records (sensor_crossings.py) use THIS
tracker output for real per-crossing momentum + impact + direction.

Run (from geometry/, on EAF, Key4hep + lib_hack on LD_LIBRARY_PATH):
    ddsim --compactFile SiD_TestBeam.xml --steeringFile ../sim/run_sim_trackermom.py --numberOfEvents 1

Environment knobs (all optional; the defaults reproduce the canonical 50 GeV photon run):
    CALOMAPS_GUN_PARTICLE    gun particle (gamma, pi+, pi-, ...)      (default gamma)
    CALOMAPS_GUN_ENERGY_GEV  gun energy in GeV                        (default 50)
    CALOMAPS_RANGECUT_MM     Geant4 production range cut in mm        (default: DDSim's 0.7)
    CALOMAPS_KEEP_ALL        1 keeps every track; 0 prunes by min-KE  (default 1)
    CALOMAPS_MIN_KE_MEV      truth-persistency KE floor in MeV        (default 1.0)
    CALOMAPS_DATA_BASE       output directory root                    (default ~/CALOMAPS-data)
    CALOMAPS_SEED            random seed, an integer; run_sim_fullcascade.py reads the same knob
                             (default 424242). A --random.seed on the ddsim command line overrides
                             it. The default output name is tagged _seed<R> by the seed of the run
                             (untagged at the default seed)
    CALOMAPS_SI_THICKNESS_UM sensor silicon thickness in um; only the sim/*.sh wrappers act on
                             it, by passing a variant --compactFile. Here the default output name
                             is tagged _si<T>um by the --compactFile's own thickness (untagged at
                             the repo's 320 um); the knob only prints a warning when it disagrees.
    CALOMAPS_SI_CUT_UM       production cut in the sensor silicon in um; like the thickness, only
                             the sim/*.sh wrappers act on it, and the default output name is
                             tagged _cut<C>um by the --compactFile's own silicon region
    CALOMAPS_SI_STEP_UM      maximum Geant4 step length in the sensor silicon in um; likewise
                             acted on by the wrappers only, and tagged _stepmax<S>um by the
                             --compactFile's own silicon limitset
    CALOMAPS_TRACKER_HITS    crossing: one hit per sensor crossing (default); step: one hit per
                             Geant4 step in the silicon (see "tracker hits" below)
See docs/handbook.md "Controlling which secondaries are produced and saved" for the physics
(production cut) vs. output-only (persistency floor) distinction.

The default output name is
trackermom_<tag><E>_<N>evt[_seed<R>][_step][_si<T>um][_cut<C>um][_stepmax<S>um].root, N being the
--numberOfEvents of the ddsim command line and R the random seed when it is not 424242;
--outputFile replaces it.

Keep this file pure ASCII (non-ASCII breaks the EDM4hep RunHeader std::map conversion).
"""
import os
import sys
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, MeV, mm, deg

SIM = DD4hepSimulation()

# Reproducibility: same seed as the calorimeter cascade so the shower is identical.
SIM.random.seed = int(os.environ.get("CALOMAPS_SEED", "424242"))

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
# along the parent step instead). Lower => more soft delta-rays => more sensor crossings,
# at higher CPU cost; higher => a coarser shower. This changes the simulated physics.
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
# Geant4TrackerWeightedAction combines the steps of one sensor crossing into a single hit whose
# position is the midpoint of its first and last step points (the mid-plane point of a straight
# through-going track, not the entry face) and whose momentum is the mean of the momenta there;
# one hit per crossing = one crossing record.
# ---- tracker hits: per crossing (default) or per Geant4 step (CALOMAPS_TRACKER_HITS=step) ----
# CollectSingleDeposits makes every step its own hit: the midpoint of the step's pre- and
# post-step points, the mean of their momenta, the step length as pathLength and the step's
# deposit as EDep. In both modes the ECal keeps every step with a deposit: ECalBarrel is a
# calorimeter-type sensitive detector, so ddsim gives it the calorimeter filter (filter.calo,
# edep0), not the tracker one (filter.tracker, edep1kev: 1 keV per step); step mode also names
# edep0 for it (filter.mapDetFilter). The shower itself is unchanged; the run header records
# both settings (action.mapActions, filter.mapDetFilter), and analysis/extract_trackermom.py
# reads the hit mode from there.
_hits = os.environ.get("CALOMAPS_TRACKER_HITS", "").strip() or "crossing"
if _hits not in ("crossing", "step"):
    raise SystemExit("CALOMAPS_TRACKER_HITS=%r: expected crossing or step" % _hits)
if _hits == "step":
    SIM.action.mapActions['ECalBarrel'] = ('Geant4TrackerWeightedAction',
                                           {'HitPositionCombination': 2, 'CollectSingleDeposits': True})
    SIM.filter.mapDetFilter['ECalBarrel'] = 'edep0'
else:
    SIM.action.mapActions['ECalBarrel'] = 'Geant4TrackerWeightedAction'

# ---- default output name: event count, seed, hit mode and silicon from the ddsim command line ----
# sim/steering_output.py reads ddsim's command line: _<N>evt from --numberOfEvents, _seed<R> from
# --random.seed, else from SIM.random.seed (no tag at the default 424242), and the thickness,
# silicon production cut and silicon step limit of the --compactFile: no tag for the repo
# geometry, _si<T>um for another thickness, _cut<C>um for a cut of its own and _stepmax<S>um for a
# step limit of its own, with a WARNING when CALOMAPS_SI_THICKNESS_UM, CALOMAPS_SI_CUT_UM or
# CALOMAPS_SI_STEP_UM disagrees. Step hits add _step.
# ddsim exec()s this file without __file__; the compiled code keeps the file's path.
_si_tag, _n_tag, _seed_tag = "", "", ""
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(sys._getframe().f_code.co_filename)))
    from steering_output import events_tag, seed_tag, si_output_tag
    _n_tag = events_tag(fallback=SIM.numberOfEvents)
    _seed_tag = seed_tag(fallback=SIM.random.seed)
    _si_tag = si_output_tag()
except Exception as _err:
    print("WARNING: sim/steering_output.py could not be loaded from beside this steering file "
          "(%s): the silicon thickness of the --compactFile is unconfirmed and the default output "
          "name carries neither the event count, the seed nor a thickness tag." % _err)

# ---- output ----
_data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
_out_dir = os.path.join(_data_base, "trackermom")
os.makedirs(_out_dir, exist_ok=True)
_hits_tag = "_step" if _hits == "step" else ""
SIM.outputFile = os.path.join(_out_dir, "trackermom_%s%g%s%s%s%s.root"
                              % (_tag, _energy_gev, _n_tag, _seed_tag, _hits_tag, _si_tag))
