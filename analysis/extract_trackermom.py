"""Extract per-crossing momentum truth from the tracker-readout sim (run_sim_trackermom.py).

run_sim_trackermom.py reads the ECal Si out as a Geant4 TRACKER, so ECalBarrelHits is a
collection of SimTrackerHits -- one combined hit per charged-track crossing of a Si sensor,
each carrying the TRUE Geant4 momentum[3] at that crossing (which the calorimeter
CaloHitContribution does NOT). This is the real per-step/per-crossing momentum that
pixelav_converter.py needs (and that nb05 validates).

This reads that ROOT with uproot and writes a compact .npz holding both the MCParticle cascade
(for context / production-momentum comparison) and the tracker hits.

Units: momentum in GeV, positions/pathLength in mm, time in ns, energy in GeV.

Usage: python extract_trackermom.py [in.root] [out.npz] [event_index]
"""
import os, sys
import numpy as np
import uproot

_data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
_home = os.environ.get("CALOMAPS_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
IN  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_data_base, "trackermom", "trackermom_gamma50_1evt.root")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_home, "models", "trackermom_gamma50_1evt.npz")
EV  = int(sys.argv[3]) if len(sys.argv) > 3 else 0

t = uproot.open(IN)["events"]
br = ["MCParticles.PDG", "MCParticles.mass",
      "MCParticles.momentum.x", "MCParticles.momentum.y", "MCParticles.momentum.z",
      "MCParticles.momentumAtEndpoint.x", "MCParticles.momentumAtEndpoint.y", "MCParticles.momentumAtEndpoint.z",
      "MCParticles.vertex.x", "MCParticles.vertex.y", "MCParticles.vertex.z",
      "MCParticles.endpoint.x", "MCParticles.endpoint.y", "MCParticles.endpoint.z",
      "MCParticles.generatorStatus", "MCParticles.simulatorStatus",
      "MCParticles.daughters_begin", "MCParticles.daughters_end", "_MCParticles_daughters.index",
      # SimTrackerHit ECalBarrelHits: one combined hit per Si crossing, WITH momentum
      "ECalBarrelHits.position.x", "ECalBarrelHits.position.y", "ECalBarrelHits.position.z",
      "ECalBarrelHits.momentum.x", "ECalBarrelHits.momentum.y", "ECalBarrelHits.momentum.z",
      "ECalBarrelHits.pathLength", "ECalBarrelHits.time", "ECalBarrelHits.eDep",
      "ECalBarrelHits.cellID", "_ECalBarrelHits_particle.index"]
a = t.arrays(br, entry_start=EV, entry_stop=EV + 1)
def g(name):
    return np.asarray(a[name][0])

pdg = g("MCParticles.PDG").astype(np.int32)
mass = g("MCParticles.mass")
px, py, pz = g("MCParticles.momentum.x"), g("MCParticles.momentum.y"), g("MCParticles.momentum.z")
pex, pey, pez = (g("MCParticles.momentumAtEndpoint.x"), g("MCParticles.momentumAtEndpoint.y"),
                 g("MCParticles.momentumAtEndpoint.z"))
vsx, vsy, vsz = g("MCParticles.vertex.x"), g("MCParticles.vertex.y"), g("MCParticles.vertex.z")
vex, vey, vez = g("MCParticles.endpoint.x"), g("MCParticles.endpoint.y"), g("MCParticles.endpoint.z")
gstat = g("MCParticles.generatorStatus").astype(np.int32)
status = g("MCParticles.simulatorStatus").astype(np.int32)
dbeg = g("MCParticles.daughters_begin").astype(np.int64)
dend = g("MCParticles.daughters_end").astype(np.int64)
dau = g("_MCParticles_daughters.index").astype(np.int64)
E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
pid = np.arange(len(pdg), dtype=np.int64)

# --- tracker hits: one per Si crossing, WITH momentum ---
thx, thy, thz = g("ECalBarrelHits.position.x"), g("ECalBarrelHits.position.y"), g("ECalBarrelHits.position.z")
tpx, tpy, tpz = g("ECalBarrelHits.momentum.x"), g("ECalBarrelHits.momentum.y"), g("ECalBarrelHits.momentum.z")
tpath = g("ECalBarrelHits.pathLength")        # mm in sensor
ttime = g("ECalBarrelHits.time")              # ns
tedep = g("ECalBarrelHits.eDep")              # GeV deposited
tcellID = g("ECalBarrelHits.cellID").astype(np.uint64)
tmc = g("_ECalBarrelHits_particle.index").astype(np.int64)   # -> MCParticles index

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(
    OUT,
    pdg=pdg, mass=mass, px=px, py=py, pz=pz, E=E, pex=pex, pey=pey, pez=pez,
    vsx=vsx, vsy=vsy, vsz=vsz, vex=vex, vey=vey, vez=vez,
    pid=pid, status=status, gstat=gstat, dbeg=dbeg, dend=dend, dau=dau,
    thx=thx, thy=thy, thz=thz, tpx=tpx, tpy=tpy, tpz=tpz,
    tpath=tpath, ttime=ttime, tedep=tedep, tcellID=tcellID, tmc=tmc,
    meta=np.array(["gamma", "50.0 GeV", "+y pencil beam", "seed=424242",
                   "ECalBarrel as Geant4TrackerWeightedAction", "SimTrackerHit w/ momentum", "EDM4hep"]),
)
tp = np.sqrt(tpx**2 + tpy**2 + tpz**2)
print(f"saved {OUT}")
print(f"  n_particles={len(pdg)}  n_tracker_hits={len(thx)}")
print(f"  hit |p| (GeV): min={tp.min():.4f}  median={np.median(tp):.4f}  max={tp.max():.2f}" if len(thx) else "  no tracker hits in this event")
print(f"  total eDep in Si (GeV): {tedep.sum():.4f}")
print(f"  file size: {os.path.getsize(OUT)/1024:.0f} KB")
