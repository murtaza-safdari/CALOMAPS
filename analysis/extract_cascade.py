"""Extract the full shower cascade from the EDM4hep ROOT file into a compact .npz.

Reads the EDM4hep output of run_sim_fullcascade.py with uproot (no ROOT/PyROOT needed)
and writes a numpy archive the visualization notebook (04) loads with plain numpy.

Units: EDM4hep stores momentum / mass / hit energy in GeV, positions / vertices in mm.

Per-step hit truth (experiment "B"): when run_sim_fullcascade.py sets enableDetailedShowerMode,
each ECalBarrelHitsContributions entry (one Geant4 step deposit) carries PDG, energy, stepLength,
stepPosition (mm) and a link to the producing MCParticle. Those are extracted here -- cbeg/cend
index the per-hit contribution ranges, and the flat contribution arrays are cpdg/cE/cslen/
csx/csy/csz/cmc -- and feed analysis/pixelav_converter.py. Without detailed mode the position/PDG
fields come out zero (only energy + the MCParticle link are written).

Usage: python extract_cascade.py [in.root] [out.npz] [event_index]
"""
import os, sys
import numpy as np
import uproot

_data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
_home = os.environ.get("CALOMAPS_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
IN  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_data_base, "fullcascade", "fullcascade_gamma50_1evt.root")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_home, "models", "fullcascade_gamma50_1evt.npz")
EV  = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # cascade files are single-event


# Stamp provenance honestly: parse particle/energy from the input filename written by the
# steering files (<prefix>_<tag><E>_1evt.root); fall back to 'unknown' rather than guessing.
import re as _re
_m = _re.search(r"(?:trackermom|fullcascade)_([a-z]+?)([0-9.]+)_", os.path.basename(IN))
_meta_particle = _m.group(1) if _m else "unknown"
_meta_energy = (_m.group(2) + " GeV") if _m else "unknown"

t = uproot.open(IN)["events"]
br = ["MCParticles.PDG", "MCParticles.mass",
      "MCParticles.momentum.x", "MCParticles.momentum.y", "MCParticles.momentum.z",
      "MCParticles.momentumAtEndpoint.x", "MCParticles.momentumAtEndpoint.y", "MCParticles.momentumAtEndpoint.z",
      "MCParticles.vertex.x", "MCParticles.vertex.y", "MCParticles.vertex.z",
      "MCParticles.endpoint.x", "MCParticles.endpoint.y", "MCParticles.endpoint.z",
      "MCParticles.generatorStatus", "MCParticles.simulatorStatus",
      "MCParticles.daughters_begin", "MCParticles.daughters_end", "_MCParticles_daughters.index",
      "ECalBarrelHits.position.x", "ECalBarrelHits.position.y", "ECalBarrelHits.position.z",
      "ECalBarrelHits.energy", "ECalBarrelHits.cellID",
      "ECalBarrelHits.contributions_begin", "ECalBarrelHits.contributions_end",
      "ECalBarrelHitsContributions.PDG", "ECalBarrelHitsContributions.energy",
      "ECalBarrelHitsContributions.stepLength", "ECalBarrelHitsContributions.time",
      "ECalBarrelHitsContributions.stepPosition.x", "ECalBarrelHitsContributions.stepPosition.y",
      "ECalBarrelHitsContributions.stepPosition.z",
      "_ECalBarrelHitsContributions_particle.index"]
a = t.arrays(br, entry_start=EV, entry_stop=EV + 1)
def g(name):
    return np.asarray(a[name][0])

pdg = g("MCParticles.PDG").astype(np.int32)
mass = g("MCParticles.mass")
px, py, pz = g("MCParticles.momentum.x"), g("MCParticles.momentum.y"), g("MCParticles.momentum.z")
pex, pey, pez = (g("MCParticles.momentumAtEndpoint.x"), g("MCParticles.momentumAtEndpoint.y"),
                 g("MCParticles.momentumAtEndpoint.z"))   # momentum at end of track (GeV)
vsx, vsy, vsz = g("MCParticles.vertex.x"), g("MCParticles.vertex.y"), g("MCParticles.vertex.z")
vex, vey, vez = g("MCParticles.endpoint.x"), g("MCParticles.endpoint.y"), g("MCParticles.endpoint.z")
gstat = g("MCParticles.generatorStatus").astype(np.int32)
status = g("MCParticles.simulatorStatus").astype(np.int32)
dbeg = g("MCParticles.daughters_begin").astype(np.int64)
dend = g("MCParticles.daughters_end").astype(np.int64)
dau = g("_MCParticles_daughters.index").astype(np.int64)   # flat list; daughters of particle i = dau[dbeg[i]:dend[i]]
E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)   # GeV
hx, hy, hz = g("ECalBarrelHits.position.x"), g("ECalBarrelHits.position.y"), g("ECalBarrelHits.position.z")
he = g("ECalBarrelHits.energy")   # GeV
pid = np.arange(len(pdg), dtype=np.int64)

# --- per-step hit contributions (experiment "B"; populated only with enableDetailedShowerMode) ---
cellID = g("ECalBarrelHits.cellID").astype(np.uint64)
cbeg = g("ECalBarrelHits.contributions_begin").astype(np.int64)
cend = g("ECalBarrelHits.contributions_end").astype(np.int64)
cpdg = g("ECalBarrelHitsContributions.PDG").astype(np.int32)
cE = g("ECalBarrelHitsContributions.energy")                       # GeV deposited in the step
cslen = g("ECalBarrelHitsContributions.stepLength")               # mm
ctime = g("ECalBarrelHitsContributions.time")                  # ns; orders steps within a crossing
csx = g("ECalBarrelHitsContributions.stepPosition.x")            # mm, global
csy = g("ECalBarrelHitsContributions.stepPosition.y")
csz = g("ECalBarrelHitsContributions.stepPosition.z")
cmc = g("_ECalBarrelHitsContributions_particle.index").astype(np.int64)   # -> index into MCParticles

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(
    OUT,
    pdg=pdg, mass=mass, px=px, py=py, pz=pz, E=E,
    pex=pex, pey=pey, pez=pez,
    vsx=vsx, vsy=vsy, vsz=vsz, vex=vex, vey=vey, vez=vez,
    pid=pid, status=status, gstat=gstat,
    dbeg=dbeg, dend=dend, dau=dau,
    hx=hx, hy=hy, hz=hz, he=he, cellID=cellID,
    cbeg=cbeg, cend=cend, cpdg=cpdg, cE=cE, cslen=cslen, ctime=ctime, csx=csx, csy=csy, csz=csz, cmc=cmc,
    meta=np.array([_meta_particle, _meta_energy, "+y pencil beam", "keepAllParticles",
                   "userParticleHandler=off", "enableDetailedShowerMode", "EDM4hep"]),
)
_detailed = bool(np.any((csx != 0) | (csy != 0) | (csz != 0)))
print(f"saved {OUT}")
print(f"  n_particles={len(pdg)}  n_hits={len(hx)}  n_contributions={len(cpdg)}")
print(f"  particle E range (GeV): {E.min():.4f} .. {E.max():.2f}")
print(f"  total hit energy deposited (GeV): {he.sum():.4f}")
print(f"  contribution stepPosition populated: {_detailed}"
      + ("" if _detailed else "  <-- WARN: detailed mode NOT active; experiment-B (Variant A) unavailable"))
print(f"  file size: {os.path.getsize(OUT)/1024:.0f} KB")
