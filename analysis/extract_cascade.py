"""Extract the full shower cascade from the EDM4hep ROOT file into a compact .npz.

Reads the EDM4hep output of run_sim_fullcascade.py with uproot (no ROOT/PyROOT needed)
and writes a numpy archive the visualization notebook (04) loads with plain numpy.

Units: EDM4hep stores momentum / mass / hit energy in GeV, positions / vertices in mm.
The per-step hit truth (CaloHitContribution: PDG, energy, stepPosition, link to MCParticle)
is reachable via ECalBarrelHits.contributions_{begin,end}; it is left for experiment "B" /
the PIXELAV converter and is not extracted here.

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

t = uproot.open(IN)["events"]
br = ["MCParticles.PDG", "MCParticles.mass",
      "MCParticles.momentum.x", "MCParticles.momentum.y", "MCParticles.momentum.z",
      "MCParticles.momentumAtEndpoint.x", "MCParticles.momentumAtEndpoint.y", "MCParticles.momentumAtEndpoint.z",
      "MCParticles.vertex.x", "MCParticles.vertex.y", "MCParticles.vertex.z",
      "MCParticles.endpoint.x", "MCParticles.endpoint.y", "MCParticles.endpoint.z",
      "MCParticles.generatorStatus", "MCParticles.simulatorStatus",
      "ECalBarrelHits.position.x", "ECalBarrelHits.position.y", "ECalBarrelHits.position.z",
      "ECalBarrelHits.energy"]
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
E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)   # GeV
hx, hy, hz = g("ECalBarrelHits.position.x"), g("ECalBarrelHits.position.y"), g("ECalBarrelHits.position.z")
he = g("ECalBarrelHits.energy")   # GeV
pid = np.arange(len(pdg), dtype=np.int64)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(
    OUT,
    pdg=pdg, mass=mass, px=px, py=py, pz=pz, E=E,
    pex=pex, pey=pey, pez=pez,
    vsx=vsx, vsy=vsy, vsz=vsz, vex=vex, vey=vey, vez=vez,
    pid=pid, status=status, gstat=gstat,
    hx=hx, hy=hy, hz=hz, he=he,
    meta=np.array(["gamma", "50.0 GeV", "+Y pencil beam", "keepAllParticles",
                   "userParticleHandler=off", "EDM4hep"]),
)
print(f"saved {OUT}")
print(f"  n_particles={len(pdg)}  n_hits={len(hx)}")
print(f"  particle E range (GeV): {E.min():.4f} .. {E.max():.2f}")
print(f"  total hit energy deposited (GeV): {he.sum():.4f}")
print(f"  file size: {os.path.getsize(OUT)/1024:.0f} KB")
