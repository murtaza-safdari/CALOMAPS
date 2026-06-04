"""Extract the full shower cascade from the DD4hep-native ROOT file into a .npz.

Reads /tmp/fc_dd4hep.root (DD4hep Geant4Output2ROOT format; needs libDDG4IO for the
dd4hep::sim::Geant4Particle / Geant4Calorimeter::Hit dictionaries) and writes a compact
numpy archive the visualization notebook can load with plain numpy (no PyROOT needed).

Units: momenta/energy/mass in GeV, positions/vertices in mm.
"""
import os, sys, ROOT, numpy as np

# The DD4hep-native ROOT file stores dd4hep::sim::Geant4Particle / Geant4Calorimeter::Hit
# objects via custom streamers; libDDG4IO provides their dictionaries. Load returns non-zero
# on failure WITHOUT raising — without the dictionary, the branch reads below segfault or
# raise obscure cppyy errors, so fail loudly here.
if ROOT.gSystem.Load("libDDG4IO") < 0:
    raise RuntimeError("Failed to load libDDG4IO (is the Key4hep environment sourced?)")

# Input: the DD4hep-native ROOT file from run_sim_fullcascade.py.
# Output: a compact .npz under models/. Both overridable via argv.
_data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
_home = os.environ.get("CALOMAPS_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
IN  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_data_base, "fullcascade", "fullcascade_gamma50_1evt.root")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_home, "models", "fullcascade_gamma50_1evt.npz")

f = ROOT.TFile.Open(IN)
if f is None or f.IsZombie():
    raise FileNotFoundError(f"Cannot open ROOT file: {IN}")
t = f.Get("EVENT")
if t is None:
    raise KeyError(f"'EVENT' tree not found in {IN} (is this a DD4hep-native ROOT file?)")
if t.GetEntry(0) <= 0:
    raise ValueError(f"event 0 not readable / empty in {IN}")

mc = t.MCParticles
n = mc.size()
pdg    = np.empty(n, np.int32)
mass   = np.empty(n, np.float64)
px = np.empty(n); py = np.empty(n); pz = np.empty(n)
vsx = np.empty(n); vsy = np.empty(n); vsz = np.empty(n)   # production vertex (mm)
vex = np.empty(n); vey = np.empty(n); vez = np.empty(n)   # end vertex (mm)
pid    = np.empty(n, np.int64)
parent = np.empty(n, np.int64)
status = np.empty(n, np.int32)
gstat  = np.empty(n, np.int32)

for i in range(n):
    p = mc[i]
    pdg[i] = p.pdgID
    mass[i] = p.mass / 1000.0
    px[i] = p.psx / 1000.0; py[i] = p.psy / 1000.0; pz[i] = p.psz / 1000.0
    vsx[i] = p.vsx; vsy[i] = p.vsy; vsz[i] = p.vsz
    vex[i] = p.vex; vey[i] = p.vey; vez[i] = p.vez
    pid[i] = p.id; parent[i] = p.g4Parent
    status[i] = p.status; gstat[i] = p.genStatus

E = np.sqrt(px*px + py*py + pz*pz + mass*mass)   # GeV

# Hits (energy deposits in silicon) — position (mm) + deposited energy (GeV)
hits = t.ECalBarrelHits
nh = hits.size()
hx = np.empty(nh); hy = np.empty(nh); hz = np.empty(nh); he = np.empty(nh)
for j in range(nh):
    h = hits[j]
    pos = h.position
    hx[j] = pos.X(); hy[j] = pos.Y(); hz[j] = pos.Z()
    he[j] = h.energyDeposit / 1000.0

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(
    OUT,
    pdg=pdg, mass=mass, px=px, py=py, pz=pz, E=E,
    vsx=vsx, vsy=vsy, vsz=vsz, vex=vex, vey=vey, vez=vez,
    pid=pid, parent=parent, status=status, gstat=gstat,
    hx=hx, hy=hy, hz=hz, he=he,
    meta=np.array(["gamma", "50.0 GeV", "+Y pencil beam", "keepAllParticles", "userParticleHandler=off"]),
)
print(f"saved {OUT}")
print(f"  n_particles={n}  n_hits={nh}")
print(f"  particle E range (GeV): {E.min():.4f} .. {E.max():.2f}")
print(f"  total hit energy deposited (GeV): {he.sum():.4f}")
print(f"  file size: {os.path.getsize(OUT)/1024:.0f} KB")
