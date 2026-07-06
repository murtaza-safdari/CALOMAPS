"""Regenerate models/config_ab_gamma50.npz for notebook 04 section 2's A/B panel.

The panel shows the cascade-persistency settings change WHICH truth is written, not the physics.
It contrasts the stock ddsim config with the full-cascade config over 20 events each (same seed
424242) and shows their per-event silicon energy-deposit DISTRIBUTIONS coincide, while the stored
MCParticle count jumps from 1 to ~78,000. (A single-event hit comparison would be misleading: once
the settings change, the same seed drives a different random shower -- only the distributions align.)

Run on EAF under Key4hep 2026-04-08 with lib_hack on LD_LIBRARY_PATH (see docs/handbook.md 10.1):
    python analysis/make_config_ab.py

Reads the existing models/fullcascade_gamma50_1evt.npz for the full-cascade MCParticle count, so
run the main experiment-B pipeline (extract_cascade.py) first.
"""
import os, subprocess
import numpy as np
import uproot

HOME = os.environ.get("CALOMAPS_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
GEO = os.path.join(HOME, "geometry")
TMP = os.environ.get("TMPDIR", "/tmp")
def20 = os.path.join(TMP, "config_ab_def20.root")
casc20 = os.path.join(TMP, "config_ab_casc20.root")
common = ["ddsim", "--compactFile", "SiD_TestBeam.xml", "-N", "20"]

# stock config (default user particle handler, no keepAllParticles) and cascade config
# (userParticleHandler="" via run_sim_fullcascade.py; KEEP_ALL=0 keeps it fast -- we only need
# the deposit here, and the deposit distribution does not depend on the persistency depth).
subprocess.run(common + ["--steeringFile", "../sim/run_sim.py", "--random.seed", "424242",
                         "--outputFile", def20], cwd=GEO, check=True,
               env={**os.environ, "CALOMAPS_GUN_ENERGY_GEV": "50"})
subprocess.run(common + ["--steeringFile", "../sim/run_sim_fullcascade.py",
                         "--outputFile", casc20], cwd=GEO, check=True,
               env={**os.environ, "CALOMAPS_KEEP_ALL": "0"})


def deposits(root):
    e = uproot.open(root)["events"]["ECalBarrelHits.energy"].array()
    return np.array([float(np.asarray(x).sum()) for x in e])


def n_mcp_event0(root):
    return int(len(uproot.open(root)["events"]["MCParticles.PDG"].array(entry_stop=1)[0]))


depD, depC = deposits(def20), deposits(casc20)
n_default = n_mcp_event0(def20)                                          # stock: primary only (=1)
dF = np.load(os.path.join(HOME, "models", "fullcascade_gamma50_1evt.npz"), allow_pickle=True)
n_full = int(len(dF["pdg"]))                                            # full cascade (~78k)

out = os.path.join(HOME, "models", "config_ab_gamma50.npz")
np.savez(out, def_dep=depD, casc_dep=depC, n_mcp_default=n_default, n_mcp_full=n_full,
         n_hits_default=0, n_hits_full=int(len(dF["hx"])))
print(f"saved {out}")
print(f"  default deposit: {depD.mean():.3f} +/- {depD.std():.3f} GeV   cascade: {depC.mean():.3f} +/- {depC.std():.3f} GeV")
print(f"  MCParticles stored: default={n_default}  full cascade={n_full:,}")
