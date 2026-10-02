"""Extract the full shower cascade from the EDM4hep ROOT file into a compact .npz.

Reads the EDM4hep output of run_sim_fullcascade.py with uproot (no ROOT/PyROOT needed)
and writes a numpy archive the visualization notebook (05) loads with plain numpy.

Units: EDM4hep stores momentum / mass / hit energy in GeV, positions / vertices in mm.

Per-step hit truth: when run_sim_fullcascade.py sets enableDetailedShowerMode,
each ECalBarrelHitsContributions entry (one Geant4 step deposit) carries PDG, energy, stepLength,
stepPosition (mm) and a link to the producing MCParticle. Those are extracted here -- cbeg/cend
index the per-hit contribution ranges, and the flat contribution arrays are cpdg/cE/cslen/
csx/csy/csz/cmc -- and feed analysis/sensor_crossings.py. Without detailed mode the position/PDG
fields come out zero (only energy + the MCParticle link are written).

Usage: python extract_cascade.py [in.root] [out.npz] [event_index] [--home CONFIG_ROOT]

Defaults: in.root = $CALOMAPS_DATA_BASE/fullcascade/fullcascade_gamma50_1evt.root, out.npz =
$CALOMAPS_HOME/models/<in.root's name>.npz (this checkout's models/ without $CALOMAPS_HOME),
event_index = 0.

--home is the config root whose geometry/ was simulated (default: $CALOMAPS_HOME, else this
repo); sim/make_pixelav_inputs.sh passes it. The npz records that geometry (geo_si_thick_mm,
geo_si_centers_mm, geo_cell_size_mm, geo_home; see sensor_crossings.geometry_provenance), and
the extraction stops before reading the ROOT file if its Si mid-planes are not the ones the
crossing code assumes, or if the silicon thickness the input's name implies is not the --home
thickness (an _si<T>um tag names T; an untagged name means this checkout's own geometry).
"""
import argparse, os, re
import numpy as np
from sensor_crossings import REPO, geometry_provenance, check_thickness_label


def main(argv=None):
    data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
    home = os.environ.get("CALOMAPS_HOME", REPO)
    ap = argparse.ArgumentParser(description="Calorimeter-readout EDM4hep ROOT file -> npz of the "
                                             "MCParticle cascade, hits and per-step contributions.")
    ap.add_argument("in_root", nargs="?",
                    default=os.path.join(data_base, "fullcascade", "fullcascade_gamma50_1evt.root"),
                    help="ddsim output of sim/run_sim_fullcascade.py "
                         "(default: $CALOMAPS_DATA_BASE/fullcascade/fullcascade_gamma50_1evt.root)")
    ap.add_argument("out_npz", nargs="?",
                    help="output npz (default: $CALOMAPS_HOME/models/<in_root name>.npz)")
    ap.add_argument("event_index", nargs="?", type=int, default=0,
                    help="event to extract (default 0; cascade files are single-event)")
    ap.add_argument("--home", default=home, metavar="CONFIG_ROOT",
                    help="config root whose geometry/ was simulated (default: $CALOMAPS_HOME, else this repo)")
    args = ap.parse_intermixed_args(argv)
    IN, EV = args.in_root, args.event_index
    OUT = args.out_npz or os.path.join(home, "models", os.path.splitext(os.path.basename(IN))[0] + ".npz")
    GEO = geometry_provenance(args.home)   # before any ROOT I/O
    check_thickness_label(IN, GEO, args.home)

    # Stamp provenance honestly: parse particle/energy from the input filename written by the
    # steering files (<prefix>_<tag><E>_1evt.root); fall back to 'unknown' rather than guessing.
    _m = re.search(r"(?:trackermom|fullcascade)_([a-z]+?)([0-9.]+)_", os.path.basename(IN))
    _meta_particle = _m.group(1) if _m else "unknown"
    _meta_energy = (_m.group(2) + " GeV") if _m else "unknown"

    import uproot
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

    # --- per-step hit contributions (populated only with enableDetailedShowerMode) ---
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

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez_compressed(
        OUT,
        pdg=pdg, mass=mass, px=px, py=py, pz=pz, E=E,
        pex=pex, pey=pey, pez=pez,
        vsx=vsx, vsy=vsy, vsz=vsz, vex=vex, vey=vey, vez=vez,
        pid=pid, status=status, gstat=gstat,
        dbeg=dbeg, dend=dend, dau=dau,
        hx=hx, hy=hy, hz=hz, he=he, cellID=cellID,
        cbeg=cbeg, cend=cend, cpdg=cpdg, cE=cE, cslen=cslen, ctime=ctime, csx=csx, csy=csy, csz=csz, cmc=cmc,
        **GEO,
        meta=np.array([_meta_particle, _meta_energy, "+y pencil beam", "keepAllParticles",
                       "userParticleHandler=off", "enableDetailedShowerMode", "EDM4hep"]),
    )
    _detailed = bool(np.any((csx != 0) | (csy != 0) | (csz != 0)))
    print(f"saved {OUT}")
    print(f"  geometry: {GEO['geo_si_thick_mm'] * 1e3:g} um Si, config root {GEO['geo_home']}")
    print(f"  n_particles={len(pdg)}  n_hits={len(hx)}  n_contributions={len(cpdg)}")
    print(f"  particle E range (GeV): {E.min():.4f} .. {E.max():.2f}")
    print(f"  total hit energy deposited (GeV): {he.sum():.4f}")
    print(f"  contribution stepPosition populated: {_detailed}"
          + ("" if _detailed else "  <-- WARN: detailed mode NOT active; per-step crossings (variant A) unavailable"))
    print(f"  file size: {os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
