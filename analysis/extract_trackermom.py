"""Extract per-crossing momentum truth from the tracker-readout sim (run_sim_trackermom.py).

run_sim_trackermom.py reads the ECal Si out as a Geant4 TRACKER, so ECalBarrelHits is a
collection of SimTrackerHits -- one combined hit per charged-track crossing of a Si sensor,
each carrying the TRUE Geant4 momentum[3] at that crossing (which the calorimeter
CaloHitContribution does NOT). This is the real per-step/per-crossing momentum that
sensor_crossings.py needs (and that notebook 06 validates).

This reads that ROOT with uproot and writes a compact .npz holding both the MCParticle cascade
(for context / production-momentum comparison) and the tracker hits.

Units: momentum in GeV, positions/pathLength in mm, time in ns, energy in GeV.

Usage: python extract_trackermom.py [in.root] [out.npz] [event_index] [--home CONFIG_ROOT]

Defaults: in.root = $CALOMAPS_DATA_BASE/trackermom/trackermom_gamma50_1evt.root, out.npz =
$CALOMAPS_HOME/models/<in.root's name>.npz (this checkout's models/ without $CALOMAPS_HOME),
event_index = 0.

--home is the config root whose geometry/ was simulated (default: $CALOMAPS_HOME, else this
repo); sim/make_pixelav_inputs.sh passes it. The npz records that geometry (geo_si_thick_mm,
geo_si_centers_mm, geo_cell_size_mm, geo_home; see sensor_crossings.geometry_provenance).
Nothing is written when the data and --home disagree:
  - before the ROOT file is read: the --home Si mid-planes must be the ones the crossing code
    assumes, and the silicon thickness the input's name implies must be the --home thickness
    (an _si<T>um tag names T; an untagged name means this checkout's own geometry);
  - after the hits are read: the path-length test of validate_crossings.py must pass for the
    --home thickness. When too few clean crossings exist to decide, the npz is written and the
    run says the data could not confirm the thickness.
"""
import argparse, os, re
import numpy as np
from sensor_crossings import REPO, geometry_provenance, check_thickness_label
from validate_crossings import MIN_SELECTED, PATH_TOL, path_length_test


def main(argv=None):
    data_base = os.environ.get("CALOMAPS_DATA_BASE", os.path.expanduser("~/CALOMAPS-data"))
    home = os.environ.get("CALOMAPS_HOME", REPO)
    ap = argparse.ArgumentParser(description="Tracker-readout EDM4hep ROOT file -> npz of the "
                                             "MCParticle cascade and the per-crossing tracker hits.")
    ap.add_argument("in_root", nargs="?",
                    default=os.path.join(data_base, "trackermom", "trackermom_gamma50_1evt.root"),
                    help="ddsim output of sim/run_sim_trackermom.py "
                         "(default: $CALOMAPS_DATA_BASE/trackermom/trackermom_gamma50_1evt.root)")
    ap.add_argument("out_npz", nargs="?",
                    help="output npz (default: $CALOMAPS_HOME/models/<in_root name>.npz)")
    ap.add_argument("event_index", nargs="?", type=int, default=0, help="event to extract (default 0)")
    ap.add_argument("--home", default=home, metavar="CONFIG_ROOT",
                    help="config root whose geometry/ was simulated (default: $CALOMAPS_HOME, else this repo)")
    args = ap.parse_intermixed_args(argv)
    IN, EV = args.in_root, args.event_index
    OUT = args.out_npz or os.path.join(home, "models", os.path.splitext(os.path.basename(IN))[0] + ".npz")
    GEO = geometry_provenance(args.home)   # before any ROOT I/O
    check_thickness_label(IN, GEO, args.home)
    thick_mm = float(GEO["geo_si_thick_mm"])

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

    b = path_length_test(dict(thx=thx, thy=thy, thz=thz, tpx=tpx, tpy=tpy, tpz=tpz, tpath=tpath,
                              tedep=tedep, tmc=tmc, pdg=pdg, vsx=vsx, vsy=vsy, vex=vex, vey=vey),
                         thick_mm)
    if b["ok"] is False:
        raise SystemExit(
            f"{IN}: the tracker-hit path lengths do not fit the {thick_mm * 1e3:g} um silicon of "
            f"--home {args.home}: median pathLength*|cos|/thickness = {b['ratio']:.4f} over "
            f"{b['n_sel']} clean crossings (allowed {1 - PATH_TOL:.2f}-{1 + PATH_TOL:.2f}), which "
            f"points to about {b['ratio'] * thick_mm * 1e3:.0f} um. --home is not the geometry this "
            f"file was simulated with; nothing written.")

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez_compressed(
        OUT,
        pdg=pdg, mass=mass, px=px, py=py, pz=pz, E=E, pex=pex, pey=pey, pez=pez,
        vsx=vsx, vsy=vsy, vsz=vsz, vex=vex, vey=vey, vez=vez,
        pid=pid, status=status, gstat=gstat, dbeg=dbeg, dend=dend, dau=dau,
        thx=thx, thy=thy, thz=thz, tpx=tpx, tpy=tpy, tpz=tpz,
        tpath=tpath, ttime=ttime, tedep=tedep, tcellID=tcellID, tmc=tmc,
        **GEO,
        meta=np.array([_meta_particle, _meta_energy, "+y pencil beam", "seed=424242",
                       "ECalBarrel as Geant4TrackerWeightedAction", "SimTrackerHit w/ momentum", "EDM4hep"]),
    )
    tp = np.sqrt(tpx**2 + tpy**2 + tpz**2)
    print(f"saved {OUT}")
    print(f"  geometry: {GEO['geo_si_thick_mm'] * 1e3:g} um Si, config root {GEO['geo_home']}")
    if b["ok"]:
        print(f"  thickness confirmed by {b['n_sel']} clean crossings: median "
              f"pathLength*|cos|/thickness = {b['ratio']:.4f}")
    else:
        print(f"  thickness NOT confirmed: {b['n_sel']} clean crossings, fewer than the {MIN_SELECTED} "
              f"the path-length test needs; the data could not confirm {thick_mm * 1e3:g} um silicon")
    print(f"  n_particles={len(pdg)}  n_tracker_hits={len(thx)}")
    print(f"  hit |p| (GeV): min={tp.min():.4f}  median={np.median(tp):.4f}  max={tp.max():.2f}" if len(thx) else "  no tracker hits in this event")
    print(f"  total eDep in Si (GeV): {tedep.sum():.4f}")
    print(f"  file size: {os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
