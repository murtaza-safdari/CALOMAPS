"""Extract per-crossing momentum truth from the tracker-readout sim (run_sim_trackermom.py).

run_sim_trackermom.py reads the ECal Si out as a Geant4 TRACKER, so ECalBarrelHits is a
collection of SimTrackerHits -- one combined hit per charged-track crossing of a Si sensor,
each carrying the TRUE Geant4 momentum[3] at that crossing (which the calorimeter
CaloHitContribution does NOT). This is the real per-step/per-crossing momentum that
sensor_crossings.py needs (and that notebook 06 validates).

This reads that ROOT with uproot and writes a compact .npz holding both the MCParticle cascade
(for context / production-momentum comparison) and the tracker hits.

Units: momentum in GeV, positions/pathLength in mm, time in ns, energy in GeV.

Usage: python extract_trackermom.py [in.root] [out.npz] [event_index] [--events A:B]
                                    [--particles all|hit] [--home CONFIG_ROOT]

Defaults: in.root = $CALOMAPS_DATA_BASE/trackermom/trackermom_gamma50_1evt.root, out.npz =
$CALOMAPS_HOME/models/<in.root's name>.npz (this checkout's models/ without $CALOMAPS_HOME).

EVENTS: every event of the file, read in chunks of CHUNK_EVENTS. event_index extracts that one
entry; --events A:B the entries A..B-1 (A defaults to 0, B to the end). The events' particles and
hits are concatenated into the same flat arrays as one event's, with the links offset so they stay
valid npz indices: tmc (hit -> particle) and dau (daughter -> particle) by the particles of the
preceding events, dbeg/dend by their daughter links (negative values are left unchanged).
  evt_id_particle  int32, per MCParticle: the event's entry number in the ROOT file
  evt_id_hit       int32, per tracker hit: the same
  num_events       number of events extracted
  pid              the particle's index in its own event's MCParticle collection

PARTICLES (--particles): all (default) keeps every MCParticle; hit keeps only those with at least
one tracker hit (about 1 in 40 in a 50 GeV photon shower), tmc pointing into the kept ones. A hit
npz has no dbeg/dend/dau: the daughters of a kept particle are mostly not kept, so a reader of
the decay tree fails on the missing key. The mode is stored as particles_kept.

PROVENANCE: the sim_* keys (seed, events simulated, gun particle and energy, DD4hep and Geant4
versions, date) and meta come from the file's ddsim run header (sensor_crossings.sim_provenance);
without one, particle and energy come from the file name and the seed is "unknown".

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
import argparse, math, os
import numpy as np
from sensor_crossings import REPO, geometry_provenance, check_thickness_label, sim_provenance
from validate_crossings import MIN_SELECTED, PATH_TOL, path_length_test

CHUNK_EVENTS = 10   # events per read: a 50 GeV photon event holds ~78k MCParticles

_PARTICLE_BRANCHES = {   # npz key -> MCParticles branch
    "pdg": "MCParticles.PDG", "mass": "MCParticles.mass",
    "px": "MCParticles.momentum.x", "py": "MCParticles.momentum.y", "pz": "MCParticles.momentum.z",
    "pex": "MCParticles.momentumAtEndpoint.x", "pey": "MCParticles.momentumAtEndpoint.y",
    "pez": "MCParticles.momentumAtEndpoint.z",
    "vsx": "MCParticles.vertex.x", "vsy": "MCParticles.vertex.y", "vsz": "MCParticles.vertex.z",
    "vex": "MCParticles.endpoint.x", "vey": "MCParticles.endpoint.y", "vez": "MCParticles.endpoint.z",
    "gstat": "MCParticles.generatorStatus", "status": "MCParticles.simulatorStatus",
    "dbeg": "MCParticles.daughters_begin", "dend": "MCParticles.daughters_end",
}
_HIT_BRANCHES = {        # SimTrackerHit ECalBarrelHits: one combined hit per Si crossing, WITH momentum
    "thx": "ECalBarrelHits.position.x", "thy": "ECalBarrelHits.position.y",
    "thz": "ECalBarrelHits.position.z",
    "tpx": "ECalBarrelHits.momentum.x", "tpy": "ECalBarrelHits.momentum.y",
    "tpz": "ECalBarrelHits.momentum.z",
    "tpath": "ECalBarrelHits.pathLength",   # mm in sensor
    "ttime": "ECalBarrelHits.time",         # ns
    "tedep": "ECalBarrelHits.eDep",         # GeV deposited
    "tcellID": "ECalBarrelHits.cellID",
    "tmc": "_ECalBarrelHits_particle.index",   # -> MCParticles index
}
_DAUGHTER_BRANCH = "_MCParticles_daughters.index"
_DTYPES = {"pdg": np.int32, "gstat": np.int32, "status": np.int32, "dbeg": np.int64,
           "dend": np.int64, "dau": np.int64, "tcellID": np.uint64, "tmc": np.int64}


def _event_range(spec, index, n_file):
    """[start, stop) of the entries to extract from a file of n_file entries."""
    if index is not None and spec is not None:
        raise SystemExit("give event_index or --events, not both")
    if index is not None:
        start, stop = index, index + 1
    elif spec is not None:
        a, sep, b = spec.partition(":")
        try:
            if not sep:
                raise ValueError
            start = int(a) if a.strip() else 0
            stop = int(b) if b.strip() else n_file
        except ValueError:
            raise SystemExit(f"--events {spec!r}: expected A:B, entries A..B-1") from None
    else:
        start, stop = 0, n_file
    if not 0 <= start < stop <= n_file:
        raise SystemExit(f"events {start}..{stop - 1} are not in the file, which holds entries "
                         f"0..{n_file - 1}")
    return start, stop


def _shift(x, offset):
    """x + offset where x >= 0 (an index), x where negative (a sentinel)."""
    return np.where(x >= 0, x + offset, x)


def _read_chunk(a, first_entry, keep_hit_only, n_part_before, n_dau_before):
    """Flat arrays of one uproot chunk `a` (events first_entry...), links offset by the particles
    and daughter links already extracted."""
    import awkward as ak
    def flat(branch):
        return ak.to_numpy(ak.flatten(a[branch], axis=1))
    n_part = ak.to_numpy(ak.num(a[_PARTICLE_BRANCHES["pdg"]], axis=1)).astype(np.int64)
    n_hit = ak.to_numpy(ak.num(a[_HIT_BRANCHES["tmc"]], axis=1)).astype(np.int64)
    evt = np.arange(first_entry, first_entry + len(n_part), dtype=np.int32)
    first_part = np.cumsum(n_part) - n_part          # each event's first particle in the chunk
    evt_of_part = np.repeat(np.arange(len(n_part)), n_part)

    out = {k: flat(b) for k, b in _HIT_BRANCHES.items()}
    tmc = out["tmc"].astype(np.int64)
    in_chunk = _shift(tmc, np.repeat(first_part, n_hit))   # hit -> particle index within the chunk
    keep = np.unique(in_chunk[tmc >= 0]) if keep_hit_only else np.arange(int(n_part.sum()))
    out["tmc"] = np.where(tmc >= 0, np.searchsorted(keep, in_chunk) + n_part_before, tmc)
    out["evt_id_hit"] = np.repeat(evt, n_hit)

    for k, b in _PARTICLE_BRANCHES.items():
        if keep_hit_only and k in ("dbeg", "dend"):
            continue
        out[k] = flat(b)[keep]
    if not keep_hit_only:
        n_dau = ak.to_numpy(ak.num(a[_DAUGHTER_BRANCH], axis=1)).astype(np.int64)
        part_shift = np.repeat(n_dau_before + np.cumsum(n_dau) - n_dau, n_part)
        out["dbeg"] = _shift(out["dbeg"].astype(np.int64), part_shift)
        out["dend"] = _shift(out["dend"].astype(np.int64), part_shift)
        out["dau"] = _shift(flat(_DAUGHTER_BRANCH).astype(np.int64),
                            np.repeat(n_part_before + first_part, n_dau))
    out["pid"] = (keep - first_part[evt_of_part[keep]]).astype(np.int64)
    out["evt_id_particle"] = evt[evt_of_part[keep]]
    return out, n_hit


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
    ap.add_argument("event_index", nargs="?", type=int,
                    help="extract only this entry of the file (default: every event)")
    ap.add_argument("--events", metavar="A:B", help="extract the entries A..B-1 (default: every event)")
    ap.add_argument("--particles", choices=("all", "hit"), default="all",
                    help="all MCParticles, or only those with a tracker hit; hit stores no "
                         "daughter links (default: all)")
    ap.add_argument("--home", default=home, metavar="CONFIG_ROOT",
                    help="config root whose geometry/ was simulated (default: $CALOMAPS_HOME, else this repo)")
    args = ap.parse_intermixed_args(argv)
    IN = args.in_root
    OUT = args.out_npz or os.path.join(home, "models", os.path.splitext(os.path.basename(IN))[0] + ".npz")
    GEO = geometry_provenance(args.home)   # before any ROOT I/O
    check_thickness_label(IN, GEO, args.home)
    thick_mm = float(GEO["geo_si_thick_mm"])
    hit_only = args.particles == "hit"

    import uproot
    f = uproot.open(IN)
    SIM, sim_source = sim_provenance(f, IN)
    t = f["events"]
    n_file = t.num_entries
    if n_file == 0:
        raise SystemExit(f"{IN}: the file holds no events")
    start, stop = _event_range(args.events, args.event_index, n_file)

    branches = list(_PARTICLE_BRANCHES.values()) + list(_HIT_BRANCHES.values()) + [_DAUGHTER_BRANCH]
    parts, hits_per_event, n_part, n_dau, entry = [], [], 0, 0, start
    for a in t.iterate(branches, entry_start=start, entry_stop=stop, step_size=CHUNK_EVENTS,
                       library="ak"):
        chunk, n_hit = _read_chunk(a, entry, hit_only, n_part, n_dau)
        parts.append(chunk)
        hits_per_event.append(n_hit)
        n_part += len(chunk["pdg"])
        n_dau += len(chunk.get("dau", ()))
        entry += len(n_hit)
    if entry != stop:
        raise SystemExit(f"{IN}: read entries {start}..{entry - 1}, expected {start}..{stop - 1}")
    D = {k: np.concatenate([c[k] for c in parts]) for k in parts[0]}
    for k, dt in _DTYPES.items():
        if k in D:
            D[k] = D[k].astype(dt)
    hits_per_event = np.concatenate(hits_per_event)
    D["E"] = np.sqrt(D["px"]**2 + D["py"]**2 + D["pz"]**2 + D["mass"]**2)

    b = path_length_test(D, thick_mm)
    if b["ok"] is False:
        raise SystemExit(
            f"{IN}: the tracker-hit path lengths do not fit the {thick_mm * 1e3:g} um silicon of "
            f"--home {args.home}: median pathLength*|cos|/thickness = {b['ratio']:.4f} over "
            f"{b['n_sel']} clean crossings (allowed {1 - PATH_TOL:.2f}-{1 + PATH_TOL:.2f}), which "
            f"points to about {b['ratio'] * thick_mm * 1e3:.0f} um. --home is not the geometry this "
            f"file was simulated with; nothing written.")

    particle, energy, seed = (str(SIM["sim_gun_particle"]), float(SIM["sim_gun_energy_GeV"]),
                              str(SIM["sim_seed"]))
    order = ["pdg", "mass", "px", "py", "pz", "E", "pex", "pey", "pez",
             "vsx", "vsy", "vsz", "vex", "vey", "vez", "pid", "status", "gstat", "dbeg", "dend", "dau",
             "thx", "thy", "thz", "tpx", "tpy", "tpz", "tpath", "ttime", "tedep", "tcellID", "tmc",
             "evt_id_particle", "evt_id_hit"]
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez_compressed(
        OUT,
        **{k: D[k] for k in order if k in D},
        num_events=np.int64(stop - start), particles_kept=np.array(args.particles),
        **SIM,
        **GEO,
        meta=np.array([particle, f"{energy:g} GeV" if math.isfinite(energy) else "unknown",
                       "+y pencil beam", f"seed={seed}",
                       "ECalBarrel as Geant4TrackerWeightedAction", "SimTrackerHit w/ momentum", "EDM4hep"]),
    )
    tpx, tpy, tpz, tedep = D["tpx"], D["tpy"], D["tpz"], D["tedep"]
    tp = np.sqrt(tpx**2 + tpy**2 + tpz**2)
    n_sim = int(SIM["sim_n_events"])
    print(f"saved {OUT}")
    print(f"  simulation ({sim_source}): {particle}, "
          + (f"{energy:g} GeV" if math.isfinite(energy) else "energy unknown")
          + f", seed {seed}, "
          + (f"{n_sim} event{'' if n_sim == 1 else 's'} simulated" if n_sim >= 0 else "events simulated unknown"))
    if n_sim >= 0 and n_sim != n_file:
        print(f"  WARNING: the file holds {n_file} events; its ddsim run header says {n_sim} were simulated")
    print(f"  events: {stop - start} extracted, entries {start}..{stop - 1} of the file's {n_file}")
    print(f"  geometry: {GEO['geo_si_thick_mm'] * 1e3:g} um Si, config root {GEO['geo_home']}")
    if b["ok"]:
        print(f"  thickness confirmed by {b['n_sel']} clean crossings: median "
              f"pathLength*|cos|/thickness = {b['ratio']:.4f}")
    else:
        print(f"  thickness NOT confirmed: {b['n_sel']} clean crossings, fewer than the {MIN_SELECTED} "
              f"the path-length test needs; the data could not confirm {thick_mm * 1e3:g} um silicon")
    kept = (f"  (--particles hit: only those with a tracker hit; no daughter links stored)"
            if hit_only else "")
    print(f"  n_particles={len(D['pdg'])}  n_tracker_hits={len(tpx)}{kept}")
    if len(hits_per_event) <= 20:
        print("  tracker hits per event: "
              + "  ".join(f"{start + i}:{n}" for i, n in enumerate(hits_per_event)))
    else:
        print(f"  tracker hits per event: min {hits_per_event.min()}  median "
              f"{np.median(hits_per_event):g}  max {hits_per_event.max()}; "
              f"{int(np.count_nonzero(hits_per_event == 0))} events without hits")
    print(f"  hit |p| (GeV): min={tp.min():.4f}  median={np.median(tp):.4f}  max={tp.max():.2f}" if len(tpx) else "  no tracker hits")
    print(f"  total eDep in Si (GeV): {tedep.sum():.4f}")
    print(f"  file size: {os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
