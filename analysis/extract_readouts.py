#!/usr/bin/env python3
"""Per-energy readout extraction for the conventional-resolution notebook
(03_resolution_conventional.ipynb).

Runs over ONE mono-energetic ddsim dataset (a fixed-energy beam) and writes a
per-energy .npz with the four readouts that notebook 03 consumes:
    all_truth, all_visible, all_mip, all_hits, all_cluster, E_nominal

The readout definitions (analog sum, MIPs-per-pixel, fired-pixel count, 8-connected
clusters) and the +y wedge / half-MIP cuts match notebooks/02_data_extraction.ipynb
-- this is the batch/per-energy CLI form of that extraction. Loop it over your fixed-energy datasets:

    for E in 1 2 5 10 20 50 100 200 400; do
      python analysis/extract_readouts.py \\
        --datadir $CALOMAPS_DATA_BASE/mono_gamma_${E}GeV \\
        --glob 'sim_photons_part*.root' --energy $E \\
        --out $CALOMAPS_HOME/models/mono_gamma/decal_mono_gamma_E$(printf '%04d' $E)GeV.npz
    done

Geometry sweeps: the pixel pitch (ECal_cell_size) and the layer depths are
re-read from <home>/geometry/*.xml, so point --home at the (possibly
copied-and-edited) geometry you simulated with and those follow automatically.
The barrel envelope is assumed unchanged: the wedge cut uses the baseline
NSIDES/RMIN/RMAX constants below, so if you edit ECalBarrel_rmin/rmax/symmetry,
update those constants to match. The MIP scale cannot come from geometry (it is
a measured Landau MPV, 85 keV for 320 um silicon): when the --home silicon thickness
is not this checkout's, the run stops before reading any data unless --mip-energy is
given. A startup probe cross-checks the first file's hit lattice and layer depths
against --home and aborts on a mismatch. The output also records the --home silicon
thickness (geo_si_thick_mm) and the MIP scale used (mip_energy_GeV).

Silicon-thickness variants: with CALOMAPS_SI_THICKNESS_UM=T the sim wrappers simulate the
copy sim/make_geometry_variant.py writes at $CALOMAPS_DATA_BASE/geometry_si<T>um_<hash> (the
run prints its compact) and tag their outputs _si<T>um, e.g. for T=100:
    --datadir $CALOMAPS_DATA_BASE/mono_gamma_50GeV_si100um --glob 'sim_photons_si100um_part*.root'
    --home $CALOMAPS_DATA_BASE/geometry_si100um_<hash> --mip-energy <MPV for 100 um>
"""
import os, glob, argparse, math
import numpy as np, uproot
# _UNITS_MM and _pv are unused here but stay importable from this module
from decal_geometry import UNITS_MM as _UNITS_MM, _pv, layer_radii, cell_size_mm, si_thickness_mm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- readout constants (baseline defaults; match notebooks/02_data_extraction.ipynb).
#      main() re-resolves CELL_SIZE from --home's geometry XML and MIP_ENERGY from
#      --mip-energy, so the CLI cannot silently disagree with the simulated detector. ----
CELL_SIZE    = 0.1           # mm, 100 um pitch (ECal_cell_size; re-read per --home)
MIP_ENERGY   = 85e-6         # GeV, Landau MPV for 320 um Si (see --mip-energy)
THRESHOLD    = 0.5 * MIP_ENERGY
NSIDES       = 12              # ECalBarrel_symmetry (NOT re-read; assumed unchanged in sweeps)
SEG_HALF_DEG = 180.0 / NSIDES  # 15 deg
RMIN, RMAX   = 1264.0, 1403.0  # mm, ECalBarrel_rmin/rmax (NOT re-read; assumed unchanged)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def naive_clusters(x, z, layer_idx, e):
    """8-connected components per layer, summed (identical to nb02's baseline)."""
    m = e > THRESHOLD
    if not m.any():
        return 0
    xi = np.round(x[m] / CELL_SIZE).astype(np.int64)
    zi = np.round(z[m] / CELL_SIZE).astype(np.int64)
    li = layer_idx[m]
    total = 0
    for ly in np.unique(li):
        sel = li == ly
        cells_ = set(zip(xi[sel].tolist(), zi[sel].tolist()))
        seen = set()
        for c0 in cells_:
            if c0 in seen:
                continue
            total += 1
            stack = [c0]
            while stack:
                ux, uz = stack.pop()
                if (ux, uz) in seen:
                    continue
                seen.add((ux, uz))
                for dx in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        v = (ux + dx, uz + dz)
                        if v in cells_ and v not in seen:
                            stack.append(v)
    return total


def compute_readouts(x, y, z, e, radii):
    """The four readouts for ONE event's hit arrays. Pure function (no ROOT) so
    it is unit-testable. Returns (V, M, H, C) or None if no hits survive the wedge.
    This is the FILLED-IN version of nb02's V/M/H/C TODO block."""
    if len(e) == 0:
        return None
    ang = np.degrees(np.arctan2(x, y))
    seg = (np.abs(ang) < SEG_HALF_DEG) & (y > RMIN - 4) & (y < RMAX + 14)
    x, z, e, y = x[seg], z[seg], e[seg], y[seg]
    if len(e) == 0:
        return None
    layer_idx = np.argmin(np.abs(radii[None, :] - y[:, None]), axis=1)
    thr = e > THRESHOLD
    # --- the four readouts (nb02 exercise solution) ---
    V = float(np.sum(e))                                           # analog: all wedge hits
    M = float(np.sum(np.maximum(1, np.round(e[thr] / MIP_ENERGY))))  # MIPs-per-pixel over fired
    H = int(np.count_nonzero(thr))                                 # binary hits above 1/2 MIP
    C = int(naive_clusters(x, z, layer_idx, e))                    # naive 8-connected clusters
    return V, M, H, C


def process_single_file(filepath, radii):
    br = ["ECalBarrelHits.position.x", "ECalBarrelHits.position.y",
          "ECalBarrelHits.position.z", "ECalBarrelHits.energy",
          "MCParticles.momentum.x", "MCParticles.momentum.y",
          "MCParticles.momentum.z", "MCParticles.mass"]
    T, V, M, H, C = [], [], [], [], []
    try:
        with uproot.open(filepath) as f:
            tr = f["events"]
            if tr.num_entries == 0:
                return None
            a = tr.arrays(br)
            truth = np.sqrt(a["MCParticles.momentum.x"][:, 0]**2 +
                            a["MCParticles.momentum.y"][:, 0]**2 +
                            a["MCParticles.momentum.z"][:, 0]**2 +
                            a["MCParticles.mass"][:, 0]**2)
            hx, hy = a["ECalBarrelHits.position.x"], a["ECalBarrelHits.position.y"]
            hz, he = a["ECalBarrelHits.position.z"], a["ECalBarrelHits.energy"]
            for ev in range(len(he)):
                out = compute_readouts(np.asarray(hx[ev]), np.asarray(hy[ev]),
                                       np.asarray(hz[ev]), np.asarray(he[ev]), radii)
                if out is None:
                    continue
                v, m, h, c = out
                T.append(float(truth[ev])); V.append(v); M.append(m); H.append(h); C.append(c)
    except Exception as ex:
        print(f"  failed {os.path.basename(filepath)}: {ex}")
        return None
    return T, V, M, H, C


def _init_readout_constants(cell_size, mip_energy):
    """Executor initializer: propagate the resolved pitch/MIP scale into each worker
    process (module globals would only survive fork, not spawn)."""
    global CELL_SIZE, MIP_ENERGY, THRESHOLD
    CELL_SIZE, MIP_ENERGY, THRESHOLD = cell_size, mip_energy, 0.5 * mip_energy


def _lattice_probe(x, z, y, radii):
    """Return an error message if the hits are inconsistent with the resolved
    geometry (i.e. --home does not match the simulated detector), else None.
    Fraction/mean based: hits from the wedge's rotated neighbour staves sit off
    the +y module's lattice, so a small off-lattice tail is normal."""
    frac_off = max(np.mean(np.abs(x / CELL_SIZE - np.round(x / CELL_SIZE)) > 0.05),
                   np.mean(np.abs(z / CELL_SIZE - np.round(z / CELL_SIZE)) > 0.05))
    if frac_off > 0.10:
        return (f"{frac_off:.0%} of hit positions are off the {CELL_SIZE} mm pixel "
                f"lattice -- the simulated pitch is finer than {CELL_SIZE} mm (or unrelated)")
    odd_frac = np.mean(np.abs(x / (2 * CELL_SIZE) - np.round(x / (2 * CELL_SIZE))) > 0.25)
    if odd_frac < 0.05:
        return (f"hits only populate every second {CELL_SIZE} mm cell -- the "
                f"simulated pitch is coarser than {CELL_SIZE} mm")
    mean_dy = float(np.mean(np.min(np.abs(y[:, None] - radii[None, :]), axis=1)))
    if mean_dy > 0.5:
        return (f"hit depths sit {mean_dy:.2f} mm (mean) from the nearest layer "
                f"plane -- the layer stack (Si thickness) does not match")
    return None


def _probe_geometry(files, radii, min_hits=500):
    """Read hits from the first readable file and abort if they are not on the
    pixel lattice / layer planes implied by --home -- catches extracting a swept
    dataset with the wrong geometry BEFORE burning the whole pool run on it."""
    xs, ys, zs, n = [], [], [], 0
    for fp in files:
        try:
            with uproot.open(fp) as f:
                a = f["events"].arrays(["ECalBarrelHits.position.x",
                                        "ECalBarrelHits.position.y",
                                        "ECalBarrelHits.position.z"])
        except Exception:
            continue
        for ev in range(len(a["ECalBarrelHits.position.x"])):
            hx = np.asarray(a["ECalBarrelHits.position.x"][ev], float)
            hy = np.asarray(a["ECalBarrelHits.position.y"][ev], float)
            hz = np.asarray(a["ECalBarrelHits.position.z"][ev], float)
            ang = np.degrees(np.arctan2(hx, hy))
            m = (np.abs(ang) < SEG_HALF_DEG) & (hy > RMIN - 4) & (hy < RMAX + 14)
            xs.append(hx[m]); ys.append(hy[m]); zs.append(hz[m]); n += int(m.sum())
            if n >= min_hits:
                break
        break                          # first readable file is enough
    if n < min_hits:
        return                         # too few hits to judge -- proceed
    err = _lattice_probe(np.concatenate(xs), np.concatenate(zs), np.concatenate(ys), radii)
    if err:
        raise SystemExit(f"geometry mismatch: {err}. Is --home pointing at the "
                         f"geometry these files were simulated with?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True, help="dir of mono-energetic .root files")
    ap.add_argument("--glob", default="sim_photons_part*.root")
    ap.add_argument("--energy", type=float, required=True, help="nominal beam energy [GeV]")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--home", default=os.environ.get("CALOMAPS_HOME", os.path.expanduser("~/CALOMAPS")),
                    help="config root whose geometry/*.xml supplies the pixel pitch and layer "
                         "depths (default: $CALOMAPS_HOME, else ~/CALOMAPS); for sweeps, point "
                         "at the copied-and-edited geometry you simulated with")
    ap.add_argument("--mip-energy", type=float, metavar="GEV",
                    help="MIP Landau MPV in GeV (default 85e-6 = 320 um Si). Required when the --home "
                         "Si thickness is not this checkout's; the 1/2-MIP threshold follows it.")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    args = ap.parse_args()

    try:
        radii = layer_radii(args.home)
        thick_mm = si_thickness_mm(args.home)
    except FileNotFoundError:
        raise SystemExit(f"--home {args.home!r} has no geometry/SiD_TestBeam.xml + "
                         f"my_custom_ecal.xml -- point it at the repo/config root")
    except ValueError as e:
        raise SystemExit(f"cannot read the geometry under --home {args.home!r}: {e}")
    base_mm = si_thickness_mm(REPO)
    if args.mip_energy is None and not math.isclose(thick_mm, base_mm, rel_tol=1e-5):
        raise SystemExit(f"--home {args.home} has {thick_mm * 1e3:g} um silicon, but the default MIP "
                         f"scale ({MIP_ENERGY * 1e6:g} keV) is the Landau MPV for {base_mm * 1e3:g} um. "
                         f"Pass --mip-energy (GeV) for {thick_mm * 1e3:g} um silicon.")
    mip = MIP_ENERGY if args.mip_energy is None else args.mip_energy
    _init_readout_constants(cell_size_mm(args.home), mip)
    files = sorted(glob.glob(os.path.join(args.datadir, args.glob)))
    if not files:
        raise SystemExit(f"no files match {os.path.join(args.datadir, args.glob)}")
    print(f"E={args.energy} GeV: {len(files)} files, {args.workers} workers  "
          f"[pitch {CELL_SIZE*1e3:g} um, Si {thick_mm*1e3:g} um from --home geometry, "
          f"MIP {MIP_ENERGY*1e6:g} keV]")
    _probe_geometry(files, radii)      # abort early on a data/geometry mismatch

    T, V, M, H, C = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_readout_constants,
                             initargs=(CELL_SIZE, MIP_ENERGY)) as ex:
        futs = {ex.submit(process_single_file, f, radii): f for f in files}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                T += r[0]; V += r[1]; M += r[2]; H += r[3]; C += r[4]
            if n % 50 == 0:
                print(f"  {n}/{len(files)}")

    all_truth = np.array(T)
    if all_truth.size == 0:
        raise SystemExit("0 events extracted -- check dataset/glob")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out,
                        all_truth=all_truth,
                        all_visible=np.array(V, np.float32),
                        all_mip=np.array(M, np.float32),
                        all_hits=np.array(H, np.int64),
                        all_cluster=np.array(C, np.int64),
                        E_nominal=np.float64(args.energy),
                        geo_si_thick_mm=np.float64(thick_mm),
                        mip_energy_GeV=np.float64(MIP_ENERGY))
    print(f"  {all_truth.size} events -> {args.out}  "
          f"(truth mean={all_truth.mean():.2f} std={all_truth.std():.3f} GeV)")


if __name__ == "__main__":
    main()
