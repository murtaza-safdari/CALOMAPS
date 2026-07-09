#!/usr/bin/env python3
"""Per-energy readout extraction for the conventional-resolution notebook
(03c_conventional_resolution.ipynb).

Runs over ONE mono-energetic ddsim dataset (a fixed-energy beam) and writes a
per-energy .npz with the four readouts that 03c consumes:
    all_truth, all_visible, all_mip, all_hits, all_cluster, E_nominal

This is the batch/per-energy CLI form of the notebook-02 extraction: it applies the +y
wedge / half-MIP cuts (given) and calls YOUR four-readout code (the compute_readouts TODO
below -- the same V/M/H/C you fill in notebook 02) on every event of a fixed-energy dataset.
Loop it over your fixed-energy datasets:

    for E in 1 2 5 10 20 50 100 200 400; do
      python analysis/extract_readouts.py \\
        --datadir $CALOMAPS_DATA_BASE/mono_gamma_${E}GeV \\
        --glob 'sim_photons_part*.root' --energy $E \\
        --out $CALOMAPS_HOME/models/mono_gamma/decal_mono_gamma_E$(printf '%04d' $E)GeV.npz
    done
"""
import os, glob, argparse, xml.etree.ElementTree as ET
import numpy as np, uproot
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- constants (must match notebooks/02_data_extraction.ipynb) ----
CELL_SIZE    = 0.1           # mm, 100 um pitch
MIP_ENERGY   = 85e-6         # GeV, Landau MPV
THRESHOLD    = 0.5 * MIP_ENERGY
NSIDES       = 12
SEG_HALF_DEG = 180.0 / NSIDES  # 15 deg
RMIN, RMAX   = 1264.0, 1403.0  # mm


def _pv(v, c):
    if not v: return 0.0
    if v in c: return _pv(c[v], c)
    s = v.replace('*', ' * ').replace('cm', '10').replace('mm', '1')
    try: return float(eval(s, {"__builtins__": None}, {}))
    except Exception:
        try: return float(v)
        except Exception: return 0.0


def layer_radii(calomaps_home):
    g = os.path.join(calomaps_home, "geometry")
    consts = {x.get("name"): x.get("value")
              for x in ET.parse(os.path.join(g, "SiD_TestBeam.xml")).getroot().findall(".//constant")}
    det = ET.parse(os.path.join(g, "my_custom_ecal.xml")).getroot().find(".//detector[@name='ECalBarrel']")
    # The ECalBarrel_o2_v03 driver starts the stack at rmin + ecal_barrel_tolerance (=env_safety,
    # 0.1 mm), not at rmin. The offset is uniform, so nearest-centre layer BINNING below is
    # unaffected; we add it so absolute depths are correct.
    cur, planes = _pv(det.find("dimensions").get("rmin"), consts) + 0.1, []
    for layer in det.findall("layer"):
        rep = int(layer.get("repeat", 1)); sl = layer.findall("slice")
        thick = sum(_pv(s.get("thickness"), consts) for s in sl)
        off = sioff = 0.0
        for s in sl:
            t = _pv(s.get("thickness"), consts)
            if s.get("material") == "Silicon": sioff = off + t / 2
            off += t
        for _ in range(rep):
            planes.append(cur + sioff); cur += thick
    return np.array(planes)


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
    # ============================================================
    # TODO: compute the four readouts for this event's wedge hits -- the SAME four
    # you implemented in notebook 02. After the wedge cut above you have, for the
    # hits in this event:
    #   e         : pixel energies [GeV]
    #   thr       : boolean mask, True where e > THRESHOLD (1/2 MIP)
    #   x, z      : pixel coordinates [mm]   ;   layer_idx : silicon layer per hit
    #     V (analog) : sum of ALL hit energies in the segment                    -> float
    #     M (MIP)    : over FIRED pixels, sum of max(1, round(E_pix/MIP_ENERGY)) -> float
    #     H (hits)   : number of fired pixels (count of the thr mask)            -> int
    #     C (cluster): naive_clusters(x, z, layer_idx, e)   (given above)        -> int
    # Solution: not distributed with this repository.
    # ============================================================
    raise NotImplementedError("port your notebook-02 readout code (V, M, H, C) here")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True, help="dir of mono-energetic .root files")
    ap.add_argument("--glob", default="sim_photons_part*.root")
    ap.add_argument("--energy", type=float, required=True, help="nominal beam energy [GeV]")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--home", default=os.environ.get("CALOMAPS_HOME", os.path.expanduser("~/CALOMAPS")))
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    args = ap.parse_args()

    radii = layer_radii(args.home)
    files = sorted(glob.glob(os.path.join(args.datadir, args.glob)))
    if not files:
        raise SystemExit(f"no files match {os.path.join(args.datadir, args.glob)}")
    print(f"E={args.energy} GeV: {len(files)} files, {args.workers} workers")

    T, V, M, H, C = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
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
                        E_nominal=np.float64(args.energy))
    print(f"  {all_truth.size} events -> {args.out}  "
          f"(truth mean={all_truth.mean():.2f} std={all_truth.std():.3f} GeV)")


if __name__ == "__main__":
    main()
