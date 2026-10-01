#!/usr/bin/env python3
"""Check a tracker-route npz (extract_trackermom.py) against the silicon thickness it claims.

    python analysis/validate_crossings.py models/trackermom_gamma50_1evt_si100um.npz
    python analysis/validate_crossings.py models/trackermom_gamma50_1evt.npz --si-thickness-um 100

The thickness under test is the npz's geo_si_thick_mm, or the 320 um baseline (SI_THICK_MM) for
an npz without geometry provenance; --si-thickness-um overrides both. Both checks use
sensor_crossings' own local frame, face assignment and layer centres:

  (a) containment: every tracker hit lies in the Si slab of its nearest layer,
      |w - centre| <= thickness/2 + 0.002 mm. A hit position is an energy-weighted point of
      the crossing's steps, all of which are inside the sensitive slab.
  (b) path length: for clean through-going crossings the Geant4 pathLength is the straight
      chord thickness/|cos(theta)|, theta being the inclination to the sensor normal. Clean
      means: a charged particle, |p| > 0.1 GeV at the crossing (multiple scattering and energy
      loss across the slab are negligible), |cos(theta)| >= 0.9, and the particle's production
      vertex and endpoint both outside that slab (it neither starts nor stops in the sensor).
      Passes when the median of pathLength*|cos(theta)|/thickness is within 5% of 1, from at
      least 10 such crossings. On the 320 um 50 GeV gamma and pi+ single events that median is
      1.0000 with the 5-95% range inside 0.1%.

Prints the hit count, the depth-residual range, the selected crossings' median path length and
their median deposited energy in keV. Exit status 0 when both checks pass, 1 otherwise.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensor_crossings as sc   # noqa: E402

DEPTH_TOL_MM = 0.002
P_MIN_GEV = 0.1
COS_MIN = 0.9
PATH_TOL = 0.05
MIN_SELECTED = 10
_KEYS = ("thx", "thy", "thz", "tpx", "tpy", "tpz", "tpath", "tedep", "tmc", "pdg",
         "vsx", "vsy", "vex", "vey")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check tracker-route crossings against the silicon thickness.")
    ap.add_argument("npz", help="tracker-route npz from analysis/extract_trackermom.py")
    ap.add_argument("--si-thickness-um", type=float, metavar="T",
                    help="thickness to test instead of the npz's geo_si_thick_mm / the 320 um default")
    args = ap.parse_args(argv)

    d = sc.load_cascade(args.npz)
    missing = [k for k in _KEYS if k not in d]
    if missing:
        sys.exit(f"{args.npz}: not a tracker-route npz from extract_trackermom.py "
                 f"(missing {', '.join(missing)})")
    if args.si_thickness_um is not None:
        thick, source = args.si_thickness_um / 1000.0, "--si-thickness-um"
    elif "geo_si_thick_mm" in d:
        thick, source = sc.si_thickness_mm(d), "npz geo_si_thick_mm"
    else:
        thick, source = sc.si_thickness_mm(d), "SI_THICK_MM default; the npz has no geometry provenance"
    if not thick > 0:
        sys.exit(f"thickness must be positive, got {thick * 1e3:g} um")
    half = thick / 2

    thx, thy, thz = (np.asarray(d[k], float) for k in ("thx", "thy", "thz"))
    tpx, tpy, tpz = (np.asarray(d[k], float) for k in ("tpx", "tpy", "tpz"))
    tpath, tedep = np.asarray(d["tpath"], float), np.asarray(d["tedep"], float)
    mc = np.asarray(d["tmc"]).astype(int)
    print(f"validate_crossings: {args.npz}")
    print(f"  thickness under test: {thick * 1e3:g} um ({source})")
    if thx.size == 0:
        print("  no tracker hits -> FAIL")
        return 1

    centers = sc.si_layer_centers()
    phin = sc.face_phi(thx, thy)
    _, _, w = sc.to_local(thx, thy, thz, phin)
    layer = np.argmin(np.abs(w[:, None] - centers[None, :]), axis=1)
    resid = w - centers[layer]
    n_out = int(np.count_nonzero(np.abs(resid) > half + DEPTH_TOL_MM))
    ok_a = n_out == 0
    print(f"  (a) containment: {thx.size} hits; depth residual w - centre in "
          f"[{resid.min():+.4f}, {resid.max():+.4f}] mm; allowed |.| <= {half + DEPTH_TOL_MM:.4f} mm; "
          f"{n_out} outside -> {'PASS' if ok_a else 'FAIL'}")

    _, _, pw = sc.to_local(tpx, tpy, tpz, phin)
    p = np.sqrt(tpx**2 + tpy**2 + tpz**2)
    cos = np.abs(pw) / np.where(p > 0, p, np.inf)
    c_hit = centers[layer]
    cphi, sphi = np.cos(phin), np.sin(phin)
    w_vtx = np.asarray(d["vsx"], float)[mc] * cphi + np.asarray(d["vsy"], float)[mc] * sphi
    w_end = np.asarray(d["vex"], float)[mc] * cphi + np.asarray(d["vey"], float)[mc] * sphi
    through = (np.abs(w_vtx - c_hit) > half) & (np.abs(w_end - c_hit) > half)
    sel = sc.is_charged(np.asarray(d["pdg"])[mc]) & (p > P_MIN_GEV) & (cos >= COS_MIN) & through
    n_sel = int(np.count_nonzero(sel))
    print(f"  (b) path length: {n_sel} clean crossings (charged, |p| > {P_MIN_GEV:g} GeV, "
          f"|cos(theta)| >= {COS_MIN:g}, vertex and endpoint outside the slab)")
    if n_sel < MIN_SELECTED:
        ok_b = False
        print(f"      fewer than {MIN_SELECTED} clean crossings; the thickness cannot be tested -> FAIL")
    else:
        ratio = np.median(tpath[sel] * cos[sel] / thick)
        ok_b = abs(ratio - 1.0) <= PATH_TOL
        print(f"      median pathLength {np.median(tpath[sel]):.4f} mm vs median thickness/|cos| "
              f"{np.median(thick / cos[sel]):.4f} mm; median pathLength*|cos|/thickness = {ratio:.4f} "
              f"(allowed {1 - PATH_TOL:.2f}-{1 + PATH_TOL:.2f}) -> {'PASS' if ok_b else 'FAIL'}")
        print(f"      median eDep of these crossings: {np.median(tedep[sel]) * 1e6:.1f} keV")

    ok = ok_a and ok_b
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
