#!/usr/bin/env python3
"""Check a tracker-route npz (extract_trackermom.py) against the silicon thickness it claims.

    python $CALOMAPS_HOME/analysis/validate_crossings.py \
        $CALOMAPS_HOME/models/trackermom_gamma50_1evt_si100um.npz
    python $CALOMAPS_HOME/analysis/validate_crossings.py \
        $CALOMAPS_HOME/models/trackermom_gamma50_1evt.npz --si-thickness-um 310   # FAILs

The thickness under test is the npz's geo_si_thick_mm, or the 320 um baseline (SI_THICK_MM) for
an npz without geometry provenance. --si-thickness-um tests another value; when the npz records
a different geo_si_thick_mm the run says so and fails, whatever the data show. Both checks use
the frame the crossing records use, sensor_crossings.tracker_hit_frame: face and layer from the
hit's cellID module and layer when the npz holds them (tmodule, tlayer), and the layer centres
of sensor_crossings. Their depth tests are sensor_crossings.tracker_hit_slab, which also sets the
crossing records' outside_si, starts_in_si and stops_in_si flags:

  (a) containment: the tracker hits lie in the Si slab of their layer, |w - centre| <=
      thickness/2 + 0.002 mm, w being the depth along the face normal. The hit position is the
      midpoint of the hit's first and last step points (the run header's action.tracker
      HitPositionCombination 2), inside the slab when both points are. Passes when at most
      0.1% of the hits are outside (a thickness below the true one puts ~20% outside). The
      allowance is for merged hits: the tracker action closes a hit when the track leaves the
      Si volume, but the ECal's step filter (the run header's filter.calo, edep0, which ddsim
      gives the calorimeter-type ECalBarrel) drops steps without a deposit, so the exit step of
      a track that deposits nothing there never reaches the action, and the 12 staves are
      placements of one volume, so a later deposit of the same track in the same layer of
      another stave joins the open hit and the midpoint falls between the two. Seen for
      neutrons, 1 hit in 90,438 of a 20-event 50 GeV photon run, 434 mm from its slab.
      An npz without tmodule/tlayer takes the face the hit's azimuth points at and the layer
      nearest in depth; the barrel's staves overlap past the 30-degree sector boundaries, so a
      hit in an overlap region gets the neighbouring face and lands outside its slab (about 2
      hits in 10,000): re-extract such an npz with extract_trackermom.py.
  (b) path length (path_length_test, which extract_trackermom.py also applies before it writes
      an npz): for clean through-going crossings the Geant4 pathLength is the straight chord
      thickness/|cos(theta)|, theta being the inclination to the sensor normal. Clean means: a
      charged particle, |p| > 0.1 GeV at the crossing (multiple scattering and energy loss
      across the slab are negligible), |cos(theta)| >= 0.9, and the particle's production vertex
      and endpoint both outside that slab (it neither starts nor stops in the sensor). Passes
      when the median of pathLength*|cos(theta)|/thickness is within 1% of 1, from at least 10
      such crossings. On the 320 um 50 GeV single events that median is 1.000001 (gamma, 244
      crossings; 5-95% range 0.9997-1.0003, extremes 0.9967 and 1.0010) and 1.000000 (pi+, 30
      crossings), so a thickness 1% off, 3.2 um at 320 um, already fails.

Every thickness variant has the baseline's Si mid-planes, so only (b), and (a) for a thickness
below the true one, can tell a mislabelled npz from a correct one.

An npz of step hits (hit_mode step: sim/run_sim_trackermom.py with CALOMAPS_TRACKER_HITS=step,
one hit per Geant4 step) has no crossings, so (b) does not apply and is skipped; step_depth_test,
which extract_trackermom.py also applies to such an npz, takes the place of both checks:
  (a) containment: as above, but no hit may lie outside its slab. A step lies inside one volume,
      so its midpoint lies inside the slab whenever both its points do.
  (b) reach: the depth residuals reach both faces, the largest outward and inward |w - centre|
      within 1% of thickness/2, over at least 1000 hits. Short steps start or end on the faces
      (the 50 GeV photon showers at 100 um come within 0.003 um of them), so a thickness above
      the true one leaves a gap; (a) catches one below it.

Prints the hit count, the depth-residual range, the hits outside their slab, the selected
crossings' median path length and their median deposited energy in keV (for step hits, the
reach of the residuals instead). Exit status 0 when both checks pass; 3 when (a) passes but
fewer than 10 clean crossings (1000 step hits) leave the thickness untested by (b); 1 otherwise.
"""
import argparse, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensor_crossings as sc   # noqa: E402

DEPTH_TOL_MM = sc.SLAB_TOL_MM   # the same tolerance sets the records' outside_si flag
MAX_OUTSIDE_FRACTION = 1e-3
P_MIN_GEV = 0.1
COS_MIN = 0.9
PATH_TOL = 0.01
MIN_SELECTED = 10
UNDECIDED = 3
STEP_MAX_OUTSIDE = 0
REACH_TOL = 0.01
MIN_STEPS = 1000
_KEYS = ("thx", "thy", "thz", "tpx", "tpy", "tpz", "tpath", "tedep", "tmc", "pdg",
         "vsx", "vsy", "vex", "vey")


def step_depth_test(d, thick_mm):
    """The step-hit checks of the module docstring on tracker-route arrays `d` of step hits for a
    thick_mm sensor. Returns a dict:
      n           number of step hits
      n_out       hits outside their slab (sensor_crossings.tracker_hit_slab)
      contained   n_out <= STEP_MAX_OUTSIDE
      reach_in, reach_out   the largest inward and outward |w - centre| over thick_mm/2 (None
                  without hits)
      ok          the reach test: True when both are >= 1 - REACH_TOL, False when either is not,
                  None when there are fewer than MIN_STEPS hits"""
    slab = sc.tracker_hit_slab(d, thick_mm)
    resid = slab["resid"] / (thick_mm / 2)
    n_out = int(np.count_nonzero(slab["outside"]))
    out = {"n": int(resid.size), "n_out": n_out, "contained": n_out <= STEP_MAX_OUTSIDE,
           "reach_in": None, "reach_out": None, "ok": None}
    if resid.size:
        out.update(reach_in=float(max(-resid.min(), 0.0)), reach_out=float(max(resid.max(), 0.0)))
    if resid.size >= MIN_STEPS:
        out["ok"] = min(out["reach_in"], out["reach_out"]) >= 1 - REACH_TOL
    return out


def path_length_test(d, thick_mm):
    """Check (b) of the module docstring on tracker-route arrays `d` (an npz from
    extract_trackermom.py, or a dict with the same keys) for a thick_mm sensor. Returns a dict:
      ok        True / False, or None when fewer than MIN_SELECTED clean crossings
      n_sel     number of clean crossings
      ratio     median pathLength*|cos(theta)|/thickness (None when n_sel is 0)
      path_mm, chord_mm, edep_kev   medians of pathLength, thickness/|cos(theta)|, eDep"""
    tpx, tpy, tpz = (np.asarray(d[k], float) for k in ("tpx", "tpy", "tpz"))
    tpath, tedep = np.asarray(d["tpath"], float), np.asarray(d["tedep"], float)
    mc = np.asarray(d["tmc"]).astype(int)
    slab = sc.tracker_hit_slab(d, thick_mm)
    _, _, pw = sc.to_local(tpx, tpy, tpz, slab["phin"])
    p = np.sqrt(tpx**2 + tpy**2 + tpz**2)
    cos = np.abs(pw) / np.where(p > 0, p, np.inf)
    through = ~slab["starts_in"] & ~slab["stops_in"]
    sel =sc.is_charged(np.asarray(d["pdg"])[mc]) & (p > P_MIN_GEV) & (cos >= COS_MIN) & through
    n_sel = int(np.count_nonzero(sel))
    out = {"ok": None, "n_sel": n_sel, "ratio": None, "path_mm": None, "chord_mm": None, "edep_kev": None}
    if n_sel:
        out.update(ratio=float(np.median(tpath[sel] * cos[sel] / thick_mm)),
                   path_mm=float(np.median(tpath[sel])),
                   chord_mm=float(np.median(thick_mm / cos[sel])),
                   edep_kev=float(np.median(tedep[sel]) * 1e6))
    if n_sel >= MIN_SELECTED:
        out["ok"] = abs(out["ratio"] - 1.0) <= PATH_TOL
    return out


def _validate_steps(d, thick, conflict):
    """main() for an npz of step hits: step_depth_test in place of (a) and (b)."""
    s = step_depth_test(d, thick)
    half = thick / 2
    print("  hits: one per Geant4 step (hit_mode step); frame: face and layer from the "
          + ("hit cellIDs (tmodule, tlayer)" if sc.has_cellid_frame(d) else
             "hit azimuth and depth (the npz has no tmodule/tlayer)"))
    print(f"  (a) containment: {s['n']} step hits; depth residual w - centre reaches "
          f"{-s['reach_in'] * half:+.4f} and {s['reach_out'] * half:+.4f} mm; allowed |.| <= "
          f"{half + DEPTH_TOL_MM:.4f} mm; {s['n_out']} outside ({STEP_MAX_OUTSIDE} allowed) -> "
          f"{'PASS' if s['contained'] else 'FAIL'}")
    print("  (b) path length: skipped; a step hit's path length is one Geant4 step, not a crossing's. "
          "In its place, reach:")
    if s["ok"] is None:
        print(f"      fewer than {MIN_STEPS} step hits; the thickness cannot be tested -> UNDECIDED")
    else:
        print(f"      the residuals reach {s['reach_in']:.4f} (inward) and {s['reach_out']:.4f} "
              f"(outward) of thickness/2 (at least {1 - REACH_TOL:.2f} needed) -> "
              f"{'PASS' if s['ok'] else 'FAIL'}")
    if conflict or not s["contained"] or s["ok"] is False:
        print("  RESULT: FAIL")
        return 1
    if s["ok"] is None:
        print("  RESULT: UNDECIDED (containment passes; too few step hits to test the thickness)")
        return UNDECIDED
    print("  RESULT: PASS")
    return 0


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
    recorded = float(d["geo_si_thick_mm"]) if "geo_si_thick_mm" in d else None
    if args.si_thickness_um is not None:
        thick, source = args.si_thickness_um / 1000.0, "--si-thickness-um"
    elif recorded is not None:
        thick, source = recorded, "npz geo_si_thick_mm"
    else:
        thick, source = sc.si_thickness_mm(d), "SI_THICK_MM default; the npz has no geometry provenance"
    if not (math.isfinite(thick) and thick > 0):
        sys.exit(f"thickness must be positive, got {thick * 1e3:g} um")
    conflict = recorded is not None and not math.isclose(thick, recorded, rel_tol=1e-5)
    half = thick / 2

    thx = np.asarray(d["thx"], float)
    print(f"validate_crossings: {args.npz}")
    print(f"  thickness under test: {thick * 1e3:g} um ({source})")
    if conflict:
        print(f"  the npz records {recorded * 1e3:g} um silicon (geo_si_thick_mm), not the "
              f"{thick * 1e3:g} um under test -> FAIL whatever the checks below show")
    if thx.size == 0:
        print("  no tracker hits -> FAIL")
        return 1
    if sc.hit_mode(d) == "step":
        return _validate_steps(d, thick, conflict)

    cellid = sc.has_cellid_frame(d)
    slab = sc.tracker_hit_slab(d, thick)
    resid, outside = slab["resid"], slab["outside"]
    n_out = int(np.count_nonzero(outside))
    ok_a = n_out <= MAX_OUTSIDE_FRACTION * thx.size
    print("  frame: face and layer from the "
          + ("hit cellIDs (tmodule, tlayer)" if cellid else
             "hit azimuth and depth (the npz has no tmodule/tlayer)"))
    print(f"  (a) containment: {thx.size} hits; depth residual w - centre in "
          f"[{resid.min():+.4f}, {resid.max():+.4f}] mm; allowed |.| <= {half + DEPTH_TOL_MM:.4f} mm; "
          f"{n_out} outside (at most {MAX_OUTSIDE_FRACTION:.1%} allowed) -> {'PASS' if ok_a else 'FAIL'}")
    if n_out:
        charged = sc.is_charged(np.asarray(d["pdg"])[np.asarray(d["tmc"]).astype(int)])
        print(f"      outside: {int(np.count_nonzero(outside & charged))} of charged particles, "
              f"{int(np.count_nonzero(outside & ~charged))} of neutral ones (neutral hits make no crossings)")
    if n_out and not cellid:
        print("      this npz has no cellID module/layer: hits where neighbouring staves overlap "
              "get the face their azimuth points at, the wrong one. Re-extract it with "
              "analysis/extract_trackermom.py, which stores them")

    b = path_length_test(d, thick)
    print(f"  (b) path length: {b['n_sel']} clean crossings (charged, |p| > {P_MIN_GEV:g} GeV, "
          f"|cos(theta)| >= {COS_MIN:g}, vertex and endpoint outside the slab)")
    if b["ok"] is None:
        print(f"      fewer than {MIN_SELECTED} clean crossings; the thickness cannot be tested -> UNDECIDED")
    else:
        print(f"      median pathLength {b['path_mm']:.4f} mm vs median thickness/|cos| "
              f"{b['chord_mm']:.4f} mm; median pathLength*|cos|/thickness = {b['ratio']:.4f} "
              f"(allowed {1 - PATH_TOL:.2f}-{1 + PATH_TOL:.2f}) -> {'PASS' if b['ok'] else 'FAIL'}")
        print(f"      median eDep of these crossings: {b['edep_kev']:.1f} keV")

    if conflict or not ok_a or b["ok"] is False:
        print("  RESULT: FAIL")
        return 1
    if b["ok"] is None:
        print("  RESULT: UNDECIDED (containment passes; too few clean crossings to test the thickness)")
        return UNDECIDED
    print("  RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
