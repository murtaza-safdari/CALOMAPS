#!/usr/bin/env python3
"""Per-group pixel charges -> reticles: fixed blocks of the pixel grid, with the particles in them.

    python analysis/pixelav/reticles.py <charges.npz> [more charges.npz ...] --out <reticles.npz>
                                        [--overlay K] [--size-x 21 --size-y 13]
                                        [--offset-x 0 --offset-y 0]
                                        [--truth-cuts 0,500,1000,2000] [--count-radii-mm 2,5,10,20]

The inputs are charges.npz files of analysis/pixelav/charge_maps.py, batches of one sample: the
same sensor, geometry, gun and job options (SAME_KEYS, checked) and different events (two batches
with the same sim_seed must hold disjoint event ranges). Events are numbered from 0 over the
inputs in the order given, each batch's events_first..events_stop-1 in turn; groups are numbered
the same way, so a group id is unique over the inputs.

RETICLES: a reticle is a fixed block of size-x x size-y pixels of a sensor plane, by default the
PIXELAV window (21 x 13 pixels, 1.05 x 0.325 mm in a 50 x 25 um sensor), size-x along v (pitch_x)
and size-y along u (pitch_y). Reticle rv = floor((iv - offset_x)/size_x), ru = floor((iu -
offset_y)/size_y), and its pixel (ix, iy) is iv = offset_x + rv*size_x + ix, iu = offset_y +
ru*size_y + iy (img[i, iy, ix]). The tiling is fixed on the global pixel grid of deposit_jobs.py
(pixel iv covers v in [iv, iv+1) * pitch_x), shifted by --offset-x and --offset-y pixels (default
0 0), and unrelated to the job tiles. The shower axis crosses the axis module at u = v = 0, a pixel
corner: with no offset it sits on the corner of four reticles, which share the densest core; with
offsets (10, 6) it sits within half a pixel of the centre of one 21 x 13 reticle. Different
offsets cut the same events into different reticles, so they give statistically different
reticle samples of one simulation (not independent ones). A sensor plane is (super-event,
module, layer); its image is the sum of the charges of all groups on it. img is analog:
electrons, no noise and no threshold, so the analysis applies the readout model (threshold,
binary hit) and can scan it. Every reticle that holds a pixel row of the inputs is stored.

TRUTH: one part_* row per (reticle, group) with a pixel row inside the reticle. A group is one
independent charged particle in the plane, with the delta rays and other secondaries it made in
the slab (deposit_jobs.py, GROUPS): a delta ray counts with its parent, a photon converting in
the slab gives two. part_q is the group's charge inside the reticle, (part_iv, part_iu) the
reticle pixel where it left the most, and part_pdg, part_p_MeV, part_created_in_slab those of its
root. n_true[i, j] counts the groups whose charge inside reticle i exceeds truth_cuts[j] electrons:
the number of particles a counter should find, for a choice of how much of a particle must lie
inside the reticle to count. A particle crossing a reticle edge counts in each reticle where its
share passes the cut.

OVERLAY: --overlay K sums events 0..K-1 into super-event 0, K..2K-1 into 1, and so on; a remainder
of fewer than K events is dropped. Every shower is fired from GUN_ORIGIN_MM along GUN_DIRECTION
(sim/run_sim_trackermom.py), so a super-event is K showers on top of each other on one axis: the
density of particles rises K-fold at every depth and radius. For the charge this is exact: PIXELAV's
carriers do not interact, so a plane's charge is the sum of its particles' charges whichever event
they came from. It is not a shower of K times the energy, whose maximum lies about ln K radiation
lengths deeper, and its particles come from K independent cascades.

AXIS: r_mm is the distance in the plane from a reticle's centre to the point where the shower axis
crosses the mid-plane of the plane's silicon, in the sensor frame of sensor_crossings (to_local,
module_face_phi, si_layer_centers). The axis enters the module whose face it points at (the axis
module); on every other module there is no crossing and r_mm is NaN. Each plane's charge-weighted
centroid is stored beside its axis point (plane_*), and the summary gives their mean offset in
layers AXIS_CHECK_LAYERS, around the shower maximum.

OCCUPANCY: the planes are those holding charge plus every layer of the axis module in every
super-event, so a layer the showers left empty counts with all its reticles empty. For each radius
R of --count-radii-mm (default COUNT_RADII_MM), plane_n_within[p, j] counts the reticles of the
grid on plane p whose centre lies within R of the axis point (r_mm < R), with charge or without,
and plane_n_hit_within[p, j] those holding charge; both are 0 off the axis module. The occupancy
within R is n_hit_within / n_within, and n_within - n_hit_within reticles are empty.

OUTPUT <reticles.npz>, compressed:
  per reticle (sorted by super_evt, module, layer, rv, ru): img float32 [n, size_y, size_x];
      super_evt, module, layer, rv, ru (int32); r_mm (float64, mm); q_sum (float64, the sum of
      img); n_true int16 [n, n_cuts]
  per (reticle, group), sorted by reticle and group: part_reticle (row of the reticle arrays),
      part_group (global group id), part_q (float32, e), part_iv, part_iu (int16, reticle-local
      pixel), part_pdg (int32), part_p_MeV (float32), part_created_in_slab (bool), part_evt (int32,
      the global event)
  per plane (sorted by super_evt, module, layer): plane_super_evt, plane_module, plane_layer,
      plane_q, plane_u_axis_mm, plane_v_axis_mm, plane_u_centroid_mm, plane_v_centroid_mm (NaN
      without charge), plane_n_reticles (holding charge), plane_n_within, plane_n_hit_within
      [n_planes, n_radii]
  scalars: overlay, size_x, size_y, offset_x, offset_y, truth_cuts, count_radii_mm, n_within, n_empty_within
      [n_radii] (summed over the planes), n_planes, n_reticles, n_parts, n_events, n_super_events,
      n_events_dropped, axis_module, gun_origin_mm, gun_direction, total_q
  inputs: inputs (paths), input_event_offset, input_n_events, input_group_offset, and input_<key>
      for every per-batch key (BATCH_KEYS) and every other scalar that differs between batches;
      the scalars that agree (SAME_KEYS among them) as they are.
Nothing is written unless the sum of img equals the summed charge of the input pixel rows of the
events used (relative 1e-6), and every reticle's part_q sum to its img (relative 1e-6 of its
absolute charge).
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import sensor_crossings as sc                # noqa: E402

GUN_ORIGIN_MM = (0.0, 0.0, 0.0)   # sim/run_sim_trackermom.py: SIM.gun.position
GUN_DIRECTION = (0.0, 1.0, 0.0)   # theta = phi = 90 deg: along +y
SIZE_X, SIZE_Y = 21, 13           # default reticle, pixels along v and u: the PIXELAV window
TRUTH_CUTS = "0,500,1000,2000"    # electrons
COUNT_RADII_MM = "2,5,10,20"
AXIS_CHECK_LAYERS = (5, 12)       # first and last layer of the centroid-to-axis summary
CHECK_REL = 1e-6
N_LAYERS = len(sc.si_layer_centers())
# must agree between batches of one sample
SAME_KEYS = ("pitch_x_um", "pitch_y_um", "thick_um", "ev_per_pair", "pixelav_title",
             "pixelav_stimstp_ps", "hit_mode", "particles_kept", "tcellID_encoding",
             "sim_gun_particle", "sim_gun_energy_GeV", "sim_dd4hep_version", "sim_geant4_version",
             "geo_si_thick_mm", "geo_si_centers_mm", "geo_cell_size_mm", "geo_si_cut_mm",
             "geo_si_step_mm")
# describe one batch: kept per input as input_<key>
BATCH_KEYS = ("sim_seed", "sim_date_utc", "sim_n_events", "num_events", "events_first",
              "events_stop", "source_npz", "jobs_npz", "geo_home", "sensor_source", "parts",
              "n_jobs", "n_groups", "n_pix", "total_q", "total_neh", "total_n_out", "total_edep_keV")
_NEEDED = ("pix_group_id", "pix_iv", "pix_iu", "pix_q", "group_id", "group_evt_id", "group_module",
           "group_layer", "group_root_pdg", "group_root_p_MeV", "group_created_in_slab",
           "events_first", "events_stop", "pitch_x_um", "pitch_y_um")


def _floats(spec, what):
    try:
        x = np.array([float(s) for s in spec.split(",")])
    except ValueError:
        raise SystemExit(f"reticles: {what} {spec!r}: expected comma-separated numbers") from None
    if not (np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)):
        raise SystemExit(f"reticles: {what} {spec!r}: expected increasing finite numbers")
    return x


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def load_inputs(paths):
    """The arrays of every charges npz of paths; SystemExit when a key is missing, a SAME_KEYS
    value differs between them, or two share a sim_seed and overlapping event ranges."""
    D = []
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as d:
                D.append({k: d[k] for k in d.files})
        except (FileNotFoundError, OSError, ValueError) as err:
            sys.exit(f"reticles: cannot read {p}: {err}")
        missing = [k for k in _NEEDED if k not in D[-1]]
        if missing:
            sys.exit(f"reticles: {p}: no {', '.join(missing)}; make it with analysis/pixelav/charge_maps.py")
    for k in SAME_KEYS:
        have = [k in d for d in D]
        if any(have) and not (all(have) and all(_same(d[k], D[0][k]) for d in D[1:])):
            vals = "; ".join(f"{os.path.basename(p)}: {d[k] if k in d else 'none'}" for p, d in zip(paths, D))
            sys.exit(f"reticles: the inputs are not one sample, {k} differs: {vals}")
    seeds = [str(d["sim_seed"]) if "sim_seed" in d else None for d in D]
    for i in range(len(D)):
        for j in range(i):
            a = (int(D[i]["events_first"]), int(D[i]["events_stop"]))
            b = (int(D[j]["events_first"]), int(D[j]["events_stop"]))
            if seeds[i] == seeds[j] and a[0] < b[1] and b[0] < a[1]:
                sys.exit(f"reticles: {paths[j]} and {paths[i]} hold the same events (sim_seed "
                         f"{seeds[i]}, events {b[0]}..{b[1] - 1} and {a[0]}..{a[1] - 1})")
    return D


def merge(D):
    """The groups and pixel rows of all inputs on global event and group numbers: (G, P, info).
    G per group: evt, module, layer, pdg, p_MeV, created; P per pixel row: group, iv, iu, q."""
    G = {k: [] for k in ("evt", "module", "layer", "pdg", "p_MeV", "created")}
    P = {k: [] for k in ("group", "iv", "iu", "q")}
    info = {"event_offset": [], "n_events": [], "group_offset": []}
    n_evt = n_grp = 0
    for d in D:
        first, stop = int(d["events_first"]), int(d["events_stop"])
        evt = np.asarray(d["group_evt_id"], np.int64)
        if evt.size and (evt.min() < first or evt.max() >= stop):
            sys.exit(f"reticles: {d.get('jobs_npz', '')}: group events outside events_first..events_stop")
        gid = np.asarray(d["group_id"], np.int64)
        order = np.argsort(gid, kind="stable")
        pos = np.minimum(np.searchsorted(gid[order], d["pix_group_id"]), max(len(gid) - 1, 0))
        if len(d["pix_group_id"]) and not np.array_equal(gid[order][pos], d["pix_group_id"]):
            sys.exit(f"reticles: {d.get('jobs_npz', '')}: pixel rows of groups the group table does not have")
        G["evt"].append(n_evt + evt - first)
        G["module"].append(np.asarray(d["group_module"], np.int64))
        G["layer"].append(np.asarray(d["group_layer"], np.int64))
        G["pdg"].append(np.asarray(d["group_root_pdg"], np.int32))
        G["p_MeV"].append(np.asarray(d["group_root_p_MeV"], np.float32))
        G["created"].append(np.asarray(d["group_created_in_slab"], bool))
        P["group"].append(n_grp + order[pos])
        P["iv"].append(np.asarray(d["pix_iv"], np.int64))
        P["iu"].append(np.asarray(d["pix_iu"], np.int64))
        P["q"].append(np.asarray(d["pix_q"], np.float32))
        info["event_offset"].append(n_evt)
        info["n_events"].append(stop - first)
        info["group_offset"].append(n_grp)
        n_evt += stop - first
        n_grp += len(gid)
    G = {k: np.concatenate(v) for k, v in G.items()}
    P = {k: np.concatenate(v) for k, v in P.items()}
    info = {k: np.array(v, np.int64) for k, v in info.items()}
    info["n_events_total"], info["n_groups_total"] = n_evt, n_grp
    return G, P, info


def axis_module():
    """The cellID module whose face the shower axis points at."""
    m = np.arange(sc.N_FACES)
    phi = sc.face_phi(GUN_DIRECTION[0], GUN_DIRECTION[1])
    hit = np.flatnonzero(np.isclose(np.cos(sc.module_face_phi(m) - phi), 1.0))
    return int(hit[0])


def axis_point(module, layer):
    """(u, v) [mm] where the shower axis crosses the silicon mid-plane of (module, layer), in that
    plane's sensor frame; NaN off the axis module."""
    module, layer = np.asarray(module, np.int64), np.asarray(layer, np.int64)
    phin = sc.module_face_phi(module)
    ou, ov, ow = sc.to_local(*GUN_ORIGIN_MM, phin)
    du, dv, dw = sc.to_local(*GUN_DIRECTION, phin)
    on_axis = (module == axis_module()) & (dw > 0)
    t = np.where(on_axis, (sc.si_layer_centers()[layer] - ow) / np.where(on_axis, dw, 1.0), np.nan)
    return ou + t * du, ov + t * dv


def count_within(u0, v0, radii, lu, lv):
    """Number of reticle centres ((ru + 1/2) lu, (rv + 1/2) lv) closer than each radius to (u0, v0)
    [mm], measured from the grid's origin."""
    if not (np.isfinite(u0) and np.isfinite(v0)):
        return np.zeros(len(radii), np.int64)
    R = float(radii.max())
    ru = np.arange(np.floor((u0 - R) / lu) - 1, np.ceil((u0 + R) / lu) + 1)
    rv = np.arange(np.floor((v0 - R) / lv) - 1, np.ceil((v0 + R) / lv) + 1)
    r = np.hypot((ru[:, None] + 0.5) * lu - u0, (rv[None, :] + 0.5) * lv - v0)
    return np.array([np.count_nonzero(r < x) for x in radii], np.int64)


def build(G, P, info, overlay, size_x, size_y, cuts, radii, pitch_x_um, pitch_y_um,
          offset_x=0, offset_y=0):
    """The reticle, part and plane arrays of the module docstring, plus the input charge of the
    events used ("q_in")."""
    n_super = info["n_events_total"] // overlay
    used_g = G["evt"] < n_super * overlay
    use = used_g[P["group"]]
    grp, iv, iu, q = P["group"][use], P["iv"][use], P["iu"][use], P["q"][use].astype(np.float64)
    sup = G["evt"] // overlay
    pk = (sup[grp] * sc.N_FACES + G["module"][grp]) * N_LAYERS + G["layer"][grp]
    rv, ru = np.floor_divide(iv - offset_x, size_x), np.floor_divide(iu - offset_y, size_y)
    lx, ly = iv - offset_x - rv * size_x, iu - offset_y - ru * size_y

    # reticles: one per (plane, rv, ru) holding a pixel row
    if len(q):
        rv0, ru0 = int(rv.min()), int(ru.min())
        nrv, nru = int(rv.max()) - rv0 + 1, int(ru.max()) - ru0 + 1
    else:
        rv0 = ru0 = 0
        nrv = nru = 1
    if (int(pk.max(initial=0)) + 1) * nrv * nru >= 2**62:
        sys.exit("reticles: the reticle key overflows")
    ukey, rinv = np.unique((pk * nrv + (rv - rv0)) * nru + (ru - ru0), return_inverse=True)
    rinv = rinv.ravel()
    n, S = len(ukey), size_x * size_y
    upix, pinv = np.unique(rinv * S + ly * size_x + lx, return_inverse=True)
    img = np.zeros(n * S, np.float32)
    img[upix] = np.bincount(pinv.ravel(), q, len(upix))
    img = img.reshape(n, size_y, size_x)
    r_pk = ukey // (nrv * nru)
    R = {"super_evt": r_pk // (sc.N_FACES * N_LAYERS), "module": r_pk // N_LAYERS % sc.N_FACES,
         "layer": r_pk % N_LAYERS, "rv": ukey // nru % nrv + rv0, "ru": ukey % nru + ru0}
    lu, lv = size_y * pitch_y_um * 1e-3, size_x * pitch_x_um * 1e-3
    ou, ov = offset_y * pitch_y_um * 1e-3, offset_x * pitch_x_um * 1e-3   # grid origin [mm]
    u_ax, v_ax = axis_point(R["module"], R["layer"])
    r_mm = np.hypot(ou + (R["ru"] + 0.5) * lu - u_ax, ov + (R["rv"] + 0.5) * lv - v_ax)

    # truth: one row per (reticle, group); the group's largest pixel inside the reticle
    ng = info["n_groups_total"]
    ukey2, inv2 = np.unique(rinv * ng + grp, return_inverse=True)
    inv2 = inv2.ravel()
    part_q = np.bincount(inv2, q, len(ukey2))
    order = np.lexsort((-q, inv2))
    first = order[np.flatnonzero(np.r_[True, np.diff(inv2[order]) != 0])] if len(q) else order
    part_ret, part_grp = ukey2 // ng, ukey2 % ng
    n_true = np.stack([np.bincount(part_ret[part_q > c], minlength=n) for c in cuts], axis=1) \
        if len(cuts) else np.zeros((n, 0), np.int64)
    if n_true.size and n_true.max() > np.iinfo(np.int16).max:
        sys.exit("reticles: more groups in a reticle than int16 holds")

    # planes: those holding charge, and every layer of the axis module in every super-event
    am = axis_module()
    axis_pk = ((np.arange(n_super)[:, None] * sc.N_FACES + am) * N_LAYERS
               + np.arange(N_LAYERS)[None, :]).ravel()
    plane_pk = np.union1d(np.unique(r_pk), axis_pk)
    p_of_row = np.searchsorted(plane_pk, pk)
    p_of_ret = np.searchsorted(plane_pk, r_pk)
    npl = len(plane_pk)
    pl = {"super_evt": plane_pk // (sc.N_FACES * N_LAYERS), "module": plane_pk // N_LAYERS % sc.N_FACES,
          "layer": plane_pk % N_LAYERS}
    pq = np.bincount(p_of_row, q, npl)
    u_pix = (iu + 0.5) * pitch_y_um * 1e-3
    v_pix = (iv + 0.5) * pitch_x_um * 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        pu = np.where(pq != 0, np.bincount(p_of_row, q * u_pix, npl) / pq, np.nan)
        pv = np.where(pq != 0, np.bincount(p_of_row, q * v_pix, npl) / pq, np.nan)
    pl_u_ax, pl_v_ax = axis_point(pl["module"], pl["layer"])
    within_layer = {}
    n_within = np.zeros((npl, len(radii)), np.int64)
    for i in range(npl):
        key = (int(pl["module"][i]), int(pl["layer"][i]))
        if key not in within_layer:
            within_layer[key] = count_within(pl_u_ax[i] - ou, pl_v_ax[i] - ov, radii, lu, lv)
        n_within[i] = within_layer[key]
    n_hit_within = np.stack([np.bincount(p_of_ret[r_mm < x], minlength=npl) for x in radii], axis=1)

    ret = {"img": img, **{k: v.astype(np.int32) for k, v in R.items()},
           "r_mm": r_mm, "q_sum": img.sum(axis=(1, 2), dtype=np.float64), "n_true": n_true.astype(np.int16)}
    part = {"part_reticle": part_ret, "part_group": part_grp, "part_q": part_q.astype(np.float32),
            "part_iv": lx[first].astype(np.int16), "part_iu": ly[first].astype(np.int16),
            "part_pdg": G["pdg"][part_grp], "part_p_MeV": G["p_MeV"][part_grp],
            "part_created_in_slab": G["created"][part_grp],
            "part_evt": G["evt"][part_grp].astype(np.int32)}
    plane = {"plane_super_evt": pl["super_evt"].astype(np.int32),
             "plane_module": pl["module"].astype(np.int32), "plane_layer": pl["layer"].astype(np.int32),
             "plane_q": pq, "plane_u_axis_mm": pl_u_ax, "plane_v_axis_mm": pl_v_ax,
             "plane_u_centroid_mm": pu, "plane_v_centroid_mm": pv,
             "plane_n_reticles": np.bincount(p_of_ret, minlength=npl),
             "plane_n_within": n_within, "plane_n_hit_within": n_hit_within}
    return ret, part, plane, {"q_in": float(q.sum()), "n_super": n_super, "n_rows": len(q)}


def check(ret, part, q_in):
    """(relative difference of the sum of img from q_in, worst relative difference of a reticle's
    part_q sum from its img sum, the reticle): the hard checks, on the arrays as stored."""
    img = ret["img"]
    n = len(img)
    s_img, scale = np.zeros(n), np.zeros(n)
    for a in range(0, n, 65536):              # in blocks: no float64 copy of the whole img
        b = img[a:a + 65536].astype(np.float64)
        s_img[a:a + 65536], scale[a:a + 65536] = b.sum(axis=(1, 2)), np.abs(b).sum(axis=(1, 2))
    total = float(s_img.sum())
    d_total = (total - q_in) / abs(q_in) if q_in else total
    s_part = np.bincount(part["part_reticle"], part["part_q"].astype(np.float64), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, np.abs(s_part - s_img) / scale, np.abs(s_part - s_img))
    worst = int(np.argmax(rel)) if n else -1
    return d_total, (float(rel[worst]) if n else 0.0), worst


def _dist(x):
    if not len(x):
        return "none"
    return (f"mean {x.mean():.2f}, p50 {np.percentile(x, 50):.0f}, p90 {np.percentile(x, 90):.0f}, "
            f"p99 {np.percentile(x, 99):.0f}, max {x.max()}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("charges", nargs="+", help="charges.npz of analysis/pixelav/charge_maps.py, batches of one sample")
    ap.add_argument("--out", required=True, metavar="FILE", help="output npz")
    ap.add_argument("--overlay", type=int, default=1, metavar="K",
                    help="sum consecutive K events into one super-event (default 1)")
    ap.add_argument("--size-x", type=int, default=SIZE_X, metavar="NX",
                    help=f"reticle pixels along v, the pitch_x side (default {SIZE_X})")
    ap.add_argument("--size-y", type=int, default=SIZE_Y, metavar="NY",
                    help=f"reticle pixels along u, the pitch_y side (default {SIZE_Y})")
    ap.add_argument("--offset-x", type=int, default=0, metavar="OX",
                    help="shift of the reticle grid along v, in pixels (default 0; 10 centres a "
                    "21-pixel reticle on the axis)")
    ap.add_argument("--offset-y", type=int, default=0, metavar="OY",
                    help="shift of the reticle grid along u, in pixels (default 0; 6 centres a "
                    "13-pixel reticle on the axis)")
    ap.add_argument("--truth-cuts", default=TRUTH_CUTS, metavar="Q,...",
                    help=f"charge [e] a group must leave in a reticle to count in n_true (default {TRUTH_CUTS})")
    ap.add_argument("--count-radii-mm", default=COUNT_RADII_MM, metavar="R,...",
                    help=f"radii around the axis within which reticles are counted (default {COUNT_RADII_MM})")
    args = ap.parse_args(argv)
    if args.overlay < 1:
        ap.error(f"--overlay must be at least 1, got {args.overlay}")
    if args.size_x < 1 or args.size_y < 1:
        ap.error("--size-x and --size-y must be positive")
    cuts = _floats(args.truth_cuts, "--truth-cuts")
    radii = _floats(args.count_radii_mm, "--count-radii-mm")
    if not radii[0] > 0:
        ap.error("--count-radii-mm must be positive")

    D = load_inputs(args.charges)
    G, P, info = merge(D)
    px, py = float(D[0]["pitch_x_um"]), float(D[0]["pitch_y_um"])
    if info["n_events_total"] // args.overlay == 0:
        sys.exit(f"reticles: {info['n_events_total']} events make no super-event of {args.overlay}")
    ret, part, plane, tot = build(G, P, info, args.overlay, args.size_x, args.size_y, cuts, radii,
                                  px, py, args.offset_x, args.offset_y)
    d_total, d_ret, worst = check(ret, part, tot["q_in"])
    if not (abs(d_total) <= CHECK_REL and d_ret <= CHECK_REL):
        sys.exit(f"reticles: the reticles hold {d_total:+.3e} relative more charge than the input "
                 f"pixel rows of the events used ({tot['q_in']:.6g} e), and the part_q of reticle "
                 f"{worst} differ from its img by {d_ret:.3e} of its charge; nothing written")

    # inputs: per-batch keys and differing scalars per input, the rest as it is
    scalars, per_input = {}, {}
    for k in sorted(set.intersection(*(set(d) for d in D))):
        v0 = np.asarray(D[0][k])
        if k in BATCH_KEYS:
            per_input["input_" + k] = np.array([d[k] for d in D])
        elif v0.ndim == 0 or k in SAME_KEYS:
            if all(_same(d[k], v0) for d in D[1:]):
                scalars[k] = v0
            elif v0.ndim == 0:
                per_input["input_" + k] = np.array([d[k] for d in D])
    n_within = plane["plane_n_within"].sum(axis=0)
    n_hit_within = plane["plane_n_hit_within"].sum(axis=0)
    n_dropped = info["n_events_total"] - tot["n_super"] * args.overlay
    arrays = dict(
        **ret, **part, **plane,
        overlay=np.int64(args.overlay), size_x=np.int64(args.size_x), size_y=np.int64(args.size_y),
        offset_x=np.int64(args.offset_x), offset_y=np.int64(args.offset_y),
        truth_cuts=cuts, count_radii_mm=radii, n_within=n_within, n_empty_within=n_within - n_hit_within,
        n_planes=np.int64(len(plane["plane_q"])), n_reticles=np.int64(len(ret["img"])),
        n_parts=np.int64(len(part["part_q"])), n_events=np.int64(info["n_events_total"]),
        n_super_events=np.int64(tot["n_super"]), n_events_dropped=np.int64(n_dropped),
        axis_module=np.int64(axis_module()), gun_origin_mm=np.array(GUN_ORIGIN_MM),
        gun_direction=np.array(GUN_DIRECTION), total_q=np.float64(tot["q_in"]),
        inputs=np.array([os.path.abspath(p) for p in args.charges]),
        input_event_offset=info["event_offset"], input_n_events=info["n_events"],
        input_group_offset=info["group_offset"], **per_input, **scalars)
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".partial"
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    # summary
    lo, hi = AXIS_CHECK_LAYERS
    near = ((plane["plane_module"] == axis_module()) & (plane["plane_layer"] >= lo)
            & (plane["plane_layer"] <= hi) & (plane["plane_q"] != 0))
    du = plane["plane_u_centroid_mm"][near] - plane["plane_u_axis_mm"][near]
    dv = plane["plane_v_centroid_mm"][near] - plane["plane_v_axis_mm"][near]
    wq = plane["plane_q"][near]

    def offset(x):
        if not len(x):
            return "none"
        return (f"mean {x.mean():+.3f} (rms {x.std():.3f}), median {np.median(x):+.3f}, "
                f"charge-weighted {np.average(x, weights=wq):+.3f}")
    n_charged = int(np.count_nonzero(plane["plane_q"] != 0))
    lines = [
        f"reticles: {len(args.charges)} input(s), {info['n_events_total']} events; overlay "
        f"{args.overlay}: {tot['n_super']} super-events, {n_dropped} events dropped",
        f"  reticle {args.size_x} x {args.size_y} pixels ({args.size_x * px * 1e-3:g} x "
        f"{args.size_y * py * 1e-3:g} mm, v x u), grid offset ({args.offset_x}, {args.offset_y}) "
        f"pixels; axis module {axis_module()}",
        f"  planes: {len(plane['plane_q'])}, {n_charged} holding charge; reticles holding charge: "
        f"{len(ret['img'])}; (reticle, group) rows: {len(part['part_q'])}; pixel rows used: {tot['n_rows']}",
        f"  charge: img {tot['q_in'] * (1 + d_total):.6g} e, input {tot['q_in']:.6g} e, "
        f"relative difference {d_total:+.2e}; worst reticle part_q vs img {d_ret:.2e}",
        f"  plane centroid - axis point [mm], axis-module planes with charge in layers {lo}..{hi} "
        f"({len(du)}): u {offset(du)}; v {offset(dv)}",
        "  n_true over the reticles holding charge:",
    ]
    inner = ret["r_mm"] < radii[0]
    for j, c in enumerate(cuts):
        lines.append(f"    > {c:g} e: all {_dist(ret['n_true'][:, j])}; r < {radii[0]:g} mm "
                     f"{_dist(ret['n_true'][inner, j])}")
    for j, x in enumerate(radii):
        lines.append(f"  r < {x:g} mm: {int(n_within[j])} reticles on the axis planes, "
                     f"{int(n_hit_within[j])} holding charge (occupancy "
                     f"{n_hit_within[j] / max(int(n_within[j]), 1):.3f}), {int(n_within[j] - n_hit_within[j])} empty")
    lines.append(f"  written: {out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
