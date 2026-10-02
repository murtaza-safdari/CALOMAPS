#!/usr/bin/env python3
"""Reticles -> MIP counts: the number of particles in a reticle estimated from its binary hit map.

    python analysis/pixelav/mip_counting.py <reticles.npz> [more reticles.npz ...] --out-dir <dir>
                                            [--threshold 1000] [--noise 100] [--seed 1]
                                            [--folds 5] [--extrap-max-k 2] [--skip-extrapolation]
                                            [--truths n_frac,n_own,n_in] [--min-count 20]
                                            [--gbt-iter 300] [--gbt-max-rows 400000]
                                            [--count-bits 8] [--no-scan] [--no-figures]
                                            [--include-preds <preds.npz> ...]
    python analysis/pixelav/mip_counting.py --self-test

INPUTS: reticles.npz files of analysis/pixelav/reticles.py cut from the same events (the same
inputs, event offsets and event count, checked) with the same reticle size and truth objects,
typically one per --overlay K. The reticle shape comes from the files. Each file is evaluated on
its own; the estimators are trained on all of them.

READOUT: hit = img + noise > threshold, with noise Gaussian of sigma --noise electrons, independent
per pixel, drawn from numpy default_rng((seed, file index)) with the files sorted by overlay. Only
the reticles that hold charge are stored, so every number here is over those; a reticle without
charge has no hit unless noise makes one, and the noise-only rate (hits and estimated particles in
a pure-noise reticle) is given separately in the threshold scan.

TRUTH: a truth object is one part_id of reticles.py (a crossing, a connected island of a particle's
pixels on the plane, or with --truth particles a whole particle with its delta rays); a file
without part_id is read as particles (part_id = part_group, part_q_total = the sum of its part_q).
The charge cut equals the threshold and is strict everywhere: charge > cut. n_in counts the objects
whose charge inside the reticle exceeds the cut (part_q > cut; the n_true of reticles.py at that
cut, checked when the cut is one of its truth_cuts). n_own counts every object whose charge on the
plane exceeds the cut (part_q_total > cut) once, in the reticle holding the largest part of it
(equal parts: the first reticle row). n_frac gives every such object to each reticle in proportion
to its charge there, part_q / part_q_total clipped to [0, 1]: it changes smoothly when an object
straddles a reticle edge, and on each plane the objects whose rows are all present (their part_q
sum to part_q_total) sum to their number (checked). Over all reticles n_own and n_frac sum to the
objects above the cut; n_in counts an object crossing an edge in each reticle where its part passes
the cut. n_frac is the primary target; per-bin metrics bin it by rounding to an integer.

CLUSTERS: 8-connected groups of hit pixels of one reticle. Per cluster: size (pixels), width and
height of its bounding box (pixels along v = columns and along u = rows), the second moments of
its area in um (each pixel a uniform pitch_x x pitch_y rectangle), the principal-axis angle from
the v axis (degrees, -90..90), elongation sqrt(major/minor eigenvalue), fill = size / (width *
height), edge (touches the reticle border). Cluster truth t_<truth> (strict): the summed weight of
the part_* rows whose largest pixel (part_iv, part_iu) lies in the cluster, the weight being 1 for
an object n_in or n_own counts there and its charge fraction for n_frac. A counted row whose
largest pixel is not hit belongs to no cluster; its weight is reported, and for the per-cluster
training target a_<truth> (assigned) it is added to the cluster of the nearest hit pixel of its
reticle (none in a reticle without hits).

ESTIMATORS of the reticle count, all fitted on training reticles only; an empty hit map gives 0
for every binary-readout estimator:
  hits             E[n | n_hits], one table entry per hit count
  clusters         E[n | n_clusters]
  hits_clusters    E[n | n_hits, n_clusters], each cell shrunk toward the hits table with weight
                   --min-count: (sum + m * hits(n_hits)) / (count + m)
  cluster_shape    per cluster E[objects | size, width, height] shrunk toward E[objects | size]
                   with weight m, summed over the clusters of the reticle
  cluster_shape_edge  the same with the edge flag in the key
  gbt              histogram gradient-boosted trees (squared error) on the reticle summary
                   features FEATURES and the hits per row and per column; at most --gbt-max-rows
                   training reticles (a seeded random subset beyond that)
  gbt_layer        the same with the reticle's layer as one more feature (a sensor knows its layer,
                   not where the shower axis lies)
  analog           reference, not available to a binary readout: E[n | noise-free charge sum],
                   in training quantile bins, interpolated
A 1-D table entry with fewer than m training reticles pools the neighbouring entries symmetrically
until it holds m; above the largest trained value an entry scales the last one in proportion.

FOLDS: the events are split into --folds contiguous blocks of the underlying event index with
boundaries on multiples of the least common multiple of the overlays, so that no super-event of
any file is split (checked); the same event lies in the same fold in every file. For each fold the
estimators are trained on the reticles of all files whose events lie in the other folds and
predict that fold's reticles of every file (checked: no event in both), so every reticle gets one
out-of-fold prediction. EXTRAPOLATION (unless --skip-extrapolation): the same with training on
the files of overlay <= --extrap-max-k only, evaluated on the files of larger overlay.

METRICS per file and truth: bias and rms of estimate - truth in TRUTH_BINS and in R_BINS of r_mm;
per super-event totals over all its stored reticles: sum(estimate)/sum(truth) (mean and spread),
the relative spread std/mean of sum(estimate) beside those of sum(truth), sum(n_hits) and
sum(charge); data volume per reticle holding hits, from the file's reticle shape: bitmap (one bit
per pixel), zero-suppressed addresses (ceil(log2(pixels)) bits per hit), count (--count-bits, with
the share of reticles whose n_in does not fit).

OUTPUT in --out-dir: results.json; preds.npz (per file tag: the truths, super_evt, layer, r_mm,
q_sum, n_hits, n_clusters, fold, and pred__<truth>__<estimator> / xpred__<truth>__<estimator> for
the out-of-fold and extrapolation predictions, with the readout settings; the layout read by
--include-preds and analysis/pixelav/mip_counting_cnn.py); clusters.npz (the cluster table of the
main readout); fig_*.png. A printed summary. --include-preds adds the predictions of other
preds.npz files of the same inputs and readout (checked) to the metrics and figures.
"""
import argparse, json, math, os, sys
import numpy as np
from scipy import ndimage

THRESHOLD = 1000.0          # electrons
NOISE = 100.0               # electrons, Gaussian sigma per pixel
SEED = 1
N_FOLDS = 5
EXTRAP_MAX_K = 2
MIN_COUNT = 20              # training entries a table cell needs before it stands alone
GBT_ITER = 300
GBT_MAX_ROWS = 400000       # training reticles per gradient-boosted fit
COUNT_BITS = 8
SCAN_THRESHOLDS = (500.0, 1000.0, 2000.0, 4000.0)
SCAN_NOISES = (0.0, 100.0, 200.0)
SCAN_ESTIMATORS = ("hits", "clusters", "hits_clusters", "cluster_shape", "analog")
N_NOISE_MAPS = 20000        # pure-noise reticles for the noise-only rate
TRUTHS = ("n_frac", "n_own", "n_in")
COMPLETE_REL, COMPLETE_ABS_E = 1e-5, 1.0   # an object's part_q sum to part_q_total within this
FRAC_TOL = 1e-4             # per object, in sums of n_frac
TRUTH_BINS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20), (21, 40),
              (41, 80), (81, None))
R_BINS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf)   # mm; r_mm NaN (off-axis module) apart
SIZE_BINS = (1, 2, 3, 4, 5, 7, 11, 21)    # lower edges of the cluster-size histogram features
ESTIMATORS = ("hits", "clusters", "hits_clusters", "cluster_shape", "cluster_shape_edge", "gbt",
              "gbt_layer", "analog")
CHUNK = 65536               # reticles per block in the readout and the clustering
FIG_DPI = 120
# figure colour per estimator, fixed whatever subset is drawn; _tot2 and _analog variants of a
# network take its colour with a dash-dot and a dotted line, gbt_layer that of gbt dashed
COLOURS = {"hits": "#2a78d6", "clusters": "#eb6834", "hits_clusters": "#1baf7a",
           "cluster_shape": "#eda100", "gbt": "#e87ba4", "gbt_layer": "#e87ba4", "cnn_tiny": "#008300",
           "cnn_small": "#4a3aa7", "cluster_shape_edge": "#e34948", "analog": "#000000"}
FIG_ESTIMATORS = ("hits", "clusters", "hits_clusters", "cluster_shape", "gbt", "gbt_layer",
                  "cnn_tiny", "cnn_small", "analog")
_NEED = ("img", "super_evt", "module", "layer", "r_mm", "q_sum", "n_true", "truth_cuts",
         "part_reticle", "part_group", "part_q", "part_iv", "part_iu", "part_evt", "overlay",
         "size_x", "size_y", "offset_x", "offset_y", "pitch_x_um", "pitch_y_um", "n_events",
         "n_super_events", "inputs", "input_event_offset", "input_n_events", "n_empty_within",
         "count_radii_mm")
_OPTIONAL = ("part_id", "part_island", "part_q_total", "truth")
_SAME = ("inputs", "input_event_offset", "input_n_events", "n_events", "size_x", "size_y",
         "pitch_x_um", "pitch_y_um", "truth")


class Inconsistent(RuntimeError):
    """A hard check failed."""


def _require(ok, msg):
    if not ok:
        raise Inconsistent(msg)


# ----------------------------------------------------------------------------------------------
# inputs, truth, readout, clusters

def load_reticles(path):
    """The arrays of a reticles.npz that this analysis uses, as a dict. A file without part_id
    (written when every truth object was a group) is read with part_id = part_group, part_q_total =
    the sum of part_q over the rows of that id, part_island = 0 and truth "particles"."""
    try:
        with np.load(path, allow_pickle=False) as d:
            missing = [k for k in _NEED if k not in d.files]
            if missing:
                raise SystemExit(f"mip_counting: {path} lacks {', '.join(missing)}")
            R = {k: d[k] for k in _NEED}
            R.update({k: d[k] for k in _OPTIONAL if k in d.files})
    except (FileNotFoundError, OSError, ValueError) as err:
        raise SystemExit(f"mip_counting: cannot read {path}: {err}") from None
    R["path"] = os.path.abspath(path)
    for k in ("overlay", "size_x", "size_y", "offset_x", "offset_y", "n_events", "n_super_events"):
        R[k] = int(R[k])
    return with_truth_objects(R)


def with_truth_objects(R):
    """R with part_id, part_q_total (float64), part_island and truth; without part_id they are
    those of a file whose truth objects are the groups (load_reticles)."""
    if "part_id" not in R:
        R["part_id"] = R["part_group"].astype(np.int64)
        u, inv = np.unique(R["part_id"], return_inverse=True)
        inv = inv.ravel()
        R["part_q_total"] = np.bincount(inv, R["part_q"].astype(np.float64), len(u))[inv]
        R["part_island"] = np.zeros(len(R["part_id"]), np.int32)
        R["truth"] = "particles"
    else:
        missing = [k for k in ("part_q_total", "truth") if k not in R]
        if missing:
            raise SystemExit(f"mip_counting: {R['path']} has part_id but lacks {', '.join(missing)}")
        R["truth"] = str(R["truth"])
    R["part_q_total"] = R["part_q_total"].astype(np.float64)
    return R


def file_tags(raws):
    """K<overlay>, with _o<ox>_<oy> when two files share an overlay."""
    ks = [R["overlay"] for R in raws]
    tags = [f"K{R['overlay']}" + (f"_o{R['offset_x']}_{R['offset_y']}" if ks.count(R["overlay"]) > 1 else "")
            for R in raws]
    _require(len(set(tags)) == len(tags), f"two inputs with the same overlay and offsets: {tags}")
    return tags


def check_same_events(raws):
    """Every file cut from the same events with the same reticle size, pitch and truth objects."""
    r0 = raws[0]
    for R in raws[1:]:
        for k in _SAME:
            a, b = np.asarray(r0[k]), np.asarray(R[k])
            _require(a.shape == b.shape and np.array_equal(a, b),
                     f"{R['path']} and {r0['path']} differ in {k}: not cut from the same events")


def truths(R, cut):
    """(T, W, info): the reticle truths n_in, n_own, n_frac of the module docstring at the charge
    cut (int64, int64, float64), the weight of every part_* row in each (1 for a counted row, 0
    otherwise; the charge fraction for n_frac), and object counts."""
    pr, pid = R["part_reticle"], R["part_id"]
    pq, qt = R["part_q"].astype(np.float64), R["part_q_total"]
    n = len(R["img"])
    uid, inv = np.unique(pid, return_inverse=True)
    inv = inv.ravel()
    n_obj = len(uid)
    qt_obj = np.zeros(n_obj)
    qt_obj[inv] = qt
    _require(np.array_equal(qt, qt_obj[inv]), "rows of one truth object disagree on part_q_total")
    above = qt_obj > cut
    w_in = (pq > cut).astype(np.float64)
    order = np.lexsort((pr, -pq, inv))           # per object: largest part first, then first row
    first = order[np.r_[True, np.diff(inv[order]) != 0]] if len(pq) else order
    w_own = np.zeros(len(pq))
    w_own[first[above[inv[first]]]] = 1.0
    a_row = above[inv]
    w_frac = np.zeros(len(pq))
    w_frac[a_row] = np.clip(pq[a_row] / qt[a_row], 0.0, 1.0)
    T = {"n_in": np.bincount(pr, w_in, n).round().astype(np.int64),
         "n_own": np.bincount(pr, w_own, n).round().astype(np.int64),
         "n_frac": np.bincount(pr, w_frac, n)}
    _require(int(T["n_own"].sum()) == int(np.count_nonzero(above)),
             "n_own does not sum to the truth objects above the cut")
    _require(len(np.unique(pid[w_own > 0])) == int(np.count_nonzero(w_own)), "an object owned twice")
    # n_frac on each plane: the objects above the cut whose rows are all present sum to their number
    pk = (R["super_evt"].astype(np.int64) * 64 + R["module"]) * 256 + R["layer"]
    upl, plane_ret = np.unique(pk, return_inverse=True)
    plane_row = plane_ret.ravel()[pr]
    plane_obj = np.zeros(n_obj, np.int64)
    plane_obj[inv] = plane_row
    _require(np.array_equal(plane_obj[inv], plane_row), "a truth object on two planes")
    qsum = np.bincount(inv, pq, n_obj)
    complete = np.abs(qsum - qt_obj) <= COMPLETE_REL * np.abs(qt_obj) + COMPLETE_ABS_E
    ok = (complete & above)[inv]
    s = np.bincount(plane_row[ok], w_frac[ok], len(upl))
    c = np.bincount(plane_obj[complete & above], minlength=len(upl))
    _require(np.all(np.abs(s - c) <= FRAC_TOL * c + FRAC_TOL),
             "n_frac of the complete objects does not sum to their number on a plane")
    cuts = [float(x) for x in R["truth_cuts"]]
    if float(cut) in cuts:
        j = cuts.index(float(cut))
        _require(np.array_equal(T["n_in"], R["n_true"][:, j].astype(np.int64)),
                 f"n_in differs from n_true at {cut:g} e in {R['path']}")
    info = {"objects": int(n_obj), "above_cut": int(above.sum()),
            "above_cut_incomplete": int((above & ~complete).sum())}
    return T, {"n_in": w_in, "n_own": w_own, "n_frac": w_frac}, info


def readout(img, threshold, noise, rng, encode=()):
    """(hits, enc): binary hit maps img + Gaussian noise > threshold, and the encodings of the noisy
    image named in encode: tot2 (uint8 0..3, the number of the levels 1, 4 and 16 times the
    threshold it exceeds) and analog (float16 log1p(charge / threshold) above the threshold, else 0)."""
    hits = np.empty(img.shape, bool)
    enc = {e: np.empty(img.shape, np.uint8 if e == "tot2" else np.float16) for e in encode}
    for a in range(0, len(img), CHUNK):
        b = img[a:a + CHUNK]
        if noise > 0:
            b = b + rng.standard_normal(b.shape, dtype=np.float32) * np.float32(noise)
        hits[a:a + CHUNK] = b > threshold
        if "tot2" in enc:
            enc["tot2"][a:a + CHUNK] = ((b > threshold).astype(np.uint8) + (b > 4 * threshold)
                                        + (b > 16 * threshold))
        if "analog" in enc:
            enc["analog"][a:a + CHUNK] = np.where(b > threshold, np.log1p(np.maximum(b, 0) / threshold),
                                                  0.0).astype(np.float16)
    return hits, enc

def clusters(hits, pitch_x, pitch_y):
    """(pix, cl): the hit pixels (flat index into hits, increasing; cluster id) and the cluster
    table of the module docstring. Cluster ids run in reticle order."""
    n, ny, nx = hits.shape
    S = ny * nx
    st = np.zeros((3, 3, 3), bool)
    st[1] = True                              # 8-connected within a reticle, never across
    idx_all, lab_all, n_cl = [], [], 0
    for a in range(0, n, CHUNK):
        lab, k = ndimage.label(hits[a:a + CHUNK], structure=st)
        flat = lab.ravel()
        idx = np.flatnonzero(flat)
        idx_all.append(idx + a * S)
        lab_all.append(flat[idx].astype(np.int64) - 1 + n_cl)
        n_cl += k
    idx = np.concatenate(idx_all) if idx_all else np.zeros(0, np.int64)
    lab = np.concatenate(lab_all) if lab_all else np.zeros(0, np.int64)
    ret = idx // S
    iy = idx % S // nx
    ix = idx % nx
    size = np.bincount(lab, minlength=n_cl)
    cl_ret = np.zeros(n_cl, np.int64)
    cl_ret[lab] = ret
    _require(np.all(np.diff(cl_ret) >= 0), "cluster ids out of reticle order")
    o = np.argsort(lab, kind="stable")
    starts = np.r_[0, np.cumsum(size)[:-1]] if n_cl else np.zeros(0, np.int64)
    x0 = np.minimum.reduceat(ix[o], starts) if n_cl else np.zeros(0, np.int64)
    x1 = np.maximum.reduceat(ix[o], starts) if n_cl else np.zeros(0, np.int64)
    y0 = np.minimum.reduceat(iy[o], starts) if n_cl else np.zeros(0, np.int64)
    y1 = np.maximum.reduceat(iy[o], starts) if n_cl else np.zeros(0, np.int64)
    x = (ix + 0.5) * pitch_x
    y = (iy + 0.5) * pitch_y
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = np.bincount(lab, x, n_cl) / size
        my = np.bincount(lab, y, n_cl) / size
        sxx = np.bincount(lab, x * x, n_cl) / size - mx * mx + pitch_x ** 2 / 12
        syy = np.bincount(lab, y * y, n_cl) / size - my * my + pitch_y ** 2 / 12
        sxy = np.bincount(lab, x * y, n_cl) / size - mx * my
    half = 0.5 * (sxx + syy)
    disc = np.sqrt(np.maximum(0.25 * (sxx - syy) ** 2 + sxy ** 2, 0.0))
    lmaj, lmin = half + disc, np.maximum(half - disc, 1e-12)
    edge_pix = (ix == 0) | (ix == nx - 1) | (iy == 0) | (iy == ny - 1)
    width, height = x1 - x0 + 1, y1 - y0 + 1
    cl = {"reticle": cl_ret, "size": size, "width": width, "height": height,
          "x0": x0, "y0": y0, "mean_x_um": mx, "mean_y_um": my,
          "sxx_um2": sxx, "syy_um2": syy, "sxy_um2": sxy,
          "theta_deg": np.degrees(0.5 * np.arctan2(2 * sxy, sxx - syy)),
          "elongation": np.sqrt(lmaj / lmin), "fill": size / (width * height),
          "edge": np.bincount(lab[edge_pix], minlength=n_cl) > 0}
    return {"idx": idx, "cl": lab, "ret": ret, "iy": iy, "ix": ix, "edge": edge_pix}, cl


def cluster_truth(pix, cl, R, w, n, ny, nx, pitch_x, pitch_y):
    """(strict, assigned, lost, unclustered): per-cluster sums of the row weights w (the rows with
    w > 0), strict = the rows whose largest pixel lies in the cluster, assigned = strict plus the
    unclustered rows given to the cluster of the nearest hit pixel of their reticle; lost = the
    unclustered weight in reticles without hits, per reticle; the summed unclustered weight."""
    S = ny * nx
    n_cl = len(cl["size"])
    sel = np.flatnonzero(w > 0)
    ww = w[sel]
    pr = R["part_reticle"][sel]
    py_ = R["part_iu"][sel].astype(np.int64)
    px_ = R["part_iv"][sel].astype(np.int64)
    flat = pr * S + py_ * nx + px_
    if len(pix["idx"]):
        k = np.minimum(np.searchsorted(pix["idx"], flat), len(pix["idx"]) - 1)
        found = pix["idx"][k] == flat
    else:
        k, found = np.zeros(len(flat), np.int64), np.zeros(len(flat), bool)
    strict = np.bincount(pix["cl"][k[found]], ww[found], n_cl)
    assigned = strict.copy()
    lost = np.zeros(n)
    un = np.flatnonzero(~found)
    if len(un):
        r_un = pr[un]
        lo = np.searchsorted(pix["ret"], r_un, side="left")
        hi = np.searchsorted(pix["ret"], r_un, side="right")
        cnt = hi - lo
        np.add.at(lost, r_un[cnt == 0], ww[un][cnt == 0])
        has = cnt > 0
        if has.any():
            c, l0 = cnt[has], lo[has]
            owner = np.repeat(np.arange(len(c)), c)
            pos = np.repeat(l0 - np.r_[0, np.cumsum(c)[:-1]], c) + np.arange(int(c.sum()))
            uy = py_[un[has]].astype(np.float64)
            ux = px_[un[has]].astype(np.float64)
            d2 = (((pix["ix"][pos] - ux[owner]) * pitch_x) ** 2
                  + ((pix["iy"][pos] - uy[owner]) * pitch_y) ** 2)
            o = np.lexsort((d2, owner))
            best = o[np.r_[True, np.diff(owner[o]) != 0]]
            np.add.at(assigned, pix["cl"][pos[best]], ww[un[has]])
    return strict, assigned, lost, float(ww[un].sum())

FEATURES = (["n_hits", "n_clusters"] + [f"size_ge{b}" for b in SIZE_BINS]
            + ["edge_hits", "edge_clusters", "max_size", "rows_hit", "cols_hit"])


def reticle_features(pix, cl, n, ny, nx):
    """[n, len(FEATURES) + ny + nx] float32: FEATURES, then hits per row (u) and per column (v)."""
    n_hits = np.bincount(pix["ret"], minlength=n)
    n_cl = np.bincount(cl["reticle"], minlength=n)
    sb = np.searchsorted(np.array(SIZE_BINS), cl["size"], side="right") - 1
    hist = np.bincount(cl["reticle"] * len(SIZE_BINS) + sb, minlength=n * len(SIZE_BINS)).reshape(n, -1)
    edge_hits = np.bincount(pix["ret"][pix["edge"]], minlength=n)
    edge_cl = np.bincount(cl["reticle"][cl["edge"]], minlength=n)
    max_size = np.zeros(n, np.int64)
    if len(cl["size"]):
        np.maximum.at(max_size, cl["reticle"], cl["size"])
    rows = np.bincount(pix["ret"] * ny + pix["iy"], minlength=n * ny).reshape(n, ny)
    cols = np.bincount(pix["ret"] * nx + pix["ix"], minlength=n * nx).reshape(n, nx)
    F = np.column_stack([n_hits, n_cl, hist, edge_hits, edge_cl, max_size,
                         (rows > 0).sum(1), (cols > 0).sum(1), rows, cols]).astype(np.float32)
    return F


def event_folds(n_events, overlays, n_folds):
    """fold of each underlying event: contiguous blocks with boundaries on multiples of the least
    common multiple of the overlays."""
    L = 1
    for k in overlays:
        L = L * k // math.gcd(L, k)
    n_blocks = n_events // L
    if n_blocks < n_folds:
        raise Inconsistent(f"{n_events} events make {n_blocks} blocks of {L} (the overlays' least "
                           f"common multiple), fewer than {n_folds} folds")
    bounds = np.array([round(f * n_blocks / n_folds) * L for f in range(n_folds + 1)])
    bounds[-1] = n_events
    fold = np.searchsorted(bounds[1:], np.arange(n_events), side="right")
    _require(np.all(np.bincount(fold, minlength=n_folds) > 0), "an empty fold")
    return fold


def super_event_fold(ev_fold, n_super, K):
    """fold of each super-event of overlay K; Inconsistent when one is split between folds."""
    ev = np.arange(n_super)[:, None] * K + np.arange(K)[None, :]
    f = ev_fold[ev]
    _require(np.all(f == f[:, :1]), f"a fold boundary splits a super-event of overlay {K}")
    return f[:, 0]


def prepare(raws, tags, threshold, noise, seed, n_folds, keep_hits=False, encode=()):
    """Per file: the readout, truths, cluster table and features of the module docstring, with the
    fold of each reticle; with keep_hits the hit maps, with encode those encodings of the noisy
    image (readout)."""
    check_same_events(raws)
    ev_fold = event_folds(raws[0]["n_events"], [R["overlay"] for R in raws], n_folds)
    out = []
    for i, (R, tag) in enumerate(zip(raws, tags)):
        K = R["overlay"]
        _require(np.array_equal(R["part_evt"] // K, R["super_evt"][R["part_reticle"]]),
                 f"{R['path']}: part_evt // overlay differs from the reticle's super_evt")
        img = R["img"]
        n, ny, nx = img.shape
        _require(nx < 64 and ny < 64, "reticles of 64 pixels or more on a side are not supported")
        px, py = float(R["pitch_x_um"]), float(R["pitch_y_um"])
        rng = np.random.default_rng((seed, i))
        hits, enc = readout(img, threshold, noise, rng, encode)
        T, W, info = truths(R, threshold)
        pix, cl = clusters(hits, px, py)
        F = reticle_features(pix, cl, n, ny, nx)
        D = {"tag": tag, "path": R["path"], "overlay": K, "n": n, "ny": ny, "nx": nx,
             "pitch_x": px, "pitch_y": py, "truth_objects": R["truth"], "truth_info": info,
             "super_evt": R["super_evt"].astype(np.int64), "module": R["module"],
             "layer": R["layer"].astype(np.int64), "r_mm": R["r_mm"], "q_sum": R["q_sum"],
             "n_super_events": R["n_super_events"], **T,
             "n_hits": F[:, 0].astype(np.int64), "n_clusters": F[:, 1].astype(np.int64),
             "feat": F, "pix": pix, "cl": cl, "unclustered": {}, "lost": {},
             "n_empty_within": R["n_empty_within"], "count_radii_mm": R["count_radii_mm"]}
        for t in TRUTHS:
            strict, assigned, lost, w_un = cluster_truth(pix, cl, R, W[t], n, ny, nx, px, py)
            cl[f"t_{t}"], cl[f"a_{t}"] = strict, assigned
            un_ret = D[t] - np.bincount(cl["reticle"], strict, n)
            _require(np.all(un_ret >= -FRAC_TOL) and abs(un_ret.sum() - w_un) <= FRAC_TOL * max(1.0, w_un),
                     f"{tag} {t}: cluster truths and unclustered objects do not add up")
            _require(np.allclose(np.bincount(cl["reticle"], assigned, n) + lost, D[t], rtol=0, atol=FRAC_TOL),
                     f"{tag} {t}: assigned cluster truths do not add up")
            D["unclustered"][t], D["lost"][t] = w_un, lost
        D["fold"] = super_event_fold(ev_fold, R["n_super_events"], K)[D["super_evt"]]
        D["events"] = (D["super_evt"][:, None] * K + np.arange(K)[None, :])
        if keep_hits:
            D["hits"] = hits
        D["enc"] = enc
        out.append(D)
    return out, ev_fold

# ----------------------------------------------------------------------------------------------
# estimators

class Lookup1D:
    """E[y | x] for integer x >= 0: one entry per value; an entry with fewer than m training rows
    pools its neighbours symmetrically; beyond the largest trained value, proportional."""

    def __init__(self, m):
        self.m = m

    def fit(self, x, y):
        x = np.asarray(x, np.int64)
        vmax = int(x.max()) if len(x) else 0
        cnt = np.bincount(x, minlength=vmax + 1).astype(np.float64)
        s = np.bincount(x, y, vmax + 1)
        C, Sm = np.r_[0.0, np.cumsum(cnt)], np.r_[0.0, np.cumsum(s)]
        t = np.zeros(vmax + 1)
        for v in range(vmax + 1):
            w = 0
            while True:
                lo, hi = max(0, v - w), min(vmax, v + w)
                c = C[hi + 1] - C[lo]
                if c >= self.m or (lo == 0 and hi == vmax):
                    break
                w += 1
            t[v] = (Sm[hi + 1] - Sm[lo]) / c if c else 0.0
        self.table, self.vmax = t, vmax
        return self

    def predict(self, x):
        x = np.asarray(x, np.int64)
        out = self.table[np.minimum(x, self.vmax)]
        big = x > self.vmax
        if self.vmax > 0:
            out[big] = self.table[self.vmax] * x[big] / self.vmax
        return out


class QuantileLookup:
    """E[y | x] for continuous x >= 0: means in training quantile bins, interpolated; below the first
    and above the last bin, proportional."""

    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit(self, x, y):
        edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)))
        b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
        c = np.bincount(b, minlength=len(edges) - 1)
        ok = c > 0
        self.xm = (np.bincount(b, x, len(c)) / np.maximum(c, 1))[ok]
        self.ym = (np.bincount(b, y, len(c)) / np.maximum(c, 1))[ok]
        return self

    def predict(self, x):
        out = np.interp(x, self.xm, self.ym)
        lo, hi = x < self.xm[0], x > self.xm[-1]
        out[lo] = self.ym[0] * x[lo] / self.xm[0]
        out[hi] = self.ym[-1] * x[hi] / self.xm[-1]
        return out


def _shrunk_table(keys, y, prior, m):
    """{key: (sum + m * prior) / (count + m)} as sorted keys and values, prior per row."""
    uk, inv = np.unique(keys, return_inverse=True)
    inv = inv.ravel()
    c = np.bincount(inv, minlength=len(uk))
    s = np.bincount(inv, y, len(uk))
    p = np.bincount(inv, prior, len(uk)) / c          # the prior is a function of the key
    return uk, (s + m * p) / (c + m)


def _lookup(uk, val, keys, fallback):
    k = np.searchsorted(uk, keys)
    k = np.minimum(k, max(len(uk) - 1, 0))
    hit = (uk[k] == keys) if len(uk) else np.zeros(len(keys), bool)
    return np.where(hit, val[k] if len(uk) else 0.0, fallback)


def view(Ds, masks):
    """The reticle and cluster arrays of the reticles selected by masks, concatenated."""
    parts, cls, off = [], [], 0
    for D, m in zip(Ds, masks):
        sel = np.flatnonzero(m)
        new = np.full(D["n"], -1, np.int64)
        new[sel] = np.arange(len(sel)) + off
        parts.append({"feat": D["feat"][sel], "q_sum": D["q_sum"][sel], "layer": D["layer"][sel],
                      **{t: D[t][sel] for t in TRUTHS}})
        keep = m[D["cl"]["reticle"]]
        c = {k: v[keep] for k, v in D["cl"].items()}
        c["reticle"] = new[c["reticle"]]
        cls.append(c)
        off += len(sel)
    V = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    V["cl"] = {k: np.concatenate([c[k] for c in cls]) for k in cls[0]}
    V["n"] = off
    V["n_hits"] = V["feat"][:, 0].astype(np.int64)
    V["n_clusters"] = V["feat"][:, 1].astype(np.int64)
    return V


def _shape_keys(cl, edge):
    k = (cl["size"].astype(np.int64) * 64 + cl["width"]) * 64 + cl["height"]
    return k * 2 + cl["edge"] if edge else k


class Estimators:
    """The estimators of the module docstring for one target, fitted on a view."""

    def __init__(self, names, m=MIN_COUNT, gbt_iter=GBT_ITER, seed=SEED, gbt_max_rows=GBT_MAX_ROWS):
        self.names, self.m, self.gbt_iter, self.seed = tuple(names), m, gbt_iter, seed
        self.gbt_max_rows = gbt_max_rows
        self.gbts = {}

    def fit(self, V, target):
        y = V[target].astype(np.float64)
        h, c = V["n_hits"], V["n_clusters"]
        self.hits = Lookup1D(self.m).fit(h, y)
        if "clusters" in self.names:
            self.clusters = Lookup1D(self.m).fit(c, y)
        if "hits_clusters" in self.names:
            self.cmax = int(c.max()) + 1
            self.hc = _shrunk_table(h * self.cmax + c, y, self.hits.predict(h), self.m)
        if "cluster_shape" in self.names or "cluster_shape_edge" in self.names:
            cy = V["cl"][f"a_{target}"].astype(np.float64)
            self.size = Lookup1D(self.m).fit(V["cl"]["size"], cy)
            prior = self.size.predict(V["cl"]["size"])
            self.shape = {e: _shrunk_table(_shape_keys(V["cl"], e), cy, prior, self.m)
                          for e in (False, True)}
        if "analog" in self.names:
            self.analog = QuantileLookup(max(20, min(400, len(y) // 200))).fit(V["q_sum"], y)
        for name in ("gbt", "gbt_layer"):
            if name not in self.names:
                continue
            from sklearn.ensemble import HistGradientBoostingRegressor
            rows = np.flatnonzero(h > 0)
            if len(rows) > self.gbt_max_rows:
                rows = np.sort(np.random.default_rng(self.seed).choice(rows, self.gbt_max_rows, replace=False))
            # squared error: each leaf is a mean of training counts, so a prediction stays within
            # the trained range; with the Poisson log link sklearn's leaf steps are not bounded and
            # sparse dense-core leaves reach thousands of particles
            model = HistGradientBoostingRegressor(
                loss="squared_error", max_iter=self.gbt_iter, learning_rate=0.1, max_leaf_nodes=31,
                min_samples_leaf=self.m, early_stopping=False, random_state=self.seed)
            model.fit(self._gbt_x(V, name, rows), y[rows])
            self.gbts[name] = model
        return self

    @staticmethod
    def _gbt_x(V, name, rows):
        return V["feat"][rows] if name == "gbt" else np.column_stack([V["feat"][rows], V["layer"][rows]])

    def predict(self, V):
        h, c = V["n_hits"], V["n_clusters"]
        empty = h == 0
        P = {}
        for name in self.names:
            if name == "hits":
                p = self.hits.predict(h)
            elif name == "clusters":
                p = self.clusters.predict(c)
            elif name == "hits_clusters":
                ok = c < self.cmax
                p = _lookup(*self.hc, h * self.cmax + np.minimum(c, self.cmax - 1), self.hits.predict(h))
                p = np.where(ok, p, self.hits.predict(h))
            elif name in ("cluster_shape", "cluster_shape_edge"):
                e = name == "cluster_shape_edge"
                pc = _lookup(*self.shape[e], _shape_keys(V["cl"], e), self.size.predict(V["cl"]["size"]))
                p = np.bincount(V["cl"]["reticle"], pc, V["n"])
            elif name in ("gbt", "gbt_layer"):
                p = np.zeros(V["n"])
                rows = np.flatnonzero(~empty)
                if len(rows):
                    p[rows] = np.maximum(self.gbts[name].predict(self._gbt_x(V, name, rows)), 0.0)
            elif name == "analog":
                p = self.analog.predict(V["q_sum"])
            else:
                raise ValueError(f"unknown estimator {name}")
            if name != "analog":
                p = np.where(empty, 0.0, p)
            P[name] = p
        return P


def crossfit(Ds, target, names, n_folds, train_tags, eval_tags, m=MIN_COUNT, gbt_iter=GBT_ITER,
             seed=SEED, n_events=None, log=None, gbt_max_rows=GBT_MAX_ROWS):
    """{tag: {name: out-of-fold prediction}} for the eval files, trained per fold on the other
    folds of the train files."""
    tr = [D for D in Ds if D["tag"] in train_tags]
    ev = [D for D in Ds if D["tag"] in eval_tags]
    P = {D["tag"]: {nm: np.full(D["n"], np.nan) for nm in names} for D in ev}
    for f in range(n_folds):
        V = view(tr, [D["fold"] != f for D in tr])
        if n_events is not None:          # no underlying event both in training and in this fold
            seen = np.zeros(n_events, bool)
            for D in tr:
                seen[D["events"][D["fold"] != f].ravel()] = True
            for D in ev:
                _require(not seen[D["events"][D["fold"] == f].ravel()].any(),
                         f"fold {f}: an event of {D['tag']} is also in the training set")
        est = Estimators(names, m, gbt_iter, seed, gbt_max_rows).fit(V, target)
        for D in ev:
            sel = D["fold"] == f
            pr = est.predict(view([D], [sel]))
            for nm in names:
                P[D["tag"]][nm][sel] = pr[nm]
        if log:
            log(f"    fold {f}: trained on {V['n']} reticles")
    for D in ev:
        for nm in names:
            _require(np.all(np.isfinite(P[D["tag"]][nm])), f"{D['tag']} {nm}: a reticle without a prediction")
    return P


# ----------------------------------------------------------------------------------------------
# metrics

def _bins_label(lo, hi):
    return f"{lo}+" if hi is None else (f"{lo}" if lo == hi else f"{lo}-{hi}")


def binned(est, truth, bins=TRUTH_BINS):
    """per truth bin (the truth rounded to an integer): N, mean truth, mean estimate, bias, rms,
    std of estimate - truth."""
    out = []
    tb = np.rint(truth)
    for lo, hi in bins:
        m = (tb >= lo) & ((tb <= hi) if hi is not None else True)
        d = est[m] - truth[m]
        N = int(m.sum())
        out.append({"bin": _bins_label(lo, hi), "N": N,
                    "truth": float(truth[m].mean()) if N else None,
                    "est": float(est[m].mean()) if N else None,
                    "bias": float(d.mean()) if N else None,
                    "rms": float(np.sqrt(np.mean(d * d))) if N else None,
                    "std": float(d.std()) if N else None})
    return out


def binned_r(est, truth, r):
    out = []
    for lo, hi in zip(R_BINS[:-1], R_BINS[1:]):
        m = (r >= lo) & (r < hi)
        out.append(_res(est[m], truth[m], f"{lo:g}-{hi:g}"))
    out.append(_res(est[~np.isfinite(r)], truth[~np.isfinite(r)], "off-axis"))
    return out


def _res(e, t, label):
    d = e - t
    N = len(d)
    return {"bin": label, "N": N, "truth": float(t.mean()) if N else None,
            "bias": float(d.mean()) if N else None, "rms": float(np.sqrt(np.mean(d * d))) if N else None}


def _spread(x):
    return float(x.std(ddof=1) / x.mean()) if len(x) > 1 and x.mean() else None


def event_totals(D, est, truth):
    """per super-event sums of estimate and truth: ratio mean and spread, relative spreads."""
    ns = D["n_super_events"]
    se = np.bincount(D["super_evt"], est, ns)
    st = np.bincount(D["super_evt"], truth, ns)
    _require(np.all(st > 0), f"{D['tag']}: a super-event without particles")
    ratio = se / st
    return {"n_super_events": ns, "ratio_mean": float(ratio.mean()),
            "ratio_std": float(ratio.std(ddof=1)) if ns > 1 else None,
            "spread_est": _spread(se), "spread_truth": _spread(st)}


def event_references(D):
    ns = D["n_super_events"]
    return {"spread_hits": _spread(np.bincount(D["super_evt"], D["n_hits"], ns)),
            "spread_charge": _spread(np.bincount(D["super_evt"], D["q_sum"], ns))}


def data_volume(D, count_bits=COUNT_BITS):
    """bits per reticle holding hits for the three formats, from the file's reticle shape, per n_own
    and per n_frac truth bin and per shower; the share of those reticles whose n_in exceeds the
    largest count that count_bits hold."""
    S = D["nx"] * D["ny"]
    abits = int(math.ceil(math.log2(S)))
    h = D["n_hits"]
    m = h > 0
    fmt = {"bitmap": np.full(len(h), S), "zero_suppressed": abits * h, "count": np.full(len(h), count_bits)}

    def per_bin(t):
        out = []
        for lo, hi in TRUTH_BINS:
            b = m & (t >= lo) & ((t <= hi) if hi is not None else True)
            out.append({"bin": _bins_label(lo, hi), "N": int(b.sum()), "mean_hits": float(h[b].mean()) if b.any() else None,
                        **{k: float(v[b].mean()) if b.any() else None for k, v in fmt.items()}})
        return out
    ns = D["n_super_events"]
    shower = {k: float(np.bincount(D["super_evt"][m], v[m], ns).mean()) for k, v in fmt.items()}
    return {"pixels": S, "address_bits": abits, "bitmap_bits": S, "count_bits": count_bits,
            "count_overflow_share": float(np.mean(D["n_in"][m] > 2 ** count_bits - 1)) if m.any() else 0.0,
            "crossover_hits": int(S // abits + 1), "reticles_with_hits_per_shower":
            float(np.bincount(D["super_evt"][m], minlength=ns).mean()),
            "mean_bits_per_reticle": {k: float(v[m].mean()) for k, v in fmt.items()},
            "bits_per_shower": shower, "per_truth_bin": per_bin(D["n_own"]),
            "per_n_frac_bin": per_bin(np.rint(D["n_frac"]))}


def evaluate(Ds, preds, xpreds=None):
    """The metrics of the module docstring, {tag: {...}}."""
    out = {}
    for D in Ds:
        tag = D["tag"]
        E = {"overlay": D["overlay"], "n_reticles": D["n"], "n_super_events": D["n_super_events"],
             "references": event_references(D), "truths": {}}
        for t in TRUTHS:
            if t not in preds.get(tag, {}):
                continue
            T = D[t].astype(np.float64)
            res = {"sum_truth": float(T.sum())}
            for nm, p in preds[tag][t].items():
                res[nm] = {"truth_bins": binned(p, T), "r_bins": binned_r(p, T, D["r_mm"]),
                           "event_totals": event_totals(D, p, T)}
            if xpreds and tag in xpreds and t in xpreds[tag]:
                res["extrapolation"] = {nm: {"truth_bins": binned(p, T), "event_totals": event_totals(D, p, T)}
                                        for nm, p in xpreds[tag][t].items()}
            E["truths"][t] = res
        out[tag] = E
    return out


# ----------------------------------------------------------------------------------------------
# predictions file

def save_preds(path, Ds, preds, xpreds, settings):
    arrays = {"tags": np.array([D["tag"] for D in Ds]), "files": np.array([D["path"] for D in Ds]),
              **{k: np.asarray(v) for k, v in settings.items()}}
    for D in Ds:
        g = D["tag"]
        for k in TRUTHS + ("super_evt", "layer", "r_mm", "q_sum", "n_hits", "n_clusters", "fold"):
            arrays[f"{g}__{k}"] = D[k]
        arrays[f"{g}__overlay"] = np.int64(D["overlay"])
        arrays[f"{g}__n_super_events"] = np.int64(D["n_super_events"])
        for kind, P in (("pred", preds), ("xpred", xpreds or {})):
            for t, by in P.get(g, {}).items():
                for nm, p in by.items():
                    arrays[f"{g}__{kind}__{t}__{nm}"] = p.astype(np.float32)
    tmp = path + ".partial"
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_preds(path, Ds, settings):
    """(preds, xpreds) of a preds.npz, after checking it holds the same files, readout and truths."""
    with np.load(path, allow_pickle=False) as d:
        A = {k: d[k] for k in d.files}
    for k, v in settings.items():
        _require(k in A and np.array_equal(np.asarray(A[k]), np.asarray(v)),
                 f"{path}: {k} differs from this run ({A.get(k)} vs {v})")
    preds, xpreds = {}, {}
    for D in Ds:
        g = D["tag"]
        for k in ("super_evt", "fold", "n_hits") + tuple(t for t in TRUTHS if f"{g}__{t}" in A):
            _require(f"{g}__{k}" in A and np.array_equal(A[f"{g}__{k}"], D[k]),
                     f"{path}: {g} {k} differs from this run")
        for key, v in A.items():
            parts = key.split("__")
            if len(parts) == 4 and parts[0] == g and parts[1] in ("pred", "xpred"):
                (preds if parts[1] == "pred" else xpreds).setdefault(g, {}).setdefault(
                    parts[2], {})[parts[3]] = v.astype(np.float64)
    return preds, xpreds


def merge_preds(a, b):
    for g, by in b.items():
        for t, P in by.items():
            for nm, p in P.items():
                _require(nm not in a.get(g, {}).get(t, {}), f"estimator {nm} given twice")
                a.setdefault(g, {}).setdefault(t, {})[nm] = p
    return a


# ----------------------------------------------------------------------------------------------
# report

def _cell(r):
    if r["N"] == 0 or r["rms"] is None:
        return "-"
    return f"{r['rms']:.2f}/{r['bias']:+.2f}"


def table(results, truth, kind="main"):
    """The printed table: per file and estimator, rms/bias per truth bin and the event totals."""
    lines = []
    for tag, E in results.items():
        if truth not in E["truths"]:
            continue
        T = E["truths"][truth]
        src = T if kind == "main" else T.get("extrapolation", {})
        names = [k for k in src if isinstance(src[k], dict) and "truth_bins" in src[k]]
        if not names:
            continue
        first = src[names[0]]["truth_bins"]
        head = f"  {tag} {truth}" + (" (trained on small overlays)" if kind != "main" else "")
        lines.append(f"{head}: {E['n_reticles']} reticles, {E['n_super_events']} super-events, "
                     f"sum truth {T['sum_truth']:.1f}")
        lines.append("    " + f"{'estimator':<19}" + "".join(f"{b['bin']:>14}" for b in first)
                     + f"{'sum ratio':>16}{'spread':>8}")
        lines.append("    " + f"{'reticles':<19}" + "".join(f"{b['N']:>14}" for b in first))
        for nm in names:
            r = src[nm]
            et = r["event_totals"]
            rs = f"{et['ratio_mean']:.3f}+-{et['ratio_std']:.3f}" if et["ratio_std"] is not None else f"{et['ratio_mean']:.3f}"
            sp = f"{et['spread_est']:.3f}" if et["spread_est"] is not None else "-"
            lines.append("    " + f"{nm:<19}" + "".join(f"{_cell(b):>14}" for b in r["truth_bins"])
                         + f"{rs:>16}{sp:>8}")
        et0 = src[names[0]]["event_totals"]
        ref = E["references"]
        if et0["spread_truth"] is not None:
            lines.append(f"    event-to-event spread (std/mean over {E['n_super_events']} super-events): "
                         f"truth {et0['spread_truth']:.3f}, hits {ref['spread_hits']:.3f}, "
                         f"charge {ref['spread_charge']:.3f}")
    return lines


def _colour(nm):
    for suffix in ("_tot2", "_analog"):
        if nm.endswith(suffix):
            nm = nm[:-len(suffix)]
    return COLOURS.get(nm, "#777777")


def _linestyle(nm):
    if nm == "analog" or nm == "gbt_layer":
        return "--"
    return "-." if nm.endswith("_tot2") else ":" if nm.endswith("_analog") else "-"


def _fine_bins(tmax):
    edges = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 40, 55, 75, 100, 140, 200, 300, 450, 700]
    return [e for e in edges if e <= tmax] + [tmax + 1]


def figures(Ds, preds, results, out_dir, truth="n_own", prefix="fig", names=None):
    """The figures of the module docstring; returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    have = [nm for nm in (names or FIG_ESTIMATORS) if all(nm in preds[D["tag"]][truth] for D in Ds)]
    written = []
    nK = len(Ds)
    label = {"n_own": "n_own (objects owned)", "n_in": "n_in (objects inside)",
             "n_frac": "n_frac (objects by charge share)"}[truth]

    def style(nm):
        return dict(color=_colour(nm), lw=2 if nm != "analog" else 1.5, ls=_linestyle(nm))

    # estimate vs truth
    fig, axs = plt.subplots(1, nK, figsize=(4.2 * nK, 4), squeeze=False)
    for ax, D in zip(axs[0], Ds):
        T = D[truth]
        e = _fine_bins(int(T.max()))
        b = np.searchsorted(e, T, side="right") - 1
        for nm in have:
            p = preds[D["tag"]][truth][nm]
            c = np.bincount(b, minlength=len(e) - 1)
            ok = c >= 5
            tm = (np.bincount(b, T, len(c)) / np.maximum(c, 1))[ok]
            pm = (np.bincount(b, p, len(c)) / np.maximum(c, 1))[ok]
            ps = np.sqrt(np.maximum((np.bincount(b, p * p, len(c)) / np.maximum(c, 1))[ok] - pm ** 2, 0))
            ax.plot(tm, pm, label=nm, **style(nm))
            if nm != "analog":
                ax.fill_between(tm, pm - ps, pm + ps, color=_colour(nm), alpha=0.12, lw=0)
        top = max(1.0, float(T.max()))
        ax.plot([0, top], [0, top], color="#999999", lw=1, ls=":")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlim(0, top * 1.1)
        ax.set_ylim(0, top * 1.3)
        ax.set_xlabel(f"true {label} per reticle")
        ax.set_ylabel("estimated particles per reticle (mean, band = std)")
        ax.set_title(f"overlay K = {D['overlay']} ({D['tag']})")
    axs[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_estimate_vs_truth_{truth}.png")
    fig.savefig(p, dpi=FIG_DPI)
    plt.close(fig)
    written.append(p)

    # relative bias and resolution vs truth
    fig, axs = plt.subplots(2, nK, figsize=(4.2 * nK, 6.5), squeeze=False, sharex=True)
    for j, D in enumerate(Ds):
        rb = results[D["tag"]]["truths"][truth]
        for nm in have:
            tb = [b for b in rb[nm]["truth_bins"] if b["N"] >= 5 and b["truth"]]
            x = [b["truth"] for b in tb]
            axs[0][j].plot(x, [b["bias"] / b["truth"] for b in tb], marker="o", ms=3, label=nm, **style(nm))
            axs[1][j].plot(x, [b["std"] / b["truth"] for b in tb], marker="o", ms=3, label=nm, **style(nm))
        axs[0][j].axhline(0, color="#999999", lw=1)
        axs[0][j].set_title(f"overlay K = {D['overlay']}")
        axs[0][j].set_ylabel("relative bias (estimate - truth) / truth")
        axs[1][j].set_ylabel("relative resolution std / truth")
        axs[1][j].set_xlabel(f"true {label} (mean in bin)")
        axs[1][j].set_xscale("log")
        axs[1][j].set_yscale("log")
    axs[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_relative_bias_resolution_{truth}.png")
    fig.savefig(p, dpi=FIG_DPI)
    plt.close(fig)
    written.append(p)

    # event totals vs K
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    Ks = [D["overlay"] for D in Ds]
    for nm in have:
        et = [results[D["tag"]]["truths"][truth][nm]["event_totals"] for D in Ds]
        err = [(e["ratio_std"] or 0) / math.sqrt(e["n_super_events"]) for e in et]
        axs[0].errorbar(Ks, [e["ratio_mean"] for e in et], yerr=err, marker="o", ms=4, capsize=2,
                        label=nm, **style(nm))
        axs[1].plot(Ks, [e["spread_est"] or np.nan for e in et], marker="o", ms=4, label=nm, **style(nm))
    ref_t = [results[D["tag"]]["truths"][truth][have[0]]["event_totals"]["spread_truth"] or np.nan for D in Ds]
    axs[1].plot(Ks, ref_t, color="#555555", lw=1.5, ls=":", marker="s", ms=4, label="truth (particles)")
    axs[1].plot(Ks, [results[D["tag"]]["references"]["spread_hits"] or np.nan for D in Ds], color="#999999",
                lw=1.5, ls="-.", marker="^", ms=4, label="hit count")
    axs[1].plot(Ks, [results[D["tag"]]["references"]["spread_charge"] or np.nan for D in Ds], color="#bbbbbb",
                lw=1.5, ls="--", marker="v", ms=4, label="charge sum")
    axs[0].axhline(1, color="#999999", lw=1)
    for ax in axs:
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("overlay K (showers per super-event)")
    axs[0].set_ylabel("sum(estimate) / sum(truth) per super-event")
    axs[1].set_ylabel("event-to-event std / mean of the sum")
    axs[1].legend(frameon=False, fontsize=7, ncol=2)
    ns = ", ".join(f"K{D['overlay']}: {D['n_super_events']}" for D in Ds)
    fig.suptitle(f"shower totals of {label}; super-events {ns}", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_event_totals_{truth}.png")
    fig.savefig(p, dpi=FIG_DPI)
    plt.close(fig)
    written.append(p)
    return written


def example_maps(Ds, preds, out_dir, targets=(1, 5, 20, 80), truth="n_frac", seed=SEED, prefix="fig"):
    """Hit maps of reticles with truth near each target, with the truths and estimates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    ncol = 2
    nrow = (len(targets) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.1 * nrow), squeeze=False)
    for ax in axs.ravel()[len(targets):]:
        ax.set_visible(False)
    for ax, tg in zip(axs.ravel(), targets):
        best = None
        for D in Ds:
            d = np.abs(D[truth] - tg)
            if best is None or d.min() < best[0]:
                best = (d.min(), D)
        dmin, D = best
        cand = np.flatnonzero(np.abs(D[truth] - tg) == dmin)
        i = int(rng.choice(cand))
        H = np.zeros(D["ny"] * D["nx"], bool)
        S = D["ny"] * D["nx"]
        sel = D["pix"]["ret"] == i
        H[D["pix"]["idx"][sel] - i * S] = True
        H = H.reshape(D["ny"], D["nx"])
        px, py = D["pitch_x"], D["pitch_y"]
        ax.imshow(H, cmap="Greys", vmin=0, vmax=1.4, extent=[0, D["nx"] * px, D["ny"] * py, 0],
                  interpolation="nearest")
        ax.set_xlabel("v [um]")
        ax.set_ylabel("u [um]")
        P = preds[D["tag"]][truth]
        est = ", ".join(f"{nm} {P[nm][i]:.1f}" for nm in ("hits", "cluster_shape", "gbt", "cnn_small")
                        if nm in P)
        ax.set_title(f"{D['tag']}: n_frac {D['n_frac'][i]:.2f}, n_own {D['n_own'][i]}, n_in {D['n_in'][i]}, "
                     f"hits {D['n_hits'][i]}\n{est}",
                     fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_example_maps.png")
    fig.savefig(p, dpi=FIG_DPI)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------------------------
# threshold scan

def noise_only(est, nx, ny, threshold, noise, seed, pitch_x, pitch_y, n_maps=N_NOISE_MAPS):
    """(per-pixel noise hit probability, mean hits, {estimator: mean estimate}) in pure-noise reticles."""
    if noise <= 0:
        return 0.0, 0.0, {nm: 0.0 for nm in est.names}
    hits, _ = readout(np.zeros((n_maps, ny, nx), np.float32), threshold, noise,
                      np.random.default_rng((seed, 999)))
    pix, cl = clusters(hits, pitch_x, pitch_y)
    F = reticle_features(pix, cl, n_maps, ny, nx)
    V = {"feat": F, "q_sum": np.zeros(n_maps), "n": n_maps, "layer": np.zeros(n_maps, np.int64),
         "n_hits": F[:, 0].astype(np.int64),
         "n_clusters": F[:, 1].astype(np.int64), "cl": cl}
    P = est.predict(V)
    return float(hits.mean()), float(F[:, 0].mean()), {nm: float(v.mean()) for nm, v in P.items()}


def scan(raws, tags, args, log):
    """Cheap estimators at each (threshold, noise): binned rms/bias per file, event totals, and the
    noise-only rate."""
    out = []
    groups = ((1, 3), (4, 12), (13, 40), (41, None))
    for thr in SCAN_THRESHOLDS:
        for nz in SCAN_NOISES:
            Ds, _ = prepare(raws, tags, thr, nz, args.seed, args.folds)
            n_events = raws[0]["n_events"]
            row = {"threshold": thr, "noise": nz, "files": {}}
            for t in args.truths:
                P = crossfit(Ds, t, SCAN_ESTIMATORS, args.folds, tags, tags, args.min_count,
                             args.gbt_iter, args.seed, n_events)
                for D in Ds:
                    F = row["files"].setdefault(D["tag"], {})
                    T = D[t].astype(np.float64)
                    F[t] = {"sum_truth": float(T.sum()),
                            **{nm: {"bins": binned(p, T, groups), "event_totals": event_totals(D, p, T)}
                               for nm, p in P[D["tag"]].items()}}
            est = Estimators(SCAN_ESTIMATORS, args.min_count, args.gbt_iter, args.seed).fit(
                view(Ds, [np.ones(D["n"], bool) for D in Ds]), args.truths[0])
            p_pix, h_mean, e_mean = noise_only(est, Ds[0]["nx"], Ds[0]["ny"], thr, nz, args.seed,
                                               float(raws[0]["pitch_x_um"]), float(raws[0]["pitch_y_um"]))
            R0 = raws[0]              # the smallest overlay
            row["noise_only"] = {"p_pixel": p_pix, "hits_per_empty_reticle": h_mean,
                                 "target": args.truths[0], "estimate_per_empty_reticle": e_mean,
                                 "empty_reticles_within_mm": float(R0["count_radii_mm"][-1]),
                                 "empty_reticles_per_shower": float(R0["n_empty_within"][-1])
                                 / (R0["n_super_events"] * R0["overlay"])}
            out.append(row)
            log(f"  scan threshold {thr:g} e, noise {nz:g} e done")
    return out


def scan_table(rows, truth="n_frac"):
    lines = [f"  threshold scan ({truth}): per file rms of estimate - truth in truth groups 1-3, 4-12, "
             f"13-40, 41+ and r = mean sum(estimate)/sum(truth) (biases in the json); noise only: hit "
             f"probability per pixel, hits and estimated particles per empty reticle, and those times the "
             f"empty reticles per shower within the largest count radius"]
    for row in rows:
        npo = row["noise_only"]
        e = npo["estimate_per_empty_reticle"]
        ne = npo["empty_reticles_per_shower"]
        lines.append(f"   threshold {row['threshold']:g} e, noise {row['noise']:g} e: noise only p {npo['p_pixel']:.1e}, "
                     f"{npo['hits_per_empty_reticle']:.3f} hits and " + ", ".join(f"{k} {v:.3f}" for k, v in e.items())
                     + f" per empty reticle; x {ne:.0f} empty reticles within {npo['empty_reticles_within_mm']:g} mm per "
                     f"shower = up to {max(e.values()) * ne:.0f} particles per shower")
        for nm in SCAN_ESTIMATORS:
            cells = []
            for tag, F in row["files"].items():
                T = F[truth][nm]
                rms = " ".join("-" if b["rms"] is None else f"{b['rms']:.2f}" for b in T["bins"])
                cells.append(f"{tag} {rms} r{T['event_totals']['ratio_mean']:.3f}")
            lines.append(f"     {nm:<14} " + " | ".join(cells))
    return lines


# ----------------------------------------------------------------------------------------------
# self-test

def _synthetic(legacy=False):
    """A reticles-like dict whose truths, clusters and folds are known exactly (overlay 2, 20
    events, two 13 x 21 reticles in super-event 0 plus one reticle in each other super-event).
    Truth objects are crossings (part_id = 10 * group + island); with legacy, the rows of one
    (reticle, group) are merged and the file carries no part_id, as written before crossings."""
    ny, nx = 13, 21
    imgs = [np.zeros((ny, nx), np.float32) for _ in range(11)]
    rows = []                # (reticle, group, island, pixels, evt)

    def add(r, g, pixels, evt, island=0):
        rows.append((r, g, island, pixels, evt))
        for x, y, c in pixels:
            imgs[r][y, x] += c

    # reticle 0 (super-event 0): two single-pixel particles; an L of 3 pixels (one particle); a
    # diagonal pair (one particle); a particle of 600 + 600 e (above the cut, no pixel over the
    # threshold); a particle of 800 e (below the cut); a particle split with reticle 1 (3000 e
    # here, 6000 e there: owned by reticle 1, n_frac 1/3 here and 2/3 there); a cluster on the
    # edge column 20
    add(0, 0, [(2, 2, 5000)], 0)
    add(0, 1, [(10, 2, 5000)], 1)
    add(0, 2, [(5, 8, 4000), (6, 8, 2000), (5, 9, 1500)], 0)
    add(0, 3, [(14, 5, 3000), (15, 6, 2500)], 1)
    add(0, 4, [(17, 10, 600), (18, 11, 600)], 0)
    add(0, 5, [(8, 11, 800)], 1)
    add(0, 6, [(20, 1, 3000)], 0)
    add(1, 6, [(0, 1, 6000)], 0)
    # a horizontal line of 3 pixels: theta 0, elongation 6 at 50 x 25 um
    add(0, 7, [(1, 5, 3000), (2, 5, 3000), (3, 5, 3000)], 1)
    # reticle 1: a particle that leaves the plane and comes back: two crossings of one group
    add(1, 30, [(10, 3, 5000)], 0, island=0)
    add(1, 30, [(10, 9, 2500)], 0, island=1)
    for s in range(1, 10):           # one single-pixel particle per other super-event
        add(s + 1, 8 + s, [(4, 4, 7000)], 2 * s)
    if legacy:
        merged = {}
        for r, g, _, pixels, evt in rows:
            merged.setdefault((r, g), [r, g, 0, [], evt])[3].extend(pixels)
        rows = [tuple(v) for v in merged.values()]
    pr = np.array([r[0] for r in rows])
    pg = np.array([r[1] for r in rows])
    pid = pg * 10 + np.array([r[2] for r in rows])
    pq = np.array([sum(c for _, _, c in r[3]) for r in rows], np.float64)
    top = [max(r[3], key=lambda p: p[2]) for r in rows]
    o = np.lexsort((pid, pr))
    sup = np.r_[0, 0, np.arange(1, 10)]
    R = {"img": np.stack(imgs), "super_evt": sup.astype(np.int32),
         "module": np.zeros(11, np.int32), "layer": np.zeros(11, np.int32),
         "r_mm": np.r_[0.1, 0.9, np.full(9, np.nan)], "q_sum": np.stack(imgs).sum(axis=(1, 2)).astype(np.float64),
         "truth_cuts": np.array([1000.0]), "part_reticle": pr[o], "part_group": pg[o],
         "part_q": pq[o].astype(np.float32), "part_iv": np.array([t[0] for t in top])[o].astype(np.int16),
         "part_iu": np.array([t[1] for t in top])[o].astype(np.int16),
         "part_evt": np.array([r[4] for r in rows])[o].astype(np.int32), "overlay": 2,
         "size_x": nx, "size_y": ny, "offset_x": 0, "offset_y": 0, "pitch_x_um": 50.0,
         "pitch_y_um": 25.0, "n_events": 20, "n_super_events": 10, "inputs": np.array(["synthetic"]),
         "input_event_offset": np.array([0]), "input_n_events": np.array([20]),
         "n_empty_within": np.array([0]), "count_radii_mm": np.array([20.0]), "path": "synthetic"}
    if not legacy:
        tot = {i: pq[pid == i].sum() for i in np.unique(pid)}
        R.update({"part_id": pid[o].astype(np.int64), "part_island": (pid[o] % 10).astype(np.int32),
                  "part_q_total": np.array([tot[i] for i in pid[o]], np.float32), "truth": "crossings"})
    R["n_true"] = np.bincount(R["part_reticle"][R["part_q"] > 1000], minlength=11)[:, None].astype(np.int16)
    return with_truth_objects(R)


def self_test():
    """Known truths, cluster table and folds of _synthetic; returns the number of failures."""
    fails = []

    def check(ok, what):
        print(f"  {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            fails.append(what)

    R = _synthetic()
    Ds, ev_fold = prepare([R], ["K2"], 1000.0, 0.0, SEED, 5)
    D = Ds[0]
    check(D["n_in"][:2].tolist() == [7, 3], f"n_in of reticles 0, 1 = 7, 3 (got {D['n_in'][:2].tolist()})")
    check(D["n_own"][:2].tolist() == [6, 3], f"n_own of reticles 0, 1 = 6, 3: the two crossings of one "
          f"particle count apart (got {D['n_own'][:2].tolist()})")
    check(np.allclose(D["n_frac"][:2], [6 + 1 / 3, 2 + 2 / 3]),
          f"n_frac of reticles 0, 1 = 6 1/3, 2 2/3 (got {np.round(D['n_frac'][:2], 4).tolist()})")
    check(int(D["n_own"].sum()) == 18 and int(D["n_in"].sum()) == 19 and abs(D["n_frac"].sum() - 18) < 1e-9,
          "sum n_own 18, sum n_frac 18, sum n_in 19")
    L = prepare([_synthetic(legacy=True)], ["K2"], 1000.0, 0.0, SEED, 5)[0][0]
    check(L["n_own"][:2].tolist() == [6, 2] and L["n_in"][:2].tolist() == [7, 2]
          and np.allclose(L["n_frac"][:2], [6 + 1 / 3, 1 + 2 / 3]),
          "a file without part_id: one object per group, part_q_total from its rows")
    Rc = _synthetic()
    drop = (Rc["part_reticle"] == 1) & (Rc["part_group"] == 6)
    for k in [k for k in Rc if k.startswith("part_")]:
        Rc[k] = Rc[k][~drop]
    Rc["n_true"] = np.bincount(Rc["part_reticle"][Rc["part_q"] > 1000], minlength=11)[:, None].astype(np.int16)
    T, _, info = truths(Rc, 1000.0)
    check(info["above_cut_incomplete"] == 1 and abs(T["n_frac"][0] - (6 + 1 / 3)) < 1e-9,
          "an object with a row missing: kept at its charge share, left out of the plane sum check")
    cl = D["cl"]
    c0 = {k: v[cl["reticle"] == 0] for k, v in cl.items()}
    c1 = {k: v[cl["reticle"] == 1] for k, v in cl.items()}
    sizes = sorted(zip(c0["size"].tolist(), c0["width"].tolist(), c0["height"].tolist(), c0["edge"].tolist()))
    want = sorted([(1, 1, 1, False), (1, 1, 1, False), (3, 2, 2, False), (2, 2, 2, False),
                   (1, 1, 1, True), (3, 3, 1, False)])
    check(sizes == want, f"reticle 0 clusters (size, width, height, edge) = {want} (got {sizes})")
    line = (c0["size"] == 3) & (c0["height"] == 1)
    check(bool(line.any()) and abs(float(c0["theta_deg"][line][0])) < 1e-9
          and abs(float(c0["elongation"][line][0]) - 6.0) < 1e-9, "horizontal line: theta 0, elongation 6")
    diag = (c0["size"] == 2)
    check(bool(diag.any()) and abs(float(c0["fill"][diag][0]) - 0.5) < 1e-12, "diagonal pair: one 8-connected cluster, fill 0.5")
    check(D["n_clusters"][:2].tolist() == [6, 3] and D["n_hits"][:2].tolist() == [11, 3],
          "reticles 0, 1: 6 and 3 clusters, 11 and 3 hits")
    check(D["unclustered"]["n_in"] == 1 and D["unclustered"]["n_own"] == 1 and abs(D["unclustered"]["n_frac"] - 1) < 1e-9,
          "one object above the cut without a hit pixel, for every truth")
    e0, e1 = c0["edge"], c1["edge"]
    check(c0["t_n_in"][e0].tolist() == [1] and c0["t_n_own"][e0].tolist() == [0]
          and np.allclose(c0["t_n_frac"][e0], [1 / 3]) and np.allclose(c1["t_n_frac"][e1], [2 / 3]),
          "split particle: edge cluster 1 inside, 0 owned, 1/3 by charge; 2/3 in the other reticle")
    check(int(c0["a_n_own"].sum()) == 6 and int(c0["t_n_own"].sum()) == 5,
          "reticle 0: 5 owned objects in clusters, 6 after assigning the unclustered one")
    check(ev_fold.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4, "folds: 5 blocks of 4 events")
    check(D["fold"].tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4], "reticle folds from super-events")
    try:
        event_folds(20, [1, 8], 5)
        check(False, "20 events with overlays 1 and 8 into 5 folds refused")
    except Inconsistent:
        check(True, "20 events with overlays 1 and 8 into 5 folds refused")
    f3 = event_folds(20, [2, 3], 3)
    check(f3.tolist() == [0] * 6 + [1] * 6 + [2] * 8, "overlays 2 and 3: boundaries on multiples of 6")
    try:
        super_event_fold(np.repeat(np.arange(5), 4), 6, 3)
        check(False, "a fold boundary inside a super-event refused")
    except Inconsistent:
        check(True, "a fold boundary inside a super-event refused")
    lk = Lookup1D(2).fit(np.array([0, 1, 1, 2, 3, 3]), np.array([0.0, 1, 1, 2, 3, 3]))
    check(np.allclose(lk.predict(np.array([1, 3, 6])), [1, 3, 6]), "1-D lookup: means and proportional tail")
    dv = data_volume({"nx": 42, "ny": 26, "n_hits": np.array([0, 5, 200]), "n_own": np.array([0, 1, 50]),
                      "n_frac": np.array([0.0, 1.0, 50.0]), "n_in": np.array([0, 1, 300]),
                      "super_evt": np.array([0, 0, 0]), "n_super_events": 1})
    check(dv["bitmap_bits"] == 1092 and dv["address_bits"] == 11
          and abs(dv["mean_bits_per_reticle"]["zero_suppressed"] - 11 * 102.5) < 1e-9
          and abs(dv["count_overflow_share"] - 0.5) < 1e-12,
          "data volume of a 42 x 26 reticle: 1092-bit bitmap, 11-bit addresses, 8-bit count overflow")
    print(f"self-test: {len(fails)} failure(s)")
    return len(fails)


# ----------------------------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def truth_floor(D):
    """Per truth: the share of reticles whose truth rounds to 0, over all stored reticles, over
    those with hits and over those with n_in >= 1; and the mean truth per hit count 1, 2, 3."""
    out = {}
    h, nin = D["n_hits"], D["n_in"]
    for t in TRUTHS:
        z = np.rint(D[t]) == 0
        out[t] = {"zero_share": float(z.mean()), "zero_share_with_hits": float(z[h > 0].mean()) if (h > 0).any() else None,
                  "zero_share_n_in_ge1": float(z[nin >= 1].mean()) if (nin >= 1).any() else None,
                  "mean_at_hits_1_2_3": [float(D[t][h == k].mean()) if (h == k).any() else None for k in (1, 2, 3)]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reticles", nargs="*", help="reticles.npz of analysis/pixelav/reticles.py, the same events")
    ap.add_argument("--out-dir", metavar="DIR", help="output directory")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, metavar="E", help="pixel threshold and object charge cut [e]")
    ap.add_argument("--noise", type=float, default=NOISE, metavar="E", help="Gaussian pixel noise sigma [e]")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--extrap-max-k", type=int, default=EXTRAP_MAX_K, metavar="K",
                    help="extrapolation test: train on the files of overlay <= K")
    ap.add_argument("--skip-extrapolation", action="store_true")
    ap.add_argument("--truths", default=",".join(TRUTHS), metavar="LIST",
                    help="comma-separated targets out of " + ", ".join(TRUTHS))
    ap.add_argument("--min-count", type=int, default=MIN_COUNT, metavar="M")
    ap.add_argument("--gbt-iter", type=int, default=GBT_ITER)
    ap.add_argument("--gbt-max-rows", type=int, default=GBT_MAX_ROWS, metavar="N")
    ap.add_argument("--count-bits", type=int, default=COUNT_BITS, metavar="B")
    ap.add_argument("--no-scan", action="store_true", help="skip the threshold and noise scan")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--include-preds", nargs="*", default=[], metavar="NPZ",
                    help="preds.npz of the same inputs and readout whose estimators join the report")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        try:
            return 1 if self_test() else 0
        except Inconsistent as err:
            print(f"self-test: stopped by a failed check: {err}")
            return 1
    if not args.reticles or not args.out_dir:
        ap.error("give reticles.npz files and --out-dir (or --self-test)")
    if args.folds < 2:
        ap.error("--folds must be at least 2")
    args.truths = [t for t in args.truths.split(",") if t]
    if not args.truths or any(t not in TRUTHS for t in args.truths) or len(set(args.truths)) != len(args.truths):
        ap.error(f"--truths takes distinct names out of {', '.join(TRUTHS)}")
    if args.gbt_max_rows < 1 or args.count_bits < 1:
        ap.error("--gbt-max-rows and --count-bits must be positive")
    os.makedirs(args.out_dir, exist_ok=True)

    def log(s):
        print(s, flush=True)

    raws = [load_reticles(p) for p in args.reticles]
    order = np.argsort([R["overlay"] for R in raws], kind="stable")
    raws = [raws[i] for i in order]
    tags = file_tags(raws)
    try:
        Ds, ev_fold = prepare(raws, tags, args.threshold, args.noise, args.seed, args.folds)
    except Inconsistent as err:
        raise SystemExit(f"mip_counting: {err}") from None
    n_events = raws[0]["n_events"]
    settings = {"threshold": args.threshold, "noise": args.noise, "seed": args.seed,
                "folds": args.folds, "event_fold": ev_fold}
    log(f"mip_counting: {len(Ds)} file(s), {n_events} events, folds of "
        f"{np.bincount(ev_fold).tolist()} events; threshold {args.threshold:g} e, noise {args.noise:g} e; "
        f"{Ds[0]['ny']} x {Ds[0]['nx']} pixels; truth objects: {Ds[0]['truth_objects']}")

    # cluster summary and the truths side by side
    summary = {}
    for D in Ds:
        cl = D["cl"]
        ti = D["truth_info"]
        s = {"n_reticles": D["n"], "n_clusters": int(len(cl["size"])),
             "zero_hit_reticles": int(np.count_nonzero(D["n_hits"] == 0)), "truth_objects": ti,
             "floor": truth_floor(D)}
        for t in TRUTHS:
            k = np.bincount(np.minimum(np.rint(cl[f"t_{t}"]).astype(np.int64), 3), minlength=4)
            s[t] = {"sum": float(D[t].sum()), "unclustered": float(D["unclustered"][t]),
                    "lost_in_empty_maps": float(D["lost"][t].sum()),
                    "clusters_with_0_1_2_3plus": k.tolist()}
        sb = np.searchsorted(np.array(SIZE_BINS), cl["size"], side="right") - 1
        s["per_size_bin"] = [{"size_ge": b, "N": int((sb == j).sum()),
                              **{f"mean_{t}": float(cl[f"t_{t}"][sb == j].mean()) if (sb == j).any() else None
                                 for t in TRUTHS},
                              "mean_elongation": float(cl["elongation"][sb == j].mean()) if (sb == j).any() else None}
                             for j, b in enumerate(SIZE_BINS)]
        summary[D["tag"]] = s
        log(f"  {D['tag']}: {D['n']} reticles, {s['n_clusters']} clusters, {s['zero_hit_reticles']} without hits; "
            f"{ti['objects']} truth objects, {ti['above_cut']} above the cut ({ti['above_cut_incomplete']} with a "
            "row missing)")
        for t in TRUTHS:
            log(f"    {t}: sum {s[t]['sum']:.1f}, {s[t]['unclustered']:.1f} without a hit max pixel "
                f"({s[t]['lost_in_empty_maps']:.1f} in reticles without hits), clusters holding 0/1/2/3+ "
                f"(rounded): {s[t]['clusters_with_0_1_2_3plus']}")
        fl = s["floor"]
        log("    share of reticles whose truth rounds to 0 (all / with hits / with n_in >= 1); mean truth at "
            "1, 2, 3 hits: " + "; ".join(
                f"{t} {fl[t]['zero_share']:.3f} / {fl[t]['zero_share_with_hits']:.3f} / "
                f"{fl[t]['zero_share_n_in_ge1']:.3f}, "
                + "/".join(f"{v:.2f}" if v is not None else "-" for v in fl[t]["mean_at_hits_1_2_3"])
                for t in TRUTHS))

    # out-of-fold predictions and the extrapolation
    preds, xpreds = {}, {}
    small = [D["tag"] for D in Ds if D["overlay"] <= args.extrap_max_k]
    large = [D["tag"] for D in Ds if D["overlay"] > args.extrap_max_k]
    try:
        for t in args.truths:
            log(f"  cross-fit {t}")
            P = crossfit(Ds, t, ESTIMATORS, args.folds, tags, tags, args.min_count, args.gbt_iter,
                         args.seed, n_events, log, args.gbt_max_rows)
            for g, by in P.items():
                preds.setdefault(g, {})[t] = by
            if small and large and not args.skip_extrapolation:
                log(f"  extrapolation {t}: train on {small}, evaluate {large}")
                X = crossfit(Ds, t, ESTIMATORS, args.folds, small, large, args.min_count, args.gbt_iter,
                             args.seed, n_events, None, args.gbt_max_rows)
                for g, by in X.items():
                    xpreds.setdefault(g, {})[t] = by
        for path in args.include_preds:
            p2, x2 = load_preds(path, Ds, settings)
            merge_preds(preds, p2)
            merge_preds(xpreds, x2)
    except Inconsistent as err:
        raise SystemExit(f"mip_counting: {err}") from None
    own = {g: {t: {k: v for k, v in by.items() if k in ESTIMATORS} for t, by in P.items()} for g, P in preds.items()}
    xown = {g: {t: {k: v for k, v in by.items() if k in ESTIMATORS} for t, by in P.items()} for g, P in xpreds.items()}
    save_preds(os.path.join(args.out_dir, "preds.npz"), Ds, own, xown, settings)
    np.savez_compressed(os.path.join(args.out_dir, "clusters.npz"),
                        **{f"{D['tag']}__{k}": v for D in Ds for k, v in D["cl"].items()})

    results = evaluate(Ds, preds, xpreds)
    volume = {D["tag"]: data_volume(D, args.count_bits) for D in Ds}
    scan_rows = []
    if not args.no_scan:
        log("  threshold and noise scan")
        try:
            scan_rows = scan(raws, tags, args, log)
        except Inconsistent as err:
            raise SystemExit(f"mip_counting: {err}") from None
    figs = []
    if not args.no_figures:
        for t in args.truths:
            figs += figures(Ds, preds, results, args.out_dir, truth=t)
        figs.append(example_maps(Ds, preds, args.out_dir, truth=args.truths[0], seed=args.seed))
    out = {"settings": {"files": [D["path"] for D in Ds], "tags": tags, "threshold": args.threshold,
                        "noise": args.noise, "seed": args.seed, "folds": args.folds,
                        "fold_events": np.bincount(ev_fold).tolist(), "extrap_max_k": args.extrap_max_k,
                        "skip_extrapolation": args.skip_extrapolation, "truths": args.truths,
                        "truth_objects": Ds[0]["truth_objects"], "reticle_shape": [Ds[0]["ny"], Ds[0]["nx"]],
                        "min_count": args.min_count, "gbt_iter": args.gbt_iter, "gbt_max_rows": args.gbt_max_rows,
                        "count_bits": args.count_bits,
                        "include_preds": [os.path.abspath(p) for p in args.include_preds],
                        "truth_cut": "charge > threshold", "features": FEATURES + ["row_u%d" % i for i in range(Ds[0]["ny"])]
                        + ["col_v%d" % i for i in range(Ds[0]["nx"])], "gbt_layer_extra_feature": "layer"},
           "clusters": summary, "results": results, "data_volume": volume, "scan": scan_rows,
           "figures": figs}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(_jsonable(out), f, indent=1)

    lines = ["", "rms/bias of estimate - truth per truth bin (n_frac rounded to an integer); sum ratio = "
             "mean+-std over super-events of sum(estimate)/sum(truth); spread = std/mean of sum(estimate)"]
    for t in args.truths:
        lines += table(results, t)
    for t in args.truths:
        lines += table(results, t, kind="extrapolation")
    lines.append("  data volume per reticle holding hits (bits) and per shower (kbit):")
    for g, v in volume.items():
        mb, sh = v["mean_bits_per_reticle"], v["bits_per_shower"]
        lines.append(f"    {g}: {v['pixels']} pixels, {v['address_bits']}-bit addresses; "
                     f"{v['reticles_with_hits_per_shower']:.0f} reticles with hits per super-event; "
                     f"per reticle bitmap {mb['bitmap']:.0f}, zero-suppressed {mb['zero_suppressed']:.1f}, count "
                     f"{mb['count']:.0f}; per super-event {sh['bitmap'] / 1e3:.1f} / {sh['zero_suppressed'] / 1e3:.1f} / "
                     f"{sh['count'] / 1e3:.1f} kbit (zero suppression is larger than the bitmap above "
                     f"{v['crossover_hits'] - 1} hits; n_in above the {v['count_bits']}-bit count in "
                     f"{v['count_overflow_share']:.2e} of the reticles)")
        for key, nm in (("per_truth_bin", "n_own"), ("per_n_frac_bin", "n_frac")):
            lines.append(f"      zero-suppressed bits by {nm} bin: " + ", ".join(
                f"{b['bin']}: {b['zero_suppressed']:.0f}" for b in v[key] if b["N"]))
    if scan_rows:
        lines += scan_table(scan_rows, args.truths[0])
    lines.append(f"  written: {args.out_dir}/results.json, preds.npz, clusters.npz, {len(figs)} figure(s)")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
