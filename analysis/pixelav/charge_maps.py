#!/usr/bin/env python3
"""PIXELAV charge maps of a set of deposit jobs -> the charge of every group on the pixel grid.

    python analysis/pixelav/charge_maps.py <jobs prefix> [--out <file>]

<prefix> is the one given to deposit_jobs.py and run_deposit_jobs.sh: the tables
<prefix>.jobs.npz and, next to them, the driver outputs <prefix>.part000.out ... (one per part,
deposit_jobs.py --parts N), or <prefix>.out for jobs written as one <prefix>.jobs.txt. Each
output may be gzipped (<prefix>.part000.out.gz); a part present both plain and gzipped is refused.

MAPS: a job's charge map is the last time slice of its <cluster> block (parse_cluster.iter_clusters;
the slices are cumulative), 13 x 21 pixels in electrons. Window column ix (0..20) is global pixel
iv = job_win_iv0 + ix and row iy (0..12) is iu = job_win_iu0 + iy (deposit_jobs.py, TILES): pixel iv
covers v in [iv, iv+1) * pitch_x and iu covers u in [iu, iu+1) * pitch_y. PIXELAV's carriers do not
interact, so maps add: the charge of a group is the sum of the maps of its jobs on the global grid,
where the guard bands of its neighbouring tiles overlap and the jobs of a split tile share one
window.

CHECKS (exit 1, nothing written): every job id of the table appears in exactly one block of the
outputs, in the part file its job_part names, and no other id appears; every block has the
injection driver's header and 1 or NCRRC complete time slices; the header's n_seg equals job_n_seg
and its deposited energy job_edep_keV, give or take the text's rounding (0.005 eV a segment); the
pairs made, neh + n_out, equal job_edep_keV / ev_per_pair within n_seg (each segment makes the
floor of its pair count or one more); every output's sensor title and pitch line agree, and the
pitch line's xsize, ysize and thick equal the table's pitch_x_um, pitch_y_um and thick_um.

OUTPUT <prefix>.charges.npz (or --out), compressed:
  per (group, pixel), sorted by group and then by iv, iu; pixels whose summed charge is 0 dropped:
      pix_group_id, pix_iv, pix_iu (int32), pix_q (float32, electrons summed over the group's jobs)
  per job, in the table's order: job_id, job_neh (pairs transported), job_n_out (pairs made outside
      the window, not transported), job_q (the map's charge)
  per group, in the table's order: every group_* column of the jobs table, group_q (its charge,
      the sum of its pix_q before rounding to float32) and group_n_pix (its pix_* rows)
  the jobs table's scalars and provenance keys (pitch_x_um, ..., sim_*, geo_*), as they are
  the collector's own: n_jobs, n_groups, n_pix, total_q, total_neh, total_n_out, total_edep_keV,
      out_files, jobs_npz, pixelav_title (the outputs' first line), pixelav_stimstp_ps
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from parse_cluster import iter_clusters, open_text   # noqa: E402

NCRRC = 20                        # the driver's time slices; "last" writes only the final one
SEG_ROUND_EV = 0.005              # the jobs text holds each segment's energy to 0.01 eV
PITCH_TOL_UM = 1e-4               # the pitch line is written with %f
_EXAMPLES = 5


def load_table(path):
    """The arrays of the jobs npz path, as a dict."""
    try:
        with np.load(path, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}
    except (FileNotFoundError, OSError, ValueError) as err:
        sys.exit(f"charge_maps: cannot read {path}: {err}")


def output_files(prefix, parts):
    """The driver output of each part of the jobs of prefix, in part order: <prefix>.partNNN.out
    for parts > 1, <prefix>.out for one, or the same name with .gz. SystemExit when one is missing
    or present both plain and gzipped."""
    names = [f"{prefix}.out"] if parts == 1 else [f"{prefix}.part{i:03d}.out" for i in range(parts)]
    found, missing, both = [], [], []
    for n in names:
        plain, gz = os.path.isfile(n), os.path.isfile(n + ".gz")
        if plain and gz:
            both.append(n)
        elif plain or gz:
            found.append(n if plain else n + ".gz")
        else:
            missing.append(n)
    if missing or both:
        sys.exit("charge_maps: " + "; ".join(
            ([f"no {', '.join(missing)} (nor .gz)"] if missing else [])
            + ([f"{', '.join(both)} present both plain and gzipped"] if both else []))
            + "; nothing written")
    return found


def read_head(path):
    """(sensor title, [xsize, ysize, thick, stimstp]) from the first two lines of a driver output."""
    with open_text(path) as f:
        title, line = f.readline().strip(), f.readline().split()
    try:
        values = [float(x) for x in line]
    except ValueError:
        values = []
    if len(values) != 4:
        sys.exit(f"charge_maps: {path}: line 2 is not 'xsize ysize thick stimstp': {' '.join(line)!r}")
    return title, values


def read_outputs(files):
    """Per <cluster> block of the files, in file order: the header fields (id, neh, n_seg, edep_eV,
    n_out), the file index, the number of time slices and the charge of the last one; and per
    nonzero pixel of that slice its block, row iy, column ix and charge. Blocks without an
    injection-driver header are counted in "bad_header" (first one in "bad_example")."""
    hdr_rows, nsl, q_blk, fileno = [], [], [], []
    p_blk, p_iy, p_ix, p_q = [], [], [], []
    bad, bad_example = 0, None
    for k, path in enumerate(files):
        for i, (hdr, slices) in enumerate(iter_clusters(path)):
            try:
                if hdr is None or len(hdr) != 5:
                    raise ValueError
                row = (int(hdr[0]), int(hdr[1]), int(hdr[2]), float(hdr[3]), int(hdr[4]))
            except ValueError:
                bad += 1
                bad_example = bad_example or f"{path} block {i}"
                continue
            b = len(hdr_rows)
            hdr_rows.append(row)
            nsl.append(slices.shape[0])
            fileno.append(k)
            if slices.shape[0] == 0:
                q_blk.append(0.0)
                continue
            m = slices[-1]
            iy, ix = np.nonzero(m)
            q_blk.append(float(m.sum()))
            p_blk.append(np.full(len(iy), b, np.int64))
            p_iy.append(iy.astype(np.int16))
            p_ix.append(ix.astype(np.int16))
            p_q.append(m[iy, ix])
    h = np.array(hdr_rows, dtype=object).reshape(-1, 5)
    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype)  # noqa: E731
    return {"id": h[:, 0].astype(np.int64), "neh": h[:, 1].astype(np.int64),
            "n_seg": h[:, 2].astype(np.int64), "edep_eV": h[:, 3].astype(np.float64),
            "n_out": h[:, 4].astype(np.int64), "file": np.array(fileno, np.int64),
            "n_slices": np.array(nsl, np.int64), "q": np.array(q_blk, np.float64),
            "pix_blk": cat(p_blk, np.int64), "pix_iy": cat(p_iy, np.int16),
            "pix_ix": cat(p_ix, np.int16), "pix_q": cat(p_q, np.float64),
            "bad_header": bad, "bad_example": bad_example}


def _ids(x):
    x = np.asarray(x)
    more = f" ... ({len(x)})" if len(x) > _EXAMPLES else ""
    return ", ".join(str(v) for v in x[:_EXAMPLES].tolist()) + more


def match_jobs(B, T, files, parts):
    """(row of each block in the job table, problems): the hard checks of the module docstring on
    the blocks B (read_outputs) against the jobs table T. Rows of blocks with an unknown id are -1."""
    problems = []
    if B["bad_header"]:
        problems.append(f"{B['bad_header']} blocks without an injection-driver header (first: "
                        f"{B['bad_example']})")
    bad_sl = ~np.isin(B["n_slices"], (1, NCRRC))
    if bad_sl.any():
        problems.append(f"jobs {_ids(B['id'][bad_sl])} have {_ids(B['n_slices'][bad_sl])} complete "
                        f"time slices, not 1 or {NCRRC}")
    tid = np.asarray(T["job_id"], np.int64)
    order = np.argsort(tid, kind="stable")
    sid = tid[order]
    if not len(sid):
        sys.exit("charge_maps: the jobs table holds no jobs")
    pos = np.minimum(np.searchsorted(sid, B["id"]), len(sid) - 1)
    known = sid[pos] == B["id"]
    row = np.where(known, order[pos], -1)
    count = np.bincount(row[known], minlength=len(tid))
    if (~known).any():
        problems.append(f"{np.count_nonzero(~known)} blocks hold job ids the table does not "
                        f"have: {_ids(B['id'][~known])}")
    if (count > 1).any():
        problems.append(f"{np.count_nonzero(count > 1)} job ids appear more than once: "
                        f"{_ids(tid[count > 1])}")
    if (count == 0).any():
        problems.append(f"{np.count_nonzero(count == 0)} job ids of the table appear in no "
                        f"output: {_ids(tid[count == 0])}")
    r, k = row[known], np.flatnonzero(known)
    part_of_file = np.arange(len(files)) if parts > 1 else np.zeros(1, np.int64)
    wrong_part = T["job_part"][r] != part_of_file[B["file"][k]]
    if wrong_part.any():
        problems.append(f"{np.count_nonzero(wrong_part)} jobs are in another part's output than "
                        f"their job_part: {_ids(B['id'][k][wrong_part])}")
    n_seg = T["job_n_seg"][r]
    bad_seg = B["n_seg"][k] != n_seg
    if bad_seg.any():
        problems.append(f"{np.count_nonzero(bad_seg)} jobs have another segment count than the "
                        f"table's: {_ids(B['id'][k][bad_seg])}")
    e_tab = T["job_edep_keV"][r] * 1e3
    bad_e = ~(np.abs(B["edep_eV"][k] - e_tab) <= n_seg * SEG_ROUND_EV + 1e-3 + 1e-9 * e_tab)
    if bad_e.any():
        problems.append(f"{np.count_nonzero(bad_e)} jobs deposit another energy than the table's "
                        f"(largest difference {np.abs(B['edep_eV'][k] - e_tab)[bad_e].max():.3f} "
                        f"eV): {_ids(B['id'][k][bad_e])}")
    pairs = B["neh"][k] + B["n_out"][k]
    expect = e_tab / float(T["ev_per_pair"])
    bad_p = ~(np.abs(pairs - expect) < n_seg * (1 + SEG_ROUND_EV / float(T["ev_per_pair"])) + 1e-6)
    if bad_p.any():
        problems.append(f"{np.count_nonzero(bad_p)} jobs made another number of pairs than "
                        f"edep / {float(T['ev_per_pair']):g} eV, give or take one a segment: "
                        f"{_ids(B['id'][k][bad_p])}")
    return row, problems


def accumulate(B, T, row):
    """The per-(group, pixel) charges and the per-job and per-group totals of the blocks B, whose
    table rows are row (each job exactly once)."""
    n_jobs = len(T["job_id"])
    J = {"neh": np.zeros(n_jobs, np.int64), "n_out": np.zeros(n_jobs, np.int64),
         "q": np.zeros(n_jobs, np.float64)}
    J["neh"][row], J["n_out"][row], J["q"][row] = B["neh"], B["n_out"], B["q"]

    gid = np.asarray(T["group_id"], np.int64)
    gorder = np.argsort(gid, kind="stable")
    gpos = np.minimum(np.searchsorted(gid[gorder], T["job_group_id"]), len(gid) - 1)
    if not np.array_equal(gid[gorder][gpos], T["job_group_id"]):
        sys.exit("charge_maps: jobs whose group id the group table does not have; nothing written")
    job_grow = gorder[gpos]

    r = row[B["pix_blk"]]
    g = job_grow[r]
    iv = T["job_win_iv0"][r] + B["pix_ix"]
    iu = T["job_win_iu0"][r] + B["pix_iy"]
    n_groups = len(gid)
    if len(g):
        iv0, iu0 = int(iv.min()), int(iu.min())
        nv, nu = int(iv.max()) - iv0 + 1, int(iu.max()) - iu0 + 1
        if n_groups * nv * nu >= 2**62:
            sys.exit(f"charge_maps: {n_groups} groups on a {nv} x {nu}-pixel grid overflow the pixel key")
        key = (g * nv + (iv - iv0)) * nu + (iu - iu0)
        ukey, inv = np.unique(key, return_inverse=True)
        q = np.bincount(inv.ravel(), B["pix_q"], len(ukey))
        keep = q != 0
        ukey, q = ukey[keep], q[keep]
        pg, rest = ukey // (nv * nu), ukey % (nv * nu)
        piv, piu = rest // nu + iv0, rest % nu + iu0
    else:
        pg = piv = piu = np.zeros(0, np.int64)
        q = np.zeros(0)
    G = {"q": np.bincount(pg, q, n_groups), "n_pix": np.bincount(pg, minlength=n_groups),
         "pairs": np.bincount(job_grow, J["neh"] + J["n_out"], n_groups)}
    pix = {"pix_group_id": gid[pg].astype(np.int32), "pix_iv": piv.astype(np.int32),
           "pix_iu": piu.astype(np.int32), "pix_q": q.astype(np.float32)}
    return pix, J, G


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prefix", help="the jobs prefix of deposit_jobs.py: <prefix>.jobs.npz and the "
                    "driver outputs <prefix>.partNNN.out[.gz] (or <prefix>.out[.gz]) next to it")
    ap.add_argument("--out", metavar="FILE", help="output npz (default <prefix>.charges.npz)")
    args = ap.parse_args(argv)
    out = args.out or args.prefix + ".charges.npz"

    jobs_npz = args.prefix + ".jobs.npz"
    T = load_table(jobs_npz)
    parts = int(T["parts"])
    files = output_files(args.prefix, parts)
    heads = [read_head(f) for f in files]
    title, (xsize, ysize, thick, stimstp) = heads[0]
    problems = []
    odd = [(f, h) for f, h in zip(files, heads) if h != heads[0]]
    if odd:
        problems.append("the outputs' sensor title or pitch line differ: "
                        + "; ".join(f"{os.path.basename(f)}: {h[0]!r} {h[1]}"
                                    for f, h in [(files[0], heads[0])] + odd[:_EXAMPLES]))
    want = (float(T["pitch_x_um"]), float(T["pitch_y_um"]), float(T["thick_um"]))
    if any(abs(a - b) > PITCH_TOL_UM for a, b in zip((xsize, ysize, thick), want)):
        problems.append(f"the outputs' sensor is {xsize:g} x {ysize:g} um, {thick:g} um thick, the "
                        f"table's {want[0]:g} x {want[1]:g} um, {want[2]:g} um thick")

    B = read_outputs(files)
    row, more = match_jobs(B, T, files, parts)
    problems += more
    if problems:
        sys.exit("charge_maps: " + "; ".join(problems) + "; nothing written")
    pix, J, G = accumulate(B, T, row)

    copied = {k: v for k, v in T.items()
              if not k.startswith(("job_", "group_")) and k not in ("job_id", "group_id")}
    n_jobs, n_groups, n_pix = len(T["job_id"]), len(T["group_id"]), len(pix["pix_q"])
    tot = {"q": float(J["q"].sum()), "neh": int(J["neh"].sum()), "n_out": int(J["n_out"].sum()),
           "edep_keV": float(T["job_edep_keV"].sum())}
    arrays = dict(
        **pix,
        job_id=T["job_id"], job_neh=J["neh"], job_n_out=J["n_out"], job_q=J["q"],
        **{k: v for k, v in T.items() if k.startswith("group_")},
        group_q=G["q"], group_n_pix=G["n_pix"],
        **copied,
        n_jobs=np.int64(n_jobs), n_groups=np.int64(n_groups), n_pix=np.int64(n_pix),
        total_q=np.float64(tot["q"]), total_neh=np.int64(tot["neh"]),
        total_n_out=np.int64(tot["n_out"]), total_edep_keV=np.float64(tot["edep_keV"]),
        out_files=np.array([os.path.abspath(f) for f in files]),
        jobs_npz=np.array(os.path.abspath(jobs_npz)),
        pixelav_title=np.array(title), pixelav_stimstp_ps=np.float64(stimstp))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".partial"
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    made = tot["neh"] + tot["n_out"]
    dq = G["q"] - G["pairs"]
    worst = int(np.argmax(np.abs(dq))) if n_groups else 0
    gz = sum(f.endswith(".gz") for f in files)
    lines = [
        f"charge maps: {os.path.abspath(args.prefix)}",
        f"  outputs: {len(files)} file(s), {gz} gzipped; sensor {title!r}, pitch {xsize:g} x "
        f"{ysize:g} um, {thick:g} um thick",
        f"  jobs: {n_jobs}, each id once; groups: {n_groups}; (group, pixel) rows: {n_pix}",
        f"  pairs made: {made} = edep / {float(T['ev_per_pair']):g} eV x "
        f"{made / (tot['edep_keV'] * 1e3 / float(T['ev_per_pair'])):.6f}",
        f"  pairs outside their windows: {tot['n_out']} ({tot['n_out'] / made:.2e} of those made), "
        f"in {int(np.count_nonzero(J['n_out']))} jobs",
        f"  charge: {tot['q']:.6g} e; sum q / sum neh = {tot['q'] / tot['neh']:.5f}",
        f"  largest |group q - pairs made|: {abs(dq[worst]):.1f} e (group "
        f"{int(T['group_id'][worst])}, {int(G['pairs'][worst])} pairs)" if n_groups else "  no groups",
        f"  written: {out}",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
