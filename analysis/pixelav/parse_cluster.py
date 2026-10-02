#!/usr/bin/env python3
"""Parse a PIXELAV cluster output file (pixel_clusters / *.out) and summarize each cluster.

Reads the files of both CALOMAPS drivers: ppixelav2_list_trkpy_real_entry (one straight track
per cluster, 20 time slices) and ppixelav2_inject_deposits (one job of energy deposits per
cluster, the last time slice or all 20).

The time slices are cumulative: slice k holds the charge collected or induced up to time k, so
a cluster's charge is its LAST time slice, never a sum over slices. Per cluster this reports
that charge (Qtot) and its charge-weighted RMS extent along x (21-px axis) and y (13-px Lorentz
axis), plus the header momentum components (list driver) or job id and pair count (injection
driver). Used to confirm the cot/axis mapping (cot_alpha -> y/13px, cot_beta -> x/21px) and to
summarize a full run.

The file is read one cluster at a time (iter_clusters), so memory does not grow with the number
of clusters, apart from one float per cluster kept for the median charge. A file whose name ends
in .gz is read through gzip.

Usage:  python parse_cluster.py clusters.out [--full]
        (--full prints only the summary line, not the first 12 clusters)
"""
import argparse
import gzip
from array import array

import numpy as np

TY, TX = 13, 21
HEADER_FIELDS = (9, 10, 5)      # list driver (without / with the deck line number), injection


def open_text(path):
    """The text file path for reading, through gzip when its name ends in .gz."""
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def iter_clusters(path):
    """Yield (hdr, slices) for every <cluster> block of a PIXELAV output file (plain, or gzipped
    with a name ending in .gz), in file order.

    hdr is the cluster's header line as a list of strings, or None when the block has none:
      list driver, 9 or 10 fields: x y z px py pz neh mody pt [deck_line] (hdr[3], hdr[4] are
        the momentum components along x and y; deck_line is the 1-based deck line);
      injection driver, 5 fields: job_id neh n_seg edep_eV n_out.
    slices is a float array of shape (n_slices, 13, 21): the block's 21-value rows in groups of
    13, a trailing incomplete group dropped; n_slices is 20 or 1 (the last time slice only), 0
    for a block with fewer than 13 rows. The slices are cumulative: slices[-1] is the charge.
    Lines starting with "<" and lines before the first <cluster> are skipped."""
    def block(hdr, rows):
        a = np.array(rows, dtype=float).reshape(-1, TX)
        nsl = a.shape[0] // TY
        return hdr, a[:nsl * TY].reshape(nsl, TY, TX)

    rows, hdr = None, None
    with open_text(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("<cluster>"):
                if rows is not None:
                    yield block(hdr, rows)
                rows, hdr = [], None
                continue
            if rows is None or s.startswith("<"):
                continue
            p = s.split()
            if hdr is None and len(p) in HEADER_FIELDS:
                hdr = p
            elif len(p) == TX:
                rows.append([float(v) for v in p])
    if rows is not None:
        yield block(hdr, rows)


def cluster_summary(slices):
    """(Qtot, x21_rms, y13_rms) of one cluster, or None when its charge is not positive. Qtot is
    the charge in the last time slice (the slices are cumulative); the RMS extents are
    charge-weighted over that slice's 13 x 21 map."""
    g = slices[-1]                                      # 13(y) x 21(x) charge
    tot = g.sum()
    if tot <= 0:
        return None
    ys, xs = np.indices(g.shape)
    xbar, ybar = (g * xs).sum() / tot, (g * ys).sum() / tot
    xrms = float(np.sqrt(max(0.0, (g * (xs - xbar) ** 2).sum() / tot)))
    yrms = float(np.sqrt(max(0.0, (g * (ys - ybar) ** 2).sum() / tot)))
    return tot, xrms, yrms


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize the clusters of a PIXELAV output file.")
    ap.add_argument("clusters", help="PIXELAV cluster output file (pixel_clusters / *.out, or *.gz)")
    ap.add_argument("--full", action="store_true",
                    help="print only the summary line, not the first 12 clusters")
    args = ap.parse_args(argv)

    qtot = array("d")                   # every parsed cluster's Qtot, for the median
    sum_x = sum_y = 0.0
    for i, (hdr, slices) in enumerate(iter_clusters(args.clusters)):
        if slices.shape[0] == 0:
            continue
        summary = cluster_summary(slices)
        if summary is None:
            continue
        tot, xrms, yrms = summary
        qtot.append(tot)
        sum_x += xrms
        sum_y += yrms
        if not args.full and i < 12:
            if hdr is not None and len(hdr) == 5:
                label = f"job {hdr[0]} neh={hdr[1]} n_out={hdr[4]}"
            elif hdr is not None:
                label = f"px(->x/21)={float(hdr[3]):+.4f} py(->y/13)={float(hdr[4]):+.4f}"
            else:
                label = "no header"
            print(f"cluster {i}: {label} | "
                  f"x21_rms={xrms:.2f} y13_rms={yrms:.2f} | Qtot={tot:.0f}")

    if len(qtot):
        q = np.frombuffer(qtot, dtype=float)
        n = len(q)
        print(f"\n[{n} clusters parsed; Qtot = last time slice]  "
              f"mean Qtot={q.mean():.0f}  median Qtot={np.median(q):.0f}  "
              f"mean x21_rms={sum_x / n:.2f}  mean y13_rms={sum_y / n:.2f}")


if __name__ == "__main__":
    main()
