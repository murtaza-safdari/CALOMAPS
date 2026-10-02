#!/usr/bin/env python3
"""Parse a PIXELAV cluster output file (pixel_clusters / *.out) and summarize each cluster.

Per cluster: sums the 20 CR-RC time slices into a 13(y) x 21(x) charge map, and reports the
charge-weighted RMS extent along x (21-px axis) and y (13-px Lorentz axis), plus the header
momentum components. Used to confirm the cot/axis mapping (cot_alpha -> y/13px, cot_beta -> x/21px)
and to summarize a full run.

The file is read one cluster at a time (iter_clusters), so memory does not grow with the number
of clusters, apart from one float per cluster kept for the median charge.

Usage:  python parse_cluster.py clusters.out [--full]
        (--full prints only the summary line, not the first 12 clusters)
"""
import argparse
from array import array

import numpy as np

TY, TX = 13, 21


def iter_clusters(path):
    """Yield (hdr, slices) for every <cluster> block of a PIXELAV output file, in file order.

    hdr is the cluster's first 9-field line as a list of strings (hdr[3], hdr[4] are the
    momentum components along x and y), or None when the block has none. slices is a float
    array of shape (n_slices, 13, 21): the block's 21-value rows in groups of 13, a trailing
    incomplete group dropped; n_slices is 0 for a block with fewer than 13 rows. Lines
    starting with "<" and lines before the first <cluster> are skipped."""
    def block(hdr, rows):
        a = np.array(rows, dtype=float).reshape(-1, TX)
        nsl = a.shape[0] // TY
        return hdr, a[:nsl * TY].reshape(nsl, TY, TX)

    rows, hdr = None, None
    with open(path) as f:
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
            if hdr is None and len(p) == 9:
                hdr = p
            elif len(p) == TX:
                rows.append([float(v) for v in p])
    if rows is not None:
        yield block(hdr, rows)


def cluster_summary(slices):
    """(Qtot, x21_rms, y13_rms) of one cluster's slices, or None when its total charge is not
    positive. Qtot sums every slice; the RMS extents are charge-weighted over the 13 x 21 map."""
    g = slices.sum(0)                                   # 13(y) x 21(x) total charge
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
    ap.add_argument("clusters", help="PIXELAV cluster output file (pixel_clusters / *.out)")
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
        px, py = float(hdr[3]), float(hdr[4])
        qtot.append(tot)
        sum_x += xrms
        sum_y += yrms
        if not args.full and i < 12:
            print(f"cluster {i}: px(->x/21)={px:+.4f} py(->y/13)={py:+.4f} | "
                  f"x21_rms={xrms:.2f} y13_rms={yrms:.2f} | Qtot={tot:.0f}")

    if len(qtot):
        q = np.frombuffer(qtot, dtype=float)
        n = len(q)
        print(f"\n[{n} clusters parsed]  "
              f"mean Qtot={q.mean():.0f}  median Qtot={np.median(q):.0f}  "
              f"mean x21_rms={sum_x / n:.2f}  mean y13_rms={sum_y / n:.2f}")


if __name__ == "__main__":
    main()
