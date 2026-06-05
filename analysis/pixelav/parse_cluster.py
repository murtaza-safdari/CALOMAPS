#!/usr/bin/env python3
"""Parse a PIXELAV cluster output file (pixel_clusters / *.out) and summarize each cluster.

Per cluster: sums the 20 CR-RC time slices into a 13(y) x 21(x) charge map, and reports the
charge-weighted RMS extent along x (21-px axis) and y (13-px Lorentz axis), plus the header
momentum components. Used to confirm the cot/axis mapping (cot_alpha -> y/13px, cot_beta -> x/21px)
and to summarize a full run.

Usage:  python parse_cluster.py clusters.out [--full]
"""
import sys
import numpy as np

TY, TX = 13, 21
fn = sys.argv[1]
full = "--full" in sys.argv[2:]

clusters, cur, hdr = [], None, None
with open(fn) as f:
    for line in f:
        s = line.strip()
        if s.startswith("<cluster>"):
            if cur is not None:
                clusters.append((hdr, cur))
            cur, hdr = [], None
            continue
        if cur is None or s.startswith("<time slice") or s.startswith("<"):
            continue
        p = s.split()
        if hdr is None and len(p) == 9:
            hdr = p
        elif len(p) == TX:
            cur.append([float(v) for v in p])
    if cur is not None:
        clusters.append((hdr, cur))

rows_summary = []
for i, (hdr, rows) in enumerate(clusters):
    a = np.array(rows)
    if a.size == 0:
        continue
    nsl = a.shape[0] // TY
    g = a[:nsl * TY].reshape(nsl, TY, TX).sum(0)        # 13(y) x 21(x) total charge
    tot = g.sum()
    if tot <= 0:
        continue
    ys, xs = np.indices(g.shape)
    xbar, ybar = (g * xs).sum() / tot, (g * ys).sum() / tot
    xrms = float(np.sqrt(max(0.0, (g * (xs - xbar) ** 2).sum() / tot)))
    yrms = float(np.sqrt(max(0.0, (g * (ys - ybar) ** 2).sum() / tot)))
    px, py = float(hdr[3]), float(hdr[4])
    rows_summary.append((tot, xrms, yrms, px, py))
    if not full and i < 12:
        print(f"cluster {i}: px(->x/21)={px:+.4f} py(->y/13)={py:+.4f} | "
              f"x21_rms={xrms:.2f} y13_rms={yrms:.2f} | Qtot={tot:.0f}")

if rows_summary:
    arr = np.array(rows_summary)
    print(f"\n[{len(arr)} clusters parsed]  "
          f"mean Qtot={arr[:,0].mean():.0f}  median Qtot={np.median(arr[:,0]):.0f}  "
          f"mean x21_rms={arr[:,1].mean():.2f}  mean y13_rms={arr[:,2].mean():.2f}")
