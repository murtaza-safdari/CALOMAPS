#!/usr/bin/env python3
"""One event of a per-step deposit file: step_subset.py <in.npz> <event> <out.npz>.
Per-particle and per-step arrays keep the rows of that event (event ids unchanged), with the
particle links tmc and parent renumbered to the rows kept and num_events set to 1, so that
analysis/pixelav/deposit_jobs.py runs on the result; every other array is copied; subset_event,
subset_source, subset_first_particle and subset_first_step record the cut (the source rows of
the event are consecutive): source row = subset_first_particle + row here."""
import sys
import numpy as np

src, evt, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = np.load(src, allow_pickle=False)
n_p, n_h = len(d["evt_id_particle"]), len(d["evt_id_hit"])
keep_p, keep_h = d["evt_id_particle"] == evt, d["evt_id_hit"] == evt
res = {}
for k in d.files:
    a = d[k]
    if a.ndim == 1 and len(a) == n_p and k != "geo_si_centers_mm" and k != "meta":
        res[k] = a[keep_p]
    elif a.ndim == 1 and len(a) == n_h:
        res[k] = a[keep_h]
    else:
        res[k] = a
rows_p, rows_h = np.flatnonzero(keep_p), np.flatnonzero(keep_h)
assert np.all(np.diff(rows_p) == 1) and np.all(np.diff(rows_h) == 1), "the event's rows are not consecutive"
# particle links are row indices of the source file: make them rows of this file
res["tmc"] = res["tmc"] - rows_p[0]
res["parent"] = np.where(res["parent"] >= 0, res["parent"] - rows_p[0], res["parent"])
assert res["tmc"].min() >= 0 and res["tmc"].max() < len(rows_p) and res["parent"].max() < len(rows_p), "a link leaves the event"
res["num_events"] = np.int64(1)
res["subset_event"] = np.int64(evt)
res["subset_first_particle"] = np.int64(rows_p[0])     # row of the source file that is row 0 here
res["subset_first_step"] = np.int64(rows_h[0])
res["subset_source"] = np.str_(src)
np.savez_compressed(out, **res)
print(f"{out}: event {evt}, {int(keep_p.sum())} of {n_p} particles, {int(keep_h.sum())} of {n_h} steps, "
      f"edep {res['tedep'].astype(float).sum():.6f} GeV, first particle row {rows_p[0]}, first step row {rows_h[0]}")
