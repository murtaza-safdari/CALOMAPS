# CALOMAPS teaching plan (summer-student branch)

Instructor + maintainer reference for how the notebooks teach, what each canonical notebook
shows, and what the student scaffold leaves blank. The **canonical** notebooks live on `main`
(photons) and as `-b` variants (pions) — they are the worked solutions. **This branch**
(`summer-student`) strips the science cells to guided blanks; point students back to the
canonical notebooks when they're stuck.

## The two particles

Both particles run on this branch, selected by `PARTICLE` (env `CALOMAPS_GUN_PARTICLE` in nb00):

| Particle | Sees showers that are... |
|----------|--------------------------|
| **photons** (`gamma`, the default) | clean, compact, electromagnetic |
| **pions** (`pi+`) | wide, long, fluctuating, hadronic (often MIP-punch-through in an *EM* calorimeter) |

The **same notebooks** run with a different particle; **comparing the photon and pion results**
at each stage is a first-class learning outcome, not an aside.

## The learning arc (the spine of every notebook)

1. **What a calorimeter is** — a sampling Si–W stack; energy is measured by what the silicon records.
2. **What a particle looks like inside it** — shower development layer by layer; hits in pixels.
3. **Photon vs pion** — EM (pair-production/brem) vs hadronic (nuclear interactions, π⁰→γγ, punch-through).
4. **Reconstructing the incident energy** from what we record — the digital readouts (analog, MIP, hits, clusters).
5. **Why ultra-high granularity helps**, and how **advanced clustering** improves the reconstruction — the capstone.

## Per-notebook design

For each: **Goal** (which arc steps), **canonical cells/plots** (the solution), **photon/pion
contrast**, and **scaffold** (what's given vs what the student writes here).

### nb00 — Simulate your own samples (primer)  ·  arc 1
- **Canonical:** markdown guide + one verify cell. Generate your own `.root` with `ddsim`;
  switch particle with `CALOMAPS_GUN_PARTICLE`; size a quick dataset with `CALOMAPS_NJOBS`.
- **Contrast:** `gamma` vs `pi+` — same command, one env var.
- **Scaffold:** GIVEN in full (it *is* the "how to make data" instructions). Student task =
  actually generate a small dataset (photons or pions) and run the verify cell on it.

### nb01 — The detector and its data  ·  arc 1, 2, 3
- **Goal:** see the geometry, then one shower land on it; read off the sampling fraction.
- **Canonical cells/plots:** (a) parse geometry (layer radii, 100 µm pitch, 30 layers);
  (b) read one event; (c) **transverse barrel view** (x–y, dodecagon + entry arrow);
  (d) **densest layer at 100 µm pixel resolution** (the granularity, made visible);
  (e) **30 per-layer shower slices** (shower develops then fades); (f) **longitudinal
  energy profile** (energy vs depth, W transition); (g) reverse-engineer the pitch/layers
  from data; (h) first-look aggregates (E_true spectrum, hits-vs-E, visible-vs-E → sampling fraction).
- **Contrast:** the pion shower (01b) is visibly wider/longer/patchier and often a single
  MIP track punching through; sampling fraction is lower and fluctuates more. Put the
  (d)/(e)/(f) plots side by side, photon vs pion.
- **Scaffold:** GIVEN = geometry parse, file open, plotting helpers, all markdown context.
  BLANK (with hints) = the hit selection (the +y wedge), and plots (d), (e), (f); the
  student writes the binning to a pixel grid and the per-layer/longitudinal aggregation.

### nb02 — Data extraction + the digital readouts  ·  arc 2, 4, 5(start)
- **Goal:** reduce every event to a few numbers; meet the four readouts; first taste of clustering.
- **Canonical cells/plots:** the four readouts —
  **analog** (Σ E), **MIP** (Σ max(1, round(E_pix/E_MIP))), **hits** (pixels > ½-MIP, purely
  digital), **clusters** (8-connected components per layer, summed); the parallel extraction
  to `.npz`; **readout-vs-E_true scatter (4-panel)**; **longitudinal profile per energy bin**.
  The 8-connected `naive_clusters()` is the reference baseline.
- **Contrast:** for pions the readout↔energy relations are looser (bigger spread at fixed E),
  and the longitudinal profile is deeper/flatter.
- **Scaffold:** GIVEN = the extraction harness, geometry, the `.npz` writer, plot scaffolding,
  AND `naive_clusters()` as a working baseline to beat. BLANK = compute the four readouts per
  event (analog/MIP/hits given as worked examples or hints; clusters wired to the baseline),
  and make the 4-panel plot. **Capstone seed:** "the naive clustering is deliberately simple —
  you'll improve it in nb03 and measure the payoff."

### nb03 — Reconstruction + the granularity/clustering payoff  ·  arc 4, 5
- **Goal:** turn a readout into an energy estimate with uncertainty; show which readout
  reconstructs best, and that better clustering moves the needle.
- **Canonical cells/plots:** train a Deep Quantile Ensemble per readout; **Neyman inversion**
  (measured readout → E_reco ± 1σ); **3-panel dashboard** (linearity, resolution σ/E,
  stochastic term) **with the four readouts overlaid** so the resolution ranking is explicit;
  closure check.
- **Contrast:** pion resolution is worse and less linear (hadronic fluctuations, punch-through);
  comparing the photon and pion dashboards is the headline result.
- **Scaffold:** GIVEN = the training/inversion/dashboard machinery (`quantilenet.py`,
  `dashboard.py`) and the load/plot boilerplate. BLANK = wire the chosen readout(s) through
  training → inversion → dashboard, and **the capstone**: replace `naive_clusters()` with a
  better algorithm (e.g. true connected components across layers, density/Moliere-aware
  merging, or a learned clusterer), re-extract, retrain the cluster readout, and **quantify
  the resolution improvement** on the dashboard. This is the project's culmination.

## What "improved clustering" means here (capstone guidance)
The baseline counts 8-connected pixel groups per layer independently. Better ideas the
students can try (and measure on the nb03 dashboard): 3-D clustering across layers; splitting
merged cores using local density; weighting by deposited charge; or a small learned model.
Success = a lower σ/E curve (or better linearity) than the baseline, shown on the dashboard.

## README / prose guidance (student branch)
- Lead with the arc above; make the photon/pion comparison explicit (put the photon and pion
  plots side by side at the end of each notebook).
- Every scaffold cell states the **task**, a **hint**, and the **expected result** (shape of
  the plot / ballpark number), and links the canonical solution on `main`.
- Keep the from-scratch EAF setup identical to the collaborator path (handbook §6) — students
  are collaborators too. No "this is just a student exercise" tone; this is real R&D.
