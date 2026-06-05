# CALOMAPS — the summer-student branch

Welcome. You're about to do real detector R&D: simulate a next-generation **digital
electromagnetic calorimeter** (DECAL), watch particles shower inside it, and reconstruct
their energy from what the silicon records — then try to do it *better* than the baseline.
This is not a canned exercise. The clustering capstone in the last notebook is open: there
is no answer key for it, because the answer isn't known.

This branch (`summer-student`) ships the notebooks as **scaffolds** — the context, the
plumbing, and the plotting are written for you; the *science* cells are left blank with a
task, a hint, and the expected result. You fill them in. The fully worked solutions live
on the **`main`** branch — try first, peek only when you're stuck.

---

## The learning arc

Each notebook is one step of the same story. Keep the arc in mind — it's why the notebooks
are ordered this way:

1. **What a calorimeter is** — a sampling silicon–tungsten stack. We measure a particle's
   energy by what the thin silicon layers record as the shower passes through.
2. **What a particle looks like inside it** — the shower develops layer by layer; each
   100 µm pixel is a yes/no hit. You'll *see* one shower three ways.
3. **Photon vs pion (EM vs hadronic)** — photons make clean, compact electromagnetic
   showers; pions make wide, deep, fluctuating hadronic ones (and often a single
   minimum-ionizing track punching straight through this *EM* calorimeter). This contrast
   is a first-class result, not an aside.
4. **Reconstructing the incident energy** from the digital readouts — analog charge, MIP
   counting, raw hit counts, and clusters — and ranking which readout does it best.
5. **Why ultra-high granularity helps, and how better clustering cashes it in** — the
   capstone. You replace the deliberately-naive baseline clusterer with something smarter
   and *measure* the resolution it buys you.

---

## Two students, one set of notebooks

You are one of **two students working this same branch**, each on a different particle:

| You are the... | Your particle | `PARTICLE` value | Your showers are... |
|---|---|---|---|
| **junior** | photons | `"gamma"` (the default) | clean, compact, *electromagnetic* |
| **senior** | pions | `"pi+"` | wide, long, fluctuating, *hadronic* (often punch-through) |

You run the **exact same notebooks**. The only thing that differs is one variable —
`PARTICLE` — set in a clearly-marked cell near the top of each notebook. Everything else
(the dataset, the file glob, the output filenames, the trained-model folder) is *derived*
from it, so your photon run and your partner's pion run never clobber each other.

Then you **compare**. The photon-vs-pion difference is the headline physics result of the
whole project — see "Compare with your partner" below.

---

## Setting up (from scratch on EAF)

You're a collaborator like any other, so your setup is the standard one — there's no
separate "student track."

1. **Get an EAF account and spawn a server.** Follow **`docs/handbook.md` §6** ("Setting up
   your environment"): §6.1 accounts, §6.2 spawn an EAF server, §6.3 clone the code and
   create the one-time `~/lib_hack` symlink that `ddsim` needs, §6.4 `source
   setup/setup_calomaps.sh` from a JupyterLab terminal.
2. **Check out this branch:** `git checkout summer-student`.
3. **Two kernels.** Notebooks 00/01/02 use the **`Python (Key4hep)`** kernel (CPU is fine).
   Notebook 03 trains a neural-net surrogate and wants the **`Key4hep + GPU`** kernel — see
   **`docs/handbook.md` §11.2** for the one-time GPU-kernel recipe, and spawn a GPU server
   when you reach nb03.

If anything in setup misbehaves, **`docs/troubleshooting.md`** very likely already has the
fix (it's a catalog of the environment's quirks).

---

## The notebook path

Work them in order. Each builds on the `.npz` / models the previous one produced.

### nb00 — Simulate *your* sample · `notebooks/00_simulate_your_samples.ipynb`
**Make your own data.** This one is given in full — it *is* the "how to generate data"
guide. You run `ddsim` (Geant4 via DD4hep) to shoot your particle into the detector and
write ROOT files. The particle is one environment variable:

```bash
# junior (photons) — this is the default:
CALOMAPS_NJOBS=40 CALOMAPS_GUN_PARTICLE=gamma bash sim/generate_batched.sh
#   -> data_spectrum_100um_400GeV/sim_photons_part*.root

# senior (pions) — same command, one env var changed:
CALOMAPS_NJOBS=40 CALOMAPS_GUN_PARTICLE=pi+  bash sim/generate_batched.sh
#   -> data_spectrum_100um_400GeV_piplus/sim_piplus_part*.root
```

`CALOMAPS_NJOBS` sizes the sample (number of ROOT files); start small to smoke-test, then
generate a fuller dataset. The last cell of nb00 verifies the files you produced. (See
`docs/handbook.md` §8 for a quick smoke-test sim and §9 for production runs.)

> **Your task here:** actually generate a small dataset for *your* particle and run the
> verify cell on it.

### nb01 — See the shower · `notebooks/01_detector_and_data.ipynb`
Open one file, parse the geometry (30 silicon layers, 100 µm pixels, 12-sided barrel), and
watch a single shower land on it. You'll produce: the densest layer rendered at native
100 µm resolution (the pixelization made visible), the 30 per-layer shower slices (narrow →
shower-max → fade), and the longitudinal energy-vs-depth profile. You also measure the
**sampling fraction** straight from the data.

> **You fill in:** the +y wedge hit-selection and layer assignment, and the data step of the
> three shower plots (densest layer, per-layer slices, longitudinal profile).

### nb02 — The digital readouts · `notebooks/02_data_extraction.ipynb`
Reduce every event to a handful of numbers and meet the **four readouts**: **analog**
(Σ energy), **MIP** (MIPs-per-pixel), **hits** (pixels over ½-MIP — purely digital), and
**clusters** (8-connected components per layer). You extract them over all your events in
parallel and save one small `.npz`.

> **You fill in:** the four readout computations per event, and the 4-panel
> readout-vs-energy plot. The naive `naive_clusters()` baseline is **given** — it's
> deliberately simple, and beating it is the nb03 capstone, so notice where it throws
> information away.

### nb03 — Reconstruct, and improve · `notebooks/03_ml_training_and_eval.ipynb`
Train a Deep Quantile Ensemble per readout, invert it with a Neyman construction (measured
readout → energy ± 1σ), and read off the **3-panel dashboard**: linearity, resolution
(σ/E), and the stochastic term. This ranks the readouts and shows where the digital ones
break (pixel saturation at high energy).

> **You fill in:** wiring each readout through training → inversion → dashboard, **and the
> capstone** (below).

---

## Setting `PARTICLE`

In **each** notebook there is a clearly-marked cell near the top:

```python
PARTICLE = "gamma"   # junior: "gamma";  senior: "pi+"
```

Set it once, run the cell, and the rest of the notebook keys off it:

```python
DATASET   = "data_spectrum_100um_400GeV" if PARTICLE == "gamma" else "data_spectrum_100um_400GeV_piplus"
FILE_GLOB = "sim_photons_part*.root"     if PARTICLE == "gamma" else "sim_piplus_part*.root"
```

The output names follow too — nb02 writes `models/decal_extracted_data_gamma.npz` or
`..._piplus.npz`, and nb03 reads the matching one and trains into a per-particle model
folder. So you and your partner can run on the same machine without stepping on each other.
**The only edit you make per-student is that one `PARTICLE` line.** (For nb00, it's the
`CALOMAPS_GUN_PARTICLE` env var instead.)

---

## Compare with your partner

The point of having two of you is the **photon-vs-pion comparison**. At three moments, put
your plots side by side:

1. **Shower shape (end of nb01).** Compare your densest-layer image, your per-layer slices,
   and your longitudinal profile. The pion's is visibly wider, longer, patchier — sometimes
   just a single MIP dot per layer (punch-through). The photon's is one tight EM core.
2. **Sampling fraction & readouts (end of nb02).** Whose visible-vs-true line is steeper?
   Whose readout clouds are tighter at fixed energy? Whose clustering turns over sooner? The
   pion relations are looser (hadronic fluctuations) and its longitudinal profile deeper and
   flatter.
3. **Reconstruction resolution (end of nb03).** Put your dashboards together. **Pion
   resolution is worse and less linear** than photon — that's the EM-vs-hadronic story, and
   comparing the two dashboards *is* the result. Then compare your capstone gains: smarter
   clustering usually helps **pions more**, because hadronic showers are exactly the messy,
   multi-core, across-layers objects the naive baseline handles worst.

Write the numbers down. A small table — best baseline σ/E, improved-cluster σ/E, and the
relative gain, for *both* particles — is the headline output of the project.

---

## When you're stuck: the worked solutions

The fully worked notebooks live on the **`main`** branch:

| Scaffold (this branch) | Photon solution (`main`) | Pion solution (`main`) |
|---|---|---|
| `01_detector_and_data.ipynb` | `01_detector_and_data.ipynb` | `01b_detector_and_data_pions.ipynb` |
| `02_data_extraction.ipynb` | `02_data_extraction.ipynb` | `02b_data_extraction_pions.ipynb` |
| `03_ml_training_and_eval.ipynb` | `03_ml_training_and_eval.ipynb` | `03b_ml_training_and_eval_pions.ipynb` |

Every blank cell tells you which one to look at. **Try first, peek if stuck.** You learn
the physics by struggling with the cell, not by reading the answer — but a quick look to
unblock yourself is fine. To see them: `git checkout main`, open the notebook, then
`git checkout summer-student` to come back. (The capstone has **no** solution on `main` —
that part is genuinely open.)

---

## The capstone — better clustering, measured on the dashboard

This is the culmination of the project and the reason high granularity matters.

Everything up to nb03 used the **baseline** clusterer: `naive_clusters()` counts 8-connected
pixel groups **per layer, independently**, and sums them. At 100 µm pitch a shower is
resolved into thousands of pixels across 30 layers — and the baseline throws most of that
structure away. It never looks across layers, never uses deposited charge, never splits
merged cores.

Your job: **replace it with something better, re-extract the cluster readout, retrain just
the cluster ensemble, and measure whether the resolution improved.** Ideas (pick one,
measure it):

- **3-D connected components across layers** — treat the shower as one object, not 30
  disjoint slices. The most direct fix.
- **Charge / energy weighting** — don't let a faint single-pixel blip count the same as a
  dense core.
- **Density-based (Molière-aware) core splitting** — separate two overlapping sub-showers
  instead of merging them.
- **A small learned clusterer**, if you're ambitious.

**Success = a lower σ/E curve** (or better linearity) for the Cluster readout than the
baseline, shown side-by-side on the dashboard, with the improvement stated as a number
(e.g. "σ/E at 100 GeV: 0.099 → 0.081, an 18 % relative improvement"). And if your idea
*doesn't* improve it — that's a real result too; say why. That's how research actually goes.

---

## Reference docs

- **`docs/handbook.md`** — the operational + conceptual guide (setup §6, smoke-test sim §8,
  production sim §9, extraction §10, training §11, dashboard §13).
- **`docs/DECAL_pipeline.md`** — the physics reference (why DECAL, the resolution terms).
- **`docs/troubleshooting.md`** — environment quirks and their fixes; check here first when
  something breaks.

Have fun, and don't be afraid to break things in a branch. That's what they're for.
