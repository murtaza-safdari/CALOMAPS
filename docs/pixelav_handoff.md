# Generating PIXELAV inputs — a from-scratch hand-off recipe

This is the end-to-end recipe for producing the **PIXELAV input package** that CALOMAPS
hands to collaborators running their own (custom) PIXELAV. It starts from a brand-new EAF
session and ends with a per-sensor track-crossing **deck** ready to feed a single-sensor
charge-transport simulation.

For the conceptual background — what the tracker sensitive detector records vs the
calorimeter one, the 7-column deck format, and the per-crossing record schema — see
notebook [`05a_pixelav_inputs_tracker.ipynb`](../notebooks/05a_pixelav_inputs_tracker.ipynb)
(§7 is the data-structure spec) and [`pixelav_reference.md`](pixelav_reference.md). This
document is the *operational* recipe.

---

## 0. What you will produce

For a chosen particle + energy, the pipeline writes (ROOT under `$CALOMAPS_DATA_BASE`, the
rest under `$CALOMAPS_HOME/models`):

| File | What it is |
|------|------------|
| `trackermom_<tag><E>_1evt.root` | tracker-SD readout: one `SimTrackerHit` per Si crossing, carrying the **real Geant4 momentum**, entry position, path length, energy, time, and a link to the producing MCParticle |
| `trackermom_<tag><E>_1evt.npz` | the same, plus the full MCParticle cascade, as NumPy arrays |
| **`pixelav_segments_<tag><E>_1evt.pixelav.txt`** | **the deck** — one line per crossing, 7 columns (`cot_alpha cot_beta ppion flipped modx mody pT`), lengths in microns |
| `..._1evt.pixelav.txt.columns.txt` | column legend (human reference; PIXELAV doesn't read it) |
| `..._1evt.json` / `.csv` | per-crossing records, 16 fields each (provenance + the raw numbers behind the deck) |

`<tag>` is `gamma`, `piplus`, `piminus`, … and `<E>` is the beam energy in GeV (e.g.
`trackermom_piplus50_1evt.root`). The deck is the hand-off file; the `.npz`/`.json`/`.csv`
travel with it so the recipient can trace any line back to its MCParticle.

> **All particles, no filtering (current default).** Every charged crossing is written.
> Summing PIXELAV's per-crossing charge over a whole shower over-counts the deposited
> charge (PIXELAV regenerates delta rays that Geant4 already produced as separate
> crossings) — the per-crossing physics is correct; it is the *shower sum* that needs a
> filter. The double-counting and the proposed filter are documented in notebook 06; the
> filter is **not** applied here.

---

## 1. First EAF session (from scratch)

The full walkthrough — FNAL/EAF accounts, the spawner profile, the storage map — is in
[`handbook.md`](handbook.md) §6. The short version:

```bash
# 1. Log in at https://eaf.fnal.gov and spawn a GPU (non-CMS) profile -> JupyterLab.
# 2. In a JupyterLab terminal, one-time per account:
cd /nashome/${USER:0:1}/$USER/
git clone https://github.com/murtaza-safdari/CALOMAPS.git
ln -s /nashome/${USER:0:1}/$USER/CALOMAPS/setup/setup_calomaps.sh ~/setup_calomaps.sh
# 3. Every session:
source ~/setup_calomaps.sh        # loads Key4hep, auto-creates the ~/lib_hack OpenGL shim,
                                  # chmod +x sim/*.sh, sets CALOMAPS_HOME / CALOMAPS_DATA_BASE
```

Confirm the environment with the 30-second smoke test in handbook §8 before generating
real inputs. (No GPU kernel is needed for the PIXELAV-input pipeline — that's only for the
ML notebook 03.)

---

## 2. One command: simulate → extract → deck

[`sim/make_pixelav_inputs.sh`](../sim/make_pixelav_inputs.sh) runs the whole chain (it
replaces the five manual steps that used to live in handbook §10.1):

```bash
# photons at 50 GeV (the default):
bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh

# any other particle / energy — just set the same env vars used everywhere else:
CALOMAPS_GUN_PARTICLE=pi+ bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh
CALOMAPS_GUN_PARTICLE=pi- CALOMAPS_GUN_ENERGY_GEV=80 bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh

# add the calorimeter cascade too (for the nb04 shower display / nb05b calo-SD view):
CALOMAPS_GUN_PARTICLE=pi+ bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh --fullcascade
```

What it does, in order:

1. **tracker-SD simulation** — `ddsim` with `run_sim_trackermom.py`, which maps the ECal Si
   to `Geant4TrackerWeightedAction` so each sensor crossing becomes a `SimTrackerHit` with
   the real momentum (one event; the shower is fully contained in the MCParticle tree).
2. **extract** — `extract_trackermom.py` reads the ROOT into the `.npz` arrays.
3. **convert** — `pixelav_converter.py` builds per-crossing records (variant C, the
   tracker path) and writes the 7-column deck + the `.json`/`.csv` records.

The script prints the exact output paths at the end.

---

## 3. The deck, briefly

Each line is one Si crossing (full spec in nb05a §7 and `pixelav_reference.md`):

```
cot_alpha  cot_beta  ppion  flipped  modx  mody  pT
```

- `cot_alpha`, `cot_beta` — the track direction relative to the sensor normal.
- `ppion` — a **βγ-matched pion momentum**: `p · (m_pi / m_particle)`, so PIXELAV's
  pion-only Bichsel dE/dx reproduces the *real* particle's ionization. For pions this
  equals the real momentum; for electrons it is ~273× larger (electrons are ~273× lighter).
- `flipped` — 1 if the crossing direction is outward (dw ≥ 0).
- `modx`, `mody` — the sub-pixel **mid-plane** entry point in microns, on PIXELAV's axes
  (`modx` = cylinder-z, `mody` = across-pitch). Grazing crossings (`|cot| > 10`) are skipped,
  as PIXELAV itself skips them.
- `pT` — the real momentum magnitude (GeV), carried as a label.

Stock PIXELAV randomizes the sub-pixel impact; `modx`/`mody` are the real impact for the
patched `ppixelav2_list_trkpy_real_entry` driver (see [`pixelav_journey_log.md`](pixelav_journey_log.md)).

---

## 4. Handing it off

Give your collaborator the deck plus its companions:

```
pixelav_segments_<tag><E>_1evt.pixelav.txt           <- feed this to PIXELAV
pixelav_segments_<tag><E>_1evt.pixelav.txt.columns.txt
pixelav_segments_<tag><E>_1evt.json / .csv           <- per-crossing provenance
trackermom_<tag><E>_1evt.npz                         <- full MCParticle + tracker arrays
```

They run it through their PIXELAV with the DECAL sensor model (320 µm Si, 100 µm pitch;
`analysis/pixelav/make_decal_stagea.py` generates the Stage-A field + weighting-potential
files, and `setup/setup_pixelav.sh` builds a reference PIXELAV + the patched driver if they
want a local cross-check). Each deck line yields one simulated cluster.

To validate a deck before sending it, open the **PIXELAV-input inspection notebook**
([`05c_pixelav_input_inspection.ipynb`](../notebooks/05c_pixelav_input_inspection.ipynb)) — it
reads a `.npz`/deck and walks from basic sanity checks up to the per-crossing momentum,
impact-point, and angle distributions that justify every column. The full per-crossing record
schema lives in [`pixelav_reference.md`](pixelav_reference.md).
