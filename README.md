# CALOMAPS — Ultra-High Granularity Digital Calorimeter (DECAL) R&D

> **This is the `pixelav-inputs` branch** — `main` plus the PIXELAV input-deck production
> layer: [`analysis/pixelav_converter.py`](analysis/pixelav_converter.py) (crossing records →
> 7-column PIXELAV deck, with the βγ-matched pion momentum and the verified axis/flip
> conventions), [`sim/make_pixelav_inputs.sh`](sim/make_pixelav_inputs.sh) (one command:
> sim → extract → deck), [`docs/pixelav_handoff.md`](docs/pixelav_handoff.md) +
> [`docs/pixelav_reference.md`](docs/pixelav_reference.md) (the hand-off recipe and the deck
> format / conventions reference), and notebook
> [`08_pixelav_deck_inspection.ipynb`](notebooks/08_pixelav_deck_inspection.ipynb) (validate a
> deck before sending it). Building and *running* PIXELAV itself lives on the
> `pixelav-integration` branch.

A research framework for digital electromagnetic calorimeter (DECAL) studies using simulated Monolithic Active Pixel Sensor (MAPS) silicon layers. CALOMAPS pairs a DD4hep / Geant4 simulation of a custom 100 µm-pitch silicon-tungsten ECal barrel with two deliverables, built up from first principles in the notebooks:

1. **The calorimeter, understood and measured.** Starting from a single simulated shower, the notebooks derive the detector's defining numbers from the geometry and the data themselves — radiation-length and interaction-length budgets, shower max and its ln E scaling, the Molière radius, the silicon MIP scale, the sampling fraction — and then measure the **energy resolution** σ_E/E of four readout definitions two independent ways: the conventional test-beam method (fixed-energy beams + Crystal-Ball fits + calibration inversion) and an ML density model trained on the continuous spectrum. The two methods are overlaid point-by-point across **1–400 GeV**.
2. **Per-sensor charged-track crossings.** For every charged particle at every silicon layer in the shower: the impact position (global and sensor-local), the direction, and the **true per-crossing momentum** — one validated record per crossing, produced from two independent readout routes that are cross-checked against each other. This is the input a detailed sensor-level simulation consumes; this branch converts these records into ready-to-run input decks for PIXELAV, our collaborators' silicon charge-transport simulation (see the banner above).

The pipeline runs end-to-end on the [Fermilab Elastic Analysis Facility (EAF)](https://eaf.fnal.gov). Everything from environment setup through the final physics plot is documented in [docs/handbook.md](docs/handbook.md).

---

## Why ultra-high granularity?

Traditional electromagnetic calorimeters measure analog energy in millimeter-scale cells. A DECAL inverts this: cell sizes are tens of micrometers (here, 100 µm × 100 µm) and the readout is **binary** (hit / no-hit). The bet is that at low shower energy, binary readout truncates the Landau tail of energy deposits and improves resolution; at high energy, pixel saturation makes the digital readout lose information that an analog readout would have kept. CALOMAPS quantifies this trade-off, finds the saturation knee at the current pixel pitch, and provides a framework for pitch / particle-type / geometry scans.

See [docs/DECAL_pipeline.md](docs/DECAL_pipeline.md) for the physics design document and [docs/handbook.md](docs/handbook.md) for the operational pipeline.

---

## Pipeline at a glance

```
geometry/SiD_TestBeam.xml + my_custom_ecal.xml
            │
            ▼
sim/run_sim.py  ──>  ddsim  ──>  ROOT files
     │                             │
     │ spectrum run                │ fixed-energy runs (1–400 GeV)
     ▼                             ▼
notebooks/02_data_extraction     analysis/extract_readouts.py
     │                             │
     ▼                             ▼
models/decal_extracted_data*.npz  models/mono_gamma/*.npz
     │                             │
     ▼                             ▼
notebook 04: ML CB-density net   notebook 03: Crystal-Ball fits
(analysis/cbnet.py, GPU)         + calibration inversion (CPU)
     └───────────┬────────────────┘
                 ▼
   σ_E/E vs E — two independent methods, overlaid across 1–400 GeV
```

A second product line branches off after simulation — per-sensor track crossings:

```
sim/run_sim_trackermom.py  ──> ddsim ──> SimTrackerHits (true per-crossing momentum)  [route C, primary]
    └── analysis/extract_trackermom.py ──> models/trackermom_<tag><E>_1evt.npz ─┐
sim/run_sim_fullcascade.py ──> ddsim ──> per-step CaloHitContributions           │  [route A, cross-check]
    └── analysis/extract_cascade.py    ──> models/fullcascade_<tag><E>_1evt.npz ──┤  (also feeds notebook 05)
                                                                                  ▼
                                          analysis/sensor_crossings.py ──> models/sensor_crossings_*.{json,csv}
                                            (one record per charged-track sensor crossing:
                                             impact point, direction, |p|, particle type)
notebooks 05 / 06 / 07 build up and validate it;  the pixelav-inputs branch converts the
records into PIXELAV input decks, and pixelav-integration builds & runs PIXELAV on them
```

---

## Repository layout

```
CALOMAPS/
├── README.md                          ← you are here
├── docs/
│   ├── handbook.md                    ← OPERATIONAL guide; read this first
│   ├── troubleshooting.md             ← infrastructure quirks (SSHFS, JupyterHub, CVMFS, quotas)
│   ├── DECAL_pipeline.md              ← PHYSICS design document (motivation, method, expected results)
│   ├── pixelav_handoff.md             ← [this branch] end-to-end deck hand-off recipe
│   ├── pixelav_reference.md           ← [this branch] PIXELAV deck format + conventions
│   └── figures/                       ← (generated output; the notebooks render inline)
├── setup/
│   ├── setup_calomaps.sh              ← source from JupyterLab terminal to bootstrap env
│   └── setup_gpu_kernel.sh            ← one-shot: register the GPU kernel for notebook 04
├── geometry/                          ← DD4hep XML detector descriptions
│   ├── SiD_TestBeam.xml               ← top-level compact passed to ddsim
│   ├── my_custom_ecal.xml             ← our DECAL barrel definition (only modified file)
│   └── baseline_sid_o2_v03/           ← upstream SiD baseline XMLs (untouched, for reference)
│       └── PROVENANCE.md
├── sim/                               ← Geant4 / DD4hep driver scripts
│   ├── run_sim.py                     ← ddsim steering file (particle gun + physics list)
│   ├── run_sim_fullcascade.py         ← + full-cascade truth (detailed calo mode)
│   ├── run_sim_trackermom.py          ← + per-crossing momentum (ECal Si read as a tracker)
│   ├── make_pixelav_inputs.sh         ← [this branch] one command: sim → extract → PIXELAV deck
│   ├── generate_dataset.sh
│   └── generate_batched.sh
├── analysis/                          ← Python utilities
│   ├── decal_cbfit.py                 ← shared Crystal-Ball fitter + calibration inversion (03/04)
│   ├── cbnet.py                       ← Crystal-Ball density network (notebook 04)
│   ├── extract_readouts.py            ← headless per-energy extraction (notebook 03 inputs)
│   ├── extract_cascade.py             ← full-cascade ROOT → npz (MCParticles + step truth)
│   ├── extract_trackermom.py          ← tracker-readout ROOT → npz (per-crossing momentum)
│   ├── sensor_crossings.py            ← crossings → per-crossing records (.json/.csv)
│   ├── make_config_ab.py              ← regenerates notebook 05's persistency A/B panel input
│   ├── quantilenet.py / dashboard.py / train_ensembles.py / verify_ensembles.py
│   │                                  ← legacy quantile-regression surrogate (kept for reference)
│   └── pixelav_converter.py           ← [this branch] crossing records → 7-column PIXELAV deck
├── notebooks/                         ← interactive workflow notebooks (read in order)
│   ├── 00_simulate_your_samples.ipynb ← primer: generate your own samples (any particle via env vars)
│   ├── 01_detector_and_data.ipynb     ← the detector from first principles (01b: pion contrast)
│   ├── 02_data_extraction.ipynb       ← parallel uproot extraction → .npz
│   ├── 03_resolution_conventional.ipynb   ← fixed-energy Crystal-Ball fits + inversion
│   ├── 04_resolution_ml_crystalball.ipynb ← ML CB-density net, overlaid on 03 (GPU)
│   ├── 05_shower_4vectors.ipynb       ← full-cascade MCParticle 4-vectors
│   ├── 06_sensor_crossings_tracker.ipynb  ← per-sensor crossings: tracker route (primary)
│   ├── 07_sensor_crossings_calo.ipynb     ← per-sensor crossings: calorimeter route (cross-check)
│   └── 08_pixelav_deck_inspection.ipynb  ← [this branch] validate a PIXELAV deck before sending it
└── models/                            ← (gitignored) extracted data + trained checkpoints
    ├── decal_extracted_data*.npz  /  mono_gamma/*.npz
    ├── saved_cbnet_gpu_*/             ← CB-density ensembles (trained on the A100)
    └── sensor_crossings_*             ← per-crossing record tables (.json/.csv)
```

The raw simulation ROOT files live **outside the repo**, at `$CALOMAPS_DATA_BASE` (defaults to `~/CALOMAPS-data/`). They're regenerated by `sim/generate_batched.sh` and consumed by `notebooks/02_data_extraction.ipynb`.

---

## Quick start (on EAF)

```bash
# In a JupyterLab terminal:
cd /nashome/${USER:0:1}/$USER/
git clone https://github.com/murtaza-safdari/CALOMAPS.git
ln -s /nashome/${USER:0:1}/$USER/CALOMAPS/setup/setup_calomaps.sh ~/setup_calomaps.sh
source ~/setup_calomaps.sh
```

`setup_calomaps.sh` is safe to source repeatedly; it creates the `~/lib_hack` OpenGL shim and the data directory for you. Confirm your environment with the smoke test in `docs/handbook.md` §8.

New here? Start with `notebooks/00_simulate_your_samples.ipynb` — it walks you from zero to your own samples (photons by default; any particle via `CALOMAPS_GUN_PARTICLE`) — then read `01 → 02 → 03 → 04` in order: they build the calorimetry up from first principles and end with the resolution measured two independent ways. For the sensor-level product, `05 → 06/07` go from the full shower cascade to the validated per-crossing records (PIXELAV input decks are produced on this branch (see docs/pixelav_handoff.md and notebook 08)).

The ML notebook (`04_resolution_ml_crystalball.ipynb`) needs a GPU kernel — register it once with `bash $CALOMAPS_HOME/setup/setup_gpu_kernel.sh` (see [docs/handbook.md](docs/handbook.md) §11.2).

The full setup walkthrough — accounts, EAF spawner profile, the storage map, and the GPU torch install — is in [docs/handbook.md](docs/handbook.md).

---

## Provenance

The detector geometry is built on top of the [SiD detector concept](https://confluence.slac.stanford.edu/display/ilc/sidloi) (Silicon Detector — Letter of Intent), specifically the `o2_v03` compact released October 2017 by Norman Graf, Jeremy McCormick, and Dan Protopopescu. CALOMAPS retains the SiD coordinate system, dimensions, and material definitions, but disables every subdetector except the ECal barrel (which is replaced by the custom DECAL definition in `geometry/my_custom_ecal.xml`). See [geometry/baseline_sid_o2_v03/PROVENANCE.md](geometry/baseline_sid_o2_v03/PROVENANCE.md) for the full inheritance documentation.

The resolution analysis quotes the Gaussian-core width of a Crystal-Ball fit to the response (the standard treatment of a leakage-tailed calorimeter response), turned into an energy resolution by inverting the response calibration — measured point-by-point in the conventional analysis, learned from the spectrum by the ML density network. Inversions extend the search bracket into the saturation regime and report points the readout can never reach as NaN (dropped) rather than clipping them to a fabricated value.

---

## Acknowledgements

This work uses [Key4hep](https://key4hep.github.io/), [DD4hep](https://dd4hep.web.cern.ch/), [Geant4](https://geant4.web.cern.ch/), [uproot](https://uproot.readthedocs.io/), and [PyTorch](https://pytorch.org/). The Elastic Analysis Facility ([EAF](https://eaf.fnal.gov)) at Fermilab provides the GPU compute and software environment.

---

## License

[MIT](LICENSE). See `LICENSE` for the full text.
