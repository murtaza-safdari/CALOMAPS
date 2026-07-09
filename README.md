# CALOMAPS — Ultra-High Granularity Digital Calorimeter (DECAL) R&D

A research framework for digital electromagnetic calorimeter (DECAL) studies using simulated Monolithic Active Pixel Sensor (MAPS) silicon layers. CALOMAPS pairs a DD4hep / Geant4 simulation of a custom 100 µm-pitch silicon-tungsten ECal barrel with a PyTorch Deep Quantile Ensemble surrogate model and a Neyman-construction energy reconstruction, producing the standard 3-panel calorimeter performance dashboard (linearity, resolution, stochastic term).

Beyond the reconstruction dashboard, CALOMAPS produces a second data product for sensor-level collaborators: **per-sensor charged-track crossings** of every silicon layer in the shower — impact point, direction, and true per-crossing momentum — exported as ready-to-run input decks for [PIXELAV](docs/pixelav_reference.md) (M. Swartz's detailed silicon-pixel charge-transport simulation). See [docs/pixelav_handoff.md](docs/pixelav_handoff.md) for the end-to-end hand-off recipe.

The pipeline runs end-to-end on the [Fermilab Elastic Analysis Facility (EAF)](https://eaf.fnal.gov). Everything from environment setup through the final physics plot is documented in [docs/handbook.md](docs/handbook.md).

---

## Why ultra-high granularity?

Traditional electromagnetic calorimeters measure analog energy in millimeter-scale cells. A DECAL inverts this: cell sizes are tens of micrometers (here, 100 µm × 100 µm) and the readout is **binary** (hit / no-hit). The bet is that at low shower energy, binary readout truncates the Landau tail of energy deposits and improves resolution; at high energy, pixel saturation makes the digital readout lose information that an analog readout would have kept. CALOMAPS quantifies this trade-off, finds the saturation knee at the current pixel pitch, and provides a framework for pitch / particle-type / geometry scans.

See [docs/DECAL_pipeline.md](docs/DECAL_pipeline.md) for the full physics writeup and [docs/handbook.md](docs/handbook.md) for the operational pipeline.

---

## Pipeline at a glance

```
geometry/SiD_TestBeam.xml + my_custom_ecal.xml
            │
            ▼
sim/run_sim.py  ──>  ddsim  ──>  ROOT files (1000 × 20 events)
                                      │
                                      ▼
                       notebooks/02_data_extraction.ipynb
                                      │
                                      ▼
                   models/decal_extracted_data.npz (~350 KB)
                                      │
                                      ▼
                       analysis/quantilenet.train_one_ensemble
                                      │
                                      ▼
                       models/saved_ensembles_gpu_v2/  (4 ensembles, 20 models each)
                                      │
                                      ▼
                      analysis/dashboard.plot_dashboard
                                      │
                                      ▼
                       3-panel physics dashboard PNG
```

A second product line branches off after simulation — per-sensor track crossings for PIXELAV:

```
sim/make_pixelav_inputs.sh  (one command: sim → extract → deck)
    ├── sim/run_sim_trackermom.py   ──>  ddsim  ──>  SimTrackerHits with true per-crossing momentum
    ├── analysis/extract_trackermom.py   ──>  models/trackermom_<tag><E>_1evt.npz
    └── analysis/pixelav_converter.py    ──>  models/pixelav_segments_*.pixelav.txt  (the PIXELAV deck)
                                              + .json/.csv per-crossing records (16 fields each)
notebooks 04 / 05a / 05b / 05c inspect it;  the pixelav-integration branch builds & runs PIXELAV on it
```

---

## Repository layout

```
CALOMAPS/
├── README.md                          ← you are here
├── docs/
│   ├── handbook.md                    ← OPERATIONAL guide; read this first
│   ├── troubleshooting.md             ← infrastructure quirks (SSHFS, JupyterHub, CVMFS, quotas)
│   ├── DECAL_pipeline.md              ← PHYSICS writeup (advisor's PDF transcribed)
│   ├── pixelav_reference.md           ← PIXELAV itself: what it is, input formats, conventions
│   ├── pixelav_handoff.md             ← end-to-end recipe: produce + hand off the PIXELAV deck
│   └── figures/                       ← rendered dashboards + figures
├── setup/
│   ├── setup_calomaps.sh              ← source from JupyterLab terminal to bootstrap env
│   └── setup_gpu_kernel.sh            ← one-shot: register the GPU kernel for notebooks 03/03b/03d
├── geometry/                          ← DD4hep XML detector descriptions
│   ├── SiD_TestBeam.xml               ← top-level compact passed to ddsim
│   ├── my_custom_ecal.xml             ← our DECAL barrel definition (only modified file)
│   └── baseline_sid_o2_v03/           ← upstream SiD baseline XMLs (untouched, for reference)
│       └── PROVENANCE.md
├── sim/                               ← Geant4 / DD4hep driver scripts
│   ├── run_sim.py                     ← ddsim steering file (particle gun + physics list)
│   ├── run_sim_fullcascade.py         ← + full-cascade truth (detailed calo mode; experiment A)
│   ├── run_sim_trackermom.py          ← + per-crossing momentum (ECal Si as tracker; PIXELAV source)
│   ├── make_pixelav_inputs.sh         ← one command: sim → extract → PIXELAV deck
│   ├── generate_dataset.sh
│   └── generate_batched.sh
├── analysis/                          ← Python ML / reconstruction utilities
│   ├── quantilenet.py                 ← QuantileNet model + training + load/save
│   ├── dashboard.py                   ← Neyman inversion + 3-panel dashboard plotting
│   ├── train_ensembles.py             ← CLI entry point (headless training)
│   ├── verify_ensembles.py            ← CLI entry point (regenerate dashboard)
│   ├── extract_readouts.py            ← headless nb02-equivalent extraction (per-energy datasets)
│   ├── extract_cascade.py             ← full-cascade ROOT → npz (MCParticles + step truth)
│   ├── extract_trackermom.py          ← tracker-readout ROOT → npz (per-crossing momentum)
│   ├── pixelav_converter.py           ← crossings → 7-column PIXELAV deck (+ json/csv records)
│   └── decal_cbfit.py / cbnet.py      ← Crystal-Ball fits / CB-density net (03c/03d)
├── notebooks/                         ← interactive workflow notebooks
│   ├── 00_simulate_your_samples.ipynb ← primer: generate your own samples (any particle via env vars)
│   ├── 01_detector_and_data.ipynb     ← the detector + its data (01b: pion variant)
│   ├── 02_data_extraction.ipynb       ← parallel uproot extraction → .npz (02b: pions)
│   ├── 03_ml_training_and_eval.ipynb  ← train ensembles + dashboard (03b: pions)
│   ├── 03c / 03d                      ← Crystal-Ball resolution studies (conventional / ML)
│   ├── 04_shower_4vectors.ipynb       ← full-cascade MCParticle 4-vectors
│   └── 05a / 05b / 05c                ← PIXELAV inputs: tracker route, calo route, inspection
└── models/                            ← (gitignored) extracted data + trained checkpoints
    ├── decal_extracted_data*.npz
    ├── saved_ensembles_gpu_v2/        ← photon ensembles (trained on the A100 MIG slice)
    └── pixelav_segments_*             ← per-crossing records + PIXELAV decks
```

The 21 GB of raw simulation ROOT files live **outside the repo**, at `$CALOMAPS_DATA_BASE` (defaults to `~/CALOMAPS-data/`). They're regenerated by `sim/generate_batched.sh` and consumed by `notebooks/02_data_extraction.ipynb`.

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

New here? Start with `notebooks/00_simulate_your_samples.ipynb` — it walks you from zero to your own samples (photons by default; any particle via `CALOMAPS_GUN_PARTICLE`), then `01 → 02 → 03` study the reconstruction. For the PIXELAV product line, `04 → 05a/05b → 05c` go from the shower cascade to a validated input deck ([docs/pixelav_handoff.md](docs/pixelav_handoff.md) is the recipe; PIXELAV itself is built and run on the `pixelav-integration` branch).

The ML notebook (`03_ml_training_and_eval.ipynb`) needs a GPU kernel — register it once with `bash $CALOMAPS_HOME/setup/setup_gpu_kernel.sh` (see [docs/handbook.md](docs/handbook.md) §11.2).

The full setup walkthrough — accounts, EAF spawner profile, the storage map, and the GPU torch install — is in [docs/handbook.md](docs/handbook.md).

---

## Provenance

The detector geometry is built on top of the [SiD detector concept](https://confluence.slac.stanford.edu/display/ilc/sidloi) (Silicon Detector — Letter of Intent), specifically the `o2_v03` compact released October 2017 by Norman Graf, Jeremy McCormick, and Dan Protopopescu. CALOMAPS retains the SiD coordinate system, dimensions, and material definitions, but disables every subdetector except the ECal barrel (which is replaced by the custom DECAL definition in `geometry/my_custom_ecal.xml`). See [geometry/baseline_sid_o2_v03/PROVENANCE.md](geometry/baseline_sid_o2_v03/PROVENANCE.md) for the full inheritance documentation.

The deep quantile ensemble approach follows the standard pinball-loss regression pattern; the Neyman-construction inversion is implemented with `scipy.optimize.brentq`, extending the search bracket into the saturation regime and reporting points the readout can never reach as NaN (dropped) rather than clipping them to a fabricated value.

---

## Acknowledgements

This work uses [Key4hep](https://key4hep.github.io/), [DD4hep](https://dd4hep.web.cern.ch/), [Geant4](https://geant4.web.cern.ch/), [uproot](https://uproot.readthedocs.io/), and [PyTorch](https://pytorch.org/). The Elastic Analysis Facility ([EAF](https://eaf.fnal.gov)) at Fermilab provides the GPU compute and software environment.

---

## License

[MIT](LICENSE). See `LICENSE` for the full text.
