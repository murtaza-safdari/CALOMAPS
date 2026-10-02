# Generating PIXELAV inputs — a from-scratch hand-off recipe

This is the end-to-end recipe for producing the **PIXELAV inputs** that CALOMAPS hands to
collaborators running their own (custom) PIXELAV, and for running PIXELAV on them here. It starts
from a brand-new session and ends with either a per-sensor track-crossing **deck** or a set of
**deposit jobs**, and the PIXELAV clusters they give.

For the conceptual background — what the tracker sensitive detector records vs the
calorimeter one, the 7-column deck format, and the per-crossing record schema — see
notebook [`06_sensor_crossings_tracker.ipynb`](../notebooks/06_sensor_crossings_tracker.ipynb)
(§7 is the record-schema spec) and [`pixelav_reference.md`](pixelav_reference.md) (§3b is the
deck-format spec, §3c the deposit-job format, §4 the cluster files, §6 the frames, output keys
and limits). This document is the *operational* recipe.

---

## 0. What you will produce

There are two routes from the Geant4 shower to PIXELAV, chosen with `CALOMAPS_PIXELAV_ROUTE`.

- **`track`** (the default) writes one deck line per sensor crossing. PIXELAV then generates its
  own ionisation along a straight track through the full silicon thickness. Use it for clusters
  of through-going charged tracks, and to hand a deck to a collaborator who runs PIXELAV.
- **`deposits`** hands PIXELAV every Geant4 step in the silicon, and PIXELAV only drifts and reads
  out that charge. Use it for the charge a whole shower leaves in the pixels, including the soft
  electrons that start, stop or curl inside the sensor: in 50 GeV photon showers at 100 µm, a
  quarter of the silicon energy comes from electrons below 0.1 MeV/c. It costs far more (§5.3).

On 1401 stiff through-going crossings of the same showers (100 µm silicon, the 50 × 25 µm
model) the two routes agree: median charge 8490 e vs 8478 e, mean pixels above 1000 e 2.01 vs
2.00.

For a chosen particle + energy, the **track route** writes (ROOT under
`$CALOMAPS_DATA_BASE/trackermom/`, the rest under `$CALOMAPS_OUTPUT_DIR`, default
`$CALOMAPS_HOME/models`):

| File | What it is |
|------|------------|
| `trackermom_<tag><E>_<N>evt<si>.root` | tracker-SD readout: one `SimTrackerHit` per Si crossing, carrying the **real Geant4 momentum**, position, path length, energy, time, cellID, and a link to the producing MCParticle |
| `trackermom_<tag><E>_<N>evt<si>.npz` | the same, plus the MCParticle cascade, as NumPy arrays (for N > 1 only the particles with a tracker hit) |
| **`pixelav_segments_<tag><E>_<N>evt<si>.pixelav.txt`** | **the deck** — one line per selected crossing, 7 columns (`cot_alpha cot_beta ppion flipped modx mody pT`), lengths in microns |
| `..._<N>evt<si>.pixelav.txt.columns.txt` | legend: column definitions, axis pairing, selection counts per reason, and the `# sensor: <T> um silicon` line (PIXELAV doesn't read it) |
| `..._<N>evt<si>.json` / `.csv` | per-crossing records, every crossing, 20 fields each (provenance + the raw numbers behind the deck; `deck_line` is the record's deck line or 0, `deck_drop` why it has none) |

The **deposits route** writes:

| File | What it is |
|------|------------|
| `trackermom_<tag><E>_<N>evt_step<si>.root` | tracker-SD readout with one `SimTrackerHit` per Geant4 step with a deposit in the silicon |
| `trackermom_<tag><E>_<N>evt_step<si>.npz` | the step hits and the MCParticles that have one (with their charge and parent links) |
| **`pixelav_jobs_<tag><E>_<N>evt<si>.jobs.txt`** | **the deposit jobs** (or `.part000.txt`, `.part001.txt`, … with `CALOMAPS_JOB_PARTS`); format in `pixelav_reference.md` §3c |
| `pixelav_jobs_..._<N>evt<si>.jobs.npz` | per-job and per-group tables, the frame constants and the provenance (keys in `pixelav_reference.md` §6) |
| `pixelav_jobs_..._<N>evt<si>.summary.txt` | counts, the energy closure and the window check |

`<tag>` is `gamma`, `piplus`, `piminus`, …, `<E>` is the beam energy in GeV and `<N>` the number
of events, `CALOMAPS_N_EVENTS` (default 1). `<si>` names the silicon: empty for the repo geometry,
otherwise `_si<T>um` (another thickness), `_cut<C>um` (a production cut in the silicon) and
`_stepmax<S>um` (a step limit in the silicon), in that order. The deposits route always has a cut
and a step limit, so its names carry at least `_cut1um_stepmax5um`. Examples:
`trackermom_piplus50_1evt.root`,
`pixelav_jobs_gamma50_1evt_si100um_cut1um_stepmax5um.jobs.txt`. The deck or the jobs are the
hand-off file; the `.npz`/`.json`/`.csv` travel with them so the recipient can trace any line back
to its MCParticle.

> **Which crossings the deck holds.** PIXELAV simulates every deck line as a straight track
> through the full silicon thickness at constant momentum. By default
> (`CALOMAPS_DECK_SELECT=through`) the deck holds only the crossings whose true Geant4 path in
> the silicon is within 5% (`CALOMAPS_PATH_TOLERANCE=0.05`) of that straight line; soft
> electrons that start, stop, scatter or curl in the sensor are left out, and so are tracks
> steeper than |cot| = 10, which PIXELAV skips. In 50 GeV photon showers that keeps 65% of the
> crossings, carrying 44% of the silicon energy, at 100 µm, and 54% carrying 41% at 320 µm (the shares
> move by a few percent from one shower to the next).
> `CALOMAPS_DECK_SELECT=all` writes every crossing PIXELAV processes. The legend lists the counts
> per reason; the `.json`/`.csv` records keep every crossing (`deck_line`, `deck_drop`).
> A deck is for per-crossing clusters. Its clusters do not add up to the charge the shower leaves
> in the silicon: the deck leaves crossings out, and PIXELAV generates its own ionisation, delta
> rays included, along every track it keeps. The deposits route gives the shower's charge.

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
ML notebook 04 (04_resolution_ml_crystalball).)

Nothing in this chain depends on EAF. It needs the CVMFS Key4hep stack (`ddsim`, `gcc`, and
`python3` with numpy); the scripts also run on an LPC login node, and `setup/setup_pixelav.sh`
also builds on macOS.

---

## 2. One command: simulate → extract → deck or jobs

[`sim/make_pixelav_inputs.sh`](../sim/make_pixelav_inputs.sh) runs the whole chain (it
is a one-command alternative to the manual chain in handbook §10.1 — that chain ends at the
validated crossing records via `analysis/sensor_crossings.py`; this script runs the tracker
leg and continues to the deck or the jobs, and `--fullcascade` adds the calorimeter leg):

```bash
# photons at 50 GeV (the default):
bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh

# any other particle / energy — just set the same env vars used everywhere else:
CALOMAPS_GUN_PARTICLE=pi+ bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh
CALOMAPS_GUN_PARTICLE=pi- CALOMAPS_GUN_ENERGY_GEV=80 bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh

# add the calorimeter cascade too (for the notebook-05 shower display / notebook-07 calo view):
CALOMAPS_GUN_PARTICLE=pi+ bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh --fullcascade

# 100 um silicon (pairs with the bpix100_50x25 sensor model, §5.1):
CALOMAPS_SI_THICKNESS_UM=100 bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh

# 200 events, another seed, outputs outside the home quota:
CALOMAPS_N_EVENTS=200 CALOMAPS_SEED=7 CALOMAPS_OUTPUT_DIR=/path/to/scratch \
    bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh

# every crossing PIXELAV processes, not only those its straight track reproduces (box in §0):
CALOMAPS_DECK_SELECT=all bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh

# the deposits route, 100 um silicon, jobs for a built PIXELAV run directory, in 30 parts:
CALOMAPS_PIXELAV_ROUTE=deposits CALOMAPS_SI_THICKNESS_UM=100 CALOMAPS_N_EVENTS=20 \
CALOMAPS_JOB_PARTS=30 PIXELAV_RUN_DIR=/tmp/pixelav_journey/pixelav/bpix100_50x25_run \
    bash $CALOMAPS_HOME/sim/make_pixelav_inputs.sh
```

The knobs (environment variables; all optional except the deposits route's sensor):

| Knob | Default | What it sets |
|------|---------|--------------|
| `CALOMAPS_PIXELAV_ROUTE` | `track` | `track` or `deposits` (§0) |
| `CALOMAPS_GUN_PARTICLE` | `gamma` | gun particle (`pi+`, `pi-`, `proton`, …) |
| `CALOMAPS_GUN_ENERGY_GEV` | `50` | beam energy in GeV (legacy alias `CALOMAPS_GUN_GEV`) |
| `CALOMAPS_N_EVENTS` | `1` | events to simulate; every name carries `_<N>evt`. Above 1 the npz keeps only the MCParticles with a tracker hit |
| `CALOMAPS_SEED` | `424242` (the steering files') | random seed, a non-negative integer; any seed other than the default adds `_seed<R>` after `_<N>evt` in every output name, and the extractors refuse a file whose `_seed<R>` is not the seed in its run header |
| `CALOMAPS_OUTPUT_DIR` | `$CALOMAPS_HOME/models` | directory for the npz, records, deck and jobs. 50 GeV photons take about 9 MB per event there with every MCParticle, 3.7 MB with only the hit particles |
| `CALOMAPS_DATA_BASE` | `~/CALOMAPS-data` | root of the ROOT output and of the geometry variants |
| `CALOMAPS_SI_THICKNESS_UM` | the repo geometry's, 320 | sensor thickness in µm; another value simulates a variant geometry (tag `_si<T>um`) |
| `CALOMAPS_SI_CUT_UM` | track: none (the run's own cut); deposits: `1` | Geant4 production cut in the sensor silicon only, in µm, so delta rays above about that range are tracked (tag `_cut<C>um`) |
| `CALOMAPS_SI_STEP_UM` | track: none (the calorimeter's 5 mm); deposits: `5` | largest Geant4 step in the sensor silicon, in µm (tag `_stepmax<S>um`) |
| `CALOMAPS_DECK_SELECT` | `through` | track route: `through` or `all` (box in §0) |
| `CALOMAPS_PATH_TOLERANCE` | `0.05` | track route: largest \|true path / straight chord − 1\| of a `through` deck |
| `PIXELAV_RUN_DIR` | none | deposits route: a PIXELAV run directory (§5.1); its `ppixel2.init` gives the pitches and the thickness, which must be within 1 µm of the simulated silicon's |
| `CALOMAPS_PIXEL_X_UM`, `CALOMAPS_PIXEL_Y_UM` | none | deposits route, instead of `PIXELAV_RUN_DIR`: the pitches along PIXELAV x (= v, the barrel axis) and y (= u, across the stave). One of the two is required; the run stops before simulating without |
| `CALOMAPS_JOB_PARTS` | `1` | deposits route: write the jobs in this many files of about equal energy |
| `CALOMAPS_PY` | `python3` | interpreter for the extraction, validation and conversion steps |

The script sets `CALOMAPS_TRACKER_HITS` from the route (`crossing` for track, `step` for
deposits) and stops if it is already set to the other value. The steering file's own knobs
(`CALOMAPS_RANGECUT_MM`, `CALOMAPS_KEEP_ALL`, `CALOMAPS_MIN_KE_MEV`) pass through; see handbook
§10.2. Code and geometry come from the checkout that holds the script; it warns when
`$CALOMAPS_HOME` names another directory.

What it does, in order:

1. **tracker-SD simulation** — `ddsim` with `run_sim_trackermom.py`, which maps the ECal Si
   to `Geant4TrackerWeightedAction` (`CALOMAPS_N_EVENTS` events; `CALOMAPS_SEED` sets the seed).
   Track route: each sensor crossing becomes one `SimTrackerHit` with the real momentum, at the
   midpoint of its first and last step points. Deposits route: every Geant4 step with a deposit
   in the silicon becomes one hit (step midpoint, mean momentum, step length, deposit), in a
   variant geometry whose silicon has the 1 µm cut and the 5 µm step limit. The deposits route
   first checks the sensor (`PIXELAV_RUN_DIR` or the pitches) against the silicon thickness, so
   a mismatch stops it before the simulation.
2. **extract** — `extract_trackermom.py` reads every event of the ROOT file into the `.npz`
   arrays (only the MCParticles with a tracker hit for more than one event, and always in the
   deposits route). Each hit's face and layer come from its cellID (`tmodule`, `tlayer`): the
   barrel staves overlap past the 30° sector boundaries, so a hit's azimuth does not always name
   its face.
3. **validate** — `validate_crossings.py` checks the npz against the silicon thickness it
   claims: containment of the hits in their slab, plus the path lengths of clean crossings
   (track route) or the depths of the steps (deposits route). A failure stops the run with no
   deck or jobs written.
4. **convert** — track route: `pixelav_converter.py` builds per-crossing records (variant C, the
   tracker path) and writes the 7-column deck of the selected crossings + the `.json`/`.csv`
   records of all of them. Deposits route: `analysis/pixelav/deposit_jobs.py` writes the jobs, the
   tables and the summary (§4).

With `--fullcascade` it also runs the calorimeter cascade, always one event:
`fullcascade_<tag><E>_1evt<si>.root` + `.npz`. The script prints the exact output paths at the
end.

---

## 3. The deck, briefly

Each line is one Si crossing (deck spec in `pixelav_reference.md` §3b; record schema in notebook 06 §7):

```
cot_alpha  cot_beta  ppion  flipped  modx  mody  pT
```

- `cot_alpha`, `cot_beta` — the track direction relative to the sensor normal.
- `ppion` — a **βγ-matched pion momentum**: `p · (m_pi / m_particle)`, so PIXELAV's
  pion-only Bichsel dE/dx reproduces the *real* particle's ionization. For pions this
  equals the real momentum; for electrons it is ~273× larger (electrons are ~273× lighter).
- `flipped` — 1 if the crossing direction is outward (dw ≥ 0).
- `modx`, `mody` — the sub-pixel **mid-plane** impact point in microns, on PIXELAV's axes
  (`modx` = v, along the barrel axis; `mody` = u, across the stave). Steep crossings
  (`|cot| > 10`) are skipped, as PIXELAV itself skips them; the default selection leaves out
  more (box in §0).
- `pT` — the real momentum magnitude (GeV), carried as a label.

Stock PIXELAV randomizes the sub-pixel impact; `modx`/`mody` are the real impact for the
patched `ppixelav2_list_trkpy_real_entry` driver in `analysis/pixelav/` (build and checks in
[`pixelav_journey_log.md`](pixelav_journey_log.md)). The pixel pitch PIXELAV simulates is the
sensor model's, not the calorimeter readout's.

---

## 4. The deposit jobs, briefly

`analysis/pixelav/deposit_jobs.py` turns the step npz into jobs for
`ppixelav2_inject_deposits` (format in `pixelav_reference.md` §3c):

- **Segments.** Each step becomes a straight segment centred on the hit position, along its mean
  momentum, as long as its step length, clipped in depth to the silicon slab, and cut into pieces
  of at most 5 µm (`--piece-um`). With the 5 µm step limit, consecutive segments of a charged
  particle meet within 0.3 µm for 90% of its steps.
- **Groups.** The steps of one independent charged particle in one sensor plane (event, cellID
  module, cellID layer) form a group; delta rays made inside the slab join the track that made
  them, and a photon converting in the silicon gives two groups.
- **Jobs.** Every sensor plane has one pixel grid, covered by tiles of 17 × 9 pixels. A job is one
  group inside one tile; its 21 × 13 window is the tile plus a 2-pixel guard band, which holds the
  charge that diffuses out. A job above 100,000 pairs (`--max-pairs`) is split into jobs with the
  same window; PIXELAV's carriers do not interact, so this is exact.

Nothing is written unless the jobs' energy equals the steps' and every piece lies within its
tile. Run by hand:

```bash
python3 analysis/pixelav/deposit_jobs.py <step npz> <prefix> --run-dir <PIXELAV run dir> [--parts N]
python3 analysis/pixelav/deposit_jobs.py <step npz> <prefix> --pitch-x-um 50 --pitch-y-um 25 \
        [--thick-um T] [--piece-um 5] [--max-pairs 100000] [--parts N] [--events A:B]
```

---

## 5. Running PIXELAV

### 5.1 Build PIXELAV and a sensor model

```bash
bash setup/setup_pixelav.sh                                     # decal320 sensor model
PIXELAV_SENSOR=bpix100_50x25 PIXELAV_SENSOR_DIR=<dir> bash setup/setup_pixelav.sh
```

The script clones upstream PIXELAV (the public mirror, `PIXELAV_UPSTREAM`) at the pinned commit
`8a774a3` (2024-05-13), and refuses an existing clone at another commit or with local changes to
the files it compiles. It builds three drivers: the baseline `ppixelav2_list_trkpy_n_2f`, the
patched list driver `ppixelav2_list_trkpy_real_entry` (track route), and
`ppixelav2_inject_deposits` (deposits route). It installs the chosen sensor model in a run
directory, prints a one-line summary of it (`sensor_model.py --summary`), refuses a model that is
incomplete, whose two files disagree, or whose B field is not zero, and runs a one-track and a
one-job smoke test.

| Knob | Default | What it sets |
|------|---------|--------------|
| `PIXELAV_WORK` | `/tmp/pixelav_journey` | work area. `/tmp` is wiped on an EAF container restart; point this at a lasting disk, or re-run the script |
| `PIXELAV_SENSOR` | `decal320` | the sensor model: `decal320` or `bpix100_50x25` |
| `PIXELAV_SENSOR_DIR` | none | directory with the `bpix100_50x25` model's `ppixel2.init` and `wgt_pot.init` (required for it) |
| `PIXELAV_ALLOW_BFIELD` | `0` | `1` accepts a model with a non-zero B field (the CALOMAPS simulation has none) |
| `PIXELAV_PY` | `python3` on `PATH` | interpreter for the `decal320` generator, run with `PYTHONPATH` cleared (e.g. the JupyterLab image's own `python3`, by its full path, from a Key4hep-sourced shell) |
| `PIXELAV_UPSTREAM` | `https://github.com/badeaa3/pixelav.git` | where to clone from (a local clone works); the pinned commit is checked out either way |

The two sensor models:

| `PIXELAV_SENSOR` | Run directory | Model | Pairs with |
|---|---|---|---|
| `decal320` | `$PIXELAV_WORK/pixelav/decal_run/` | generated by `analysis/pixelav/make_decal_stagea.py`: 320 µm thick, 100 × 100 µm pixels, uniform drift field (10 kV/cm), B = 0, 300 K, sampling time 12 ns (the charge needs about 9.5 ns to arrive) | the default 320 µm geometry |
| `bpix100_50x25` | `$PIXELAV_WORK/pixelav/bpix100_50x25_run/` | a TCAD-derived field map of a 100 µm thick planar sensor with 50 × 25 µm pixels, 253 K, a 26 × 26 × 51 grid and a zeroed B-field line. **Not in the repository**: obtain its two files separately; the script checks them against sha256 sums before installing them | `CALOMAPS_SI_THICKNESS_UM=100` |

Both run directories hold `SIRUTH.SPR`, `ppixel2.init`, `wgt_pot.init` and both CALOMAPS
drivers. The drivers need a large stack: `ulimit -s unlimited` (macOS: `ulimit -s 65520`).

### 5.2 Track route: the list driver

```bash
RUN=/tmp/pixelav_journey/pixelav/decal_run
DECK=$CALOMAPS_HOME/models/pixelav_segments_gamma50_1evt.pixelav.txt
python3 analysis/pixelav/sensor_model.py --check-deck $DECK $RUN
cp $DECK $RUN/track_list.txt
cd $RUN && ulimit -s unlimited && ./ppixelav2_list_trkpy_real_entry 1 track_list.txt clusters.out seedfile.txt
python3 $CALOMAPS_HOME/analysis/pixelav/parse_cluster.py clusters.out
```

- `sensor_model.py --check-deck` compares the model's thickness with the legend's
  `# sensor: <T> um silicon` line and fails on a mismatch (`--assume-deck-thickness-um T` for a
  legend without it). It then prints how the driver pairs the deck with the model and counts the
  deck lines whose track inside the silicon is longer than the window; the driver drops charge
  outside its window, so those clusters can be cut.
- The arguments are `<block> <deck> <output> <seed file>`. One run takes 30000 deck lines: block 1
  takes lines 1–30000, block 2 lines 30001–60000, and so on. For a longer deck the driver says so
  and names the next block; run each block with its own output and seed file (only block 1's
  output starts with the two header lines).
- The driver saves its random-number state and its event counter in the seed file after every
  deck line. **A seed file left at a non-zero counter, or an existing output file, stops the run**
  unless `PIXELAV_RESUME=1`, which continues from that counter and appends to the output. A
  finished deck resumed this way writes nothing. To run a deck again, remove both files.
- A deck line skipped for `|cot_alpha|` or `|cot_beta|` > 10, or for reaching the driver's
  `NEHSTORE` of 1,000,000 pairs, is printed (`SKIPPED deck line k: …`) and counted at the end. A
  deck line that is not 7 numbers stops the run.
- Each cluster's header ends with its 1-based deck line, the record with the same `deck_line`
  in the `.json`/`.csv`.
- **A cluster's charge is its last time slice.** The 20 time slices are cumulative; never sum them
  (`pixelav_reference.md` §4). `parse_cluster.py` reports per cluster that charge and its
  charge-weighted extent (`--full`: only the summary line).

### 5.3 Deposits route: the injection driver

```bash
P=$CALOMAPS_HOME/models/pixelav_jobs_gamma50_20evt_si100um_cut1um_stepmax5um
bash analysis/pixelav/run_deposit_jobs.sh $P /tmp/pixelav_journey/pixelav/bpix100_50x25_run 30
python3 analysis/pixelav/charge_maps.py $P
```

- `run_deposit_jobs.sh <jobs prefix> <PIXELAV run dir> [nproc] [seed]` runs
  `ppixelav2_inject_deposits` on every part `<prefix>.partNNN.txt` (or on `<prefix>.jobs.txt`),
  `nproc` at a time (default: the number of parts, at most 8), with the driver's seed (default
  1). Each part gives `<prefix>.partNNN.out` (clusters, last time slice) and `<prefix>.partNNN.log`;
  a single `<prefix>.jobs.txt` gives `<prefix>.out` and `<prefix>.log`. Existing outputs are
  overwritten, and a part's old `.out` is removed before it runs. It prints the totals and exits 1
  when a part fails. The run directory must be the one the jobs were built for. Before running
  anything it refuses job files other than the parts `<prefix>.jobs.npz` names (a rerun of
  `deposit_jobs.py` with another `--parts` removes those of the earlier run) and a sensor model
  with a magnetic field. `<prefix>.run.json` records the run (driver seed, sha256 of the
  sensor files and the driver, the PIXELAV and CALOMAPS commits, host, time, totals, failed
  parts); `charge_maps.py` stores it in the charge maps.
- A job's cluster depends only on the seed and its job id, so the result does not depend on how
  the jobs were split into parts. Every job set numbers its jobs from 0, so two job sets run with
  the same seed reuse the same random streams: give each job set its own seed when several are to
  be combined. Parts are the unit of parallelism: write at least as many as the
  cores you will use (`CALOMAPS_JOB_PARTS`).
- One jobs file by hand, in the run directory after `ulimit -s unlimited`:
  `./ppixelav2_inject_deposits <jobs_file> <out_file> [seed] [last|all]` (seed default 1; `last`,
  the default, writes the final time slice, `all` all 20).
- `charge_maps.py <prefix> [--out FILE]` collects the per-job windows into the charge of every
  group on its sensor plane's global pixel grid and writes `<prefix>.charges.npz` (keys in
  `pixelav_reference.md` §6). It checks that every job appears exactly once, that each job's
  segments, energy and pairs match the table, and that the outputs' sensor matches the jobs';
  otherwise it exits 1 and writes nothing. It reads gzipped outputs (`.out.gz`) too.

Measured with the 100 µm, 50 × 25 µm model: the pairs made equal deposit / 3.68 eV, and the
charge collected equals the pairs made (ratio 1.0000). A 50 GeV photon event at 100 µm holds
about 157k steps and 65 million pairs; the driver takes about 15 µs of CPU per pair, so 20 events
take about 15 minutes on 30 cores.

From the charge maps, `analysis/pixelav/reticles.py` cuts fixed blocks of pixels with the
crossings in them and `analysis/pixelav/mip_counting.py` estimates the number of crossings per
block from its hit map: [`reticle_mip_counting.md`](reticle_mip_counting.md).

---

## 6. Handing it off

Give your collaborator the deck plus its companions:

```
pixelav_segments_<tag><E>_<N>evt<si>.pixelav.txt         <- feed this to PIXELAV
pixelav_segments_<tag><E>_<N>evt<si>.pixelav.txt.columns.txt
pixelav_segments_<tag><E>_<N>evt<si>.json / .csv         <- per-crossing provenance
trackermom_<tag><E>_<N>evt<si>.npz                       <- MCParticle + tracker arrays
```

They run it through their PIXELAV with a sensor model of the deck's thickness (the legend's
`# sensor:` line): 320 µm for the default geometry. The list driver they run must pair the
columns with the axes as `ppixelav2_list_trkpy_real_entry` does (`pixelav_reference.md` §3b);
each deck line yields one simulated cluster.

Deposit jobs travel the same way: `pixelav_jobs_<tag><E>_<N>evt<si>.jobs.txt` (or the parts),
`.jobs.npz`, `.summary.txt` and the step npz. They need `ppixelav2_inject_deposits` and a sensor model with
the pitches and thickness the jobs were built for (`pitch_x_um`, `pitch_y_um`, `thick_um` in the
`.jobs.npz`).

To validate a deck before sending it, open the **PIXELAV-input inspection notebook**
([`08_pixelav_deck_inspection.ipynb`](../notebooks/08_pixelav_deck_inspection.ipynb)) — it
reads a `.npz`/deck and walks from basic sanity checks up to the per-crossing momentum,
impact-point, and angle distributions that justify every column. The full per-crossing record
schema lives in [`pixelav_reference.md`](pixelav_reference.md).
