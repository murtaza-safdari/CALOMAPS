# PIXELAV reference

PIXELAV (also written "PixelAV") is Morris Swartz's detailed, first-principles Monte Carlo of
charge deposition and transport in a silicon pixel sensor. It is the simulation engine behind
the CMS `SiPixelTemplate` / generic-error pixel templates and the calibration engine for the
CMS Phase-2 Inner Tracker. CALOMAPS feeds it from the Geant4 shower by two routes: **Stage-B
track-crossing decks** (see [`analysis/pixelav_converter.py`](../analysis/pixelav_converter.py) and
notebooks [`06_sensor_crossings_tracker.ipynb`](../notebooks/06_sensor_crossings_tracker.ipynb),
[`07_sensor_crossings_calo.ipynb`](../notebooks/07_sensor_crossings_calo.ipynb) and
[`08_pixelav_deck_inspection.ipynb`](../notebooks/08_pixelav_deck_inspection.ipynb)), and **deposit
jobs** of Geant4 step deposits (see [`analysis/pixelav/deposit_jobs.py`](../analysis/pixelav/deposit_jobs.py)).
The drivers that run them are in [`analysis/pixelav/`](../analysis/pixelav/). This file is the
collected reference on PIXELAV itself — what it is, how to get it, the exact input and output
formats — and on the conventions the CALOMAPS converter, job builder and drivers follow. The
operational recipe is [`pixelav_handoff.md`](pixelav_handoff.md).

> Status of the facts below: most are read directly from the public PIXELAV C source and the CMS
> note; a few (the "canonical" deck choice, the CMS-internal copy) are inferred or gated and are
> marked as such. Where the two public source copies disagree, both are given.

---

## 1. What it models

PIXELAV is slow and physical, not a fast parametrisation. From CMS NOTE 2002/027 §3, it chains:

1. **Charge deposition** — the *exact* Bichsel pion–electron cross section (H. Bichsel,
   Rev. Mod. Phys. 60, 663 (1988)); interaction spacing drawn from an exponential; delta-ray
   energies from Bichsel's distribution with two-body kinematics; e–h pairs Poisson-distributed
   at 3.68 eV/pair; delta-ray range from the Damerell relation. (The Landau/Vavilov shapes are
   *approximations* to what this algorithm generates.)
2. **Electric field** — a realistic 3-D map from TCAD (Silvaco Atlas), solving Poisson + the
   electron/hole continuity equations on a lattice (3×3×3 µm, four-fold pixel symmetry).
3. **Charge transport** — `dv/dt = (e/m*)[qE + q·rH·(v×B) − v/µ(E)]` integrated with a 5th-order
   Runge–Kutta, field-dependent mobilities, the Hall effect / Lorentz drift, and 3-D diffusion
   (Einstein relation). Effective masses 0.260 mₑ (e), 0.241 mₑ (h); Hall factors rH = 1.15 (e),
   0.90 (h).
4. **Signal induction** — Ramo's theorem, with Kramberger radiation-damage trapping (per-carrier
   trapping test every 100 RK steps; trapped charge induces a segmented image-charge sum). For an
   unirradiated sensor τ_eff⁻¹ = 0 and nothing is trapped. Both carriers of every pair are
   transported: a carrier that has reached its electrode counts in full, one still in the bulk
   through the weighting potential at its position, so a pixel's charge is complete only when the
   slower holes have arrived as well (about 9.5 ns across 320 µm at 10 kV/cm).
5. **Electronics** (noise, ROC response/saturation, FED digitisation) are applied *afterward in
   separate analysis code*, so they can be varied without re-running the costly transport.

**Key consequence for us:** stock PIXELAV is **not** an energy-deposit consumer. Its drivers call
`deposit()`, which generates the ionisation along one straight track through the full sensor
thickness from the track geometry + kinematics. The track route feeds it **track segments** (§3b).
The deposits route replaces `deposit()`: `ppixelav2_inject_deposits` takes the Geant4 step
deposits, makes the e–h pairs itself and runs only PIXELAV's transport and signal stages (§3c).

---

## 2. How to obtain it (public, no CERN login)

| Repo | What it is |
|------|------------|
| **`github.com/badeaa3/pixelav`** | Swartz's actual code, "housed for preservation" (A. Badea). `ppixelav2.c` (engine) + four list/random drivers, `ppixel2.init`, `wgt_pot.init`, `SIRUTH.SPR`, weighting map, build scripts (x86-64/ARM/Apple-silicon), `docs/Pixelav_ac_file_format.pdf`, `docs/Pixelav_coordinates.pdf`. **No README.** |
| **`github.com/elizahoward/Muon_Collider_Smart_Pixels`** (`…/PixelAV/`) | The **Smart Pixels** lineage — same Swartz engine plus `ppixelav2_custom.c`, a **9-column, PID-aware** track-list wrapper, `compile_pixelav.sh`, Silvaco field maps. This is the variant CALOMAPS targets (we collaborate with Smart Pixels). |
| **`github.com/CodexForster/TCADtoPixelAV`** (MIT) | Converts Silvaco/Synopsys TCAD E-field + weighting maps into the `ppixel2.init` / `wgt_pot.init` grids PIXELAV consumes. **This is the route for our custom DECAL/MAPS geometry** (supply our own TCAD field map). |
| **`github.com/OzAmram/PixelTemplateProduction`** | The downstream CMS template producer that *consumes* PIXELAV output; does **not** bundle PIXELAV. |

Not on CVMFS/CMSSW (no `*pixelav*` in CMSSW src). No public `pixelav` on gitlab.cern.ch; a
CMS-internal copy exists behind CERN SSO (TWiki `Main/MakingSiPixelTemplates`, gated). Contact:
**morris@jhu.edu**.

**Build** (badeaa3): `gcc -O2 <driver>.c -msse -lm -o <driver>` (each driver `#include`s
`ppixelav2.c`). Smart Pixels: `compile_pixelav.sh` → `gcc ppixelav2_custom.c -o
bin/ppixelav2_custom.exe -lm -O2`. Intel `icc` is ~40% faster but discontinued.

**Run:** `./<driver> <frun> <runsize>` reading `track_list.txt` + the `.init` files +
`SIRUTH.SPR` from the working directory; writes `pixel_clusters_d*.out` + a seed file. ~30 000
tracks/run, splittable across jobs by `frun`.

CALOMAPS pins upstream at commit `8a774a3` (2024-05-13); `setup/setup_pixelav.sh` checks it out
and builds the baseline list driver and the two CALOMAPS drivers in `analysis/pixelav/`:

- `ppixelav2_list_trkpy_real_entry <frun> <track list> <output> <seed file>` — the list driver
  with the deck's impact point in place of the random one (§3b). One run takes 30 000 deck lines,
  block `frun`.
- `ppixelav2_inject_deposits <jobs file> <output> [seed] [last|all]` — drift and readout of given
  energy deposits (§3c).

Both need more than the default 8 MB stack: `ulimit -s unlimited` (macOS: `ulimit -s 65520`).

---

## 3. Inputs — two separate files

### 3a. Stage A — detector / field configuration (`ppixel2.init`)

Hand-authored per sensor (or generated by `TCADtoPixelAV`). Structure:

| Line | Content |
|------|---------|
| 1 | ASCII header string (a leading `1` enables header output) |
| 2 | `pimom filebase` — pion momentum (if < 1.1 → defaults to 45 GeV) and output-file base index |
| 3 | `Bx By Bz` — magnetic field (Tesla) |
| 4 | `thick xsize ysize temp flux_e flux_h rhe rhh peaktim samptim ehole new_drde npixx npixy npixz` |
| 5… | `npixx·npixy·npixz` rows of `ix iy iz Ex Ey Ez` — the 3-D E-field grid (**V/cm**) |

Plus a **weighting-potential** map and **`SIRUTH.SPR`** (Bichsel cross-section table), same
indexed-grid format. `pixinit()` reads the weighting grid from `wgt_pot.init` itself (header
`thick xsize ysize npixx npixy npixz`, then records `ix iy iz w00 .. w22`) and stops reading it,
printing "match problem", unless those six numbers equal `ppixel2.init`'s; the upstream
`wgt_pot.init` holds only a file path, so `setup/setup_pixelav.sh` copies the data file over it.
`analysis/pixelav/sensor_model.py --summary` checks both files the way `pixinit()` reads them.
Verbatim example header (Smart Pixels `silvaco25x25x50_final_efield.out`):

```
dot1_25x25_phase3:BPix@-50V,3.57T@90deg,253K,rhe=1.10,rhh=0.7
1.00 16400
3.57 0.0 0.00
50. 25. 25. 253. 0.0 0.0 1.10 0.70 20000. 4000. 0 1 13 13 51
   1   1   1   0.000000e+00 1.600000e-05 -1.569615e+04
```

Decoding line 4: thickness 50 µm, pixel 25×25 µm, T = 253 K, rhe 1.10, rhh 0.70, peaktim 20000,
samptim 4000, ehole 0, new_drde 1, grid 13×13×51.

### 3b. Stage B — the per-track list (what `pixelav_converter.py` writes)

Plain ASCII, **one whitespace-separated track per line**, read by `fscanf`. **The column set is
driver-specific** — the two public copies differ:

- **badeaa3 base** (`ppixelav2_list_trkpy_n_2f.c`) — **7 columns**, pion-only, no PID:
  `cot_alpha  cot_beta  ppion  flipped  modx  mody  pT`
  (`modx/mody/pT` are pass-through labels for ML; a simpler driver reads only the first 4.)
- **Smart Pixels** (`ppixelav2_custom.c`, the CALOMAPS target) — **9 columns**, adds PID:
  `cot_alpha  cot_beta  ppion  flipped  ylocal  zglobal  pT  hittime  PID`
  (`fscanf("%f %f %f %d %f %f %f %f %f")`; `ylocal zglobal pT hittime` are pass-through labels.)

What each driving field means:

- **`cot_alpha`, `cot_beta`** — the track direction. In the Smart Pixels wrapper:
  `locdir_z = 1/√(1 + cot_α² + cot_β²)`, `locdir_x = cot_α·locdir_z`, `locdir_y = cot_β·locdir_z`.
  So **`cot_alpha = p_x/p_z`, `cot_beta = p_y/p_z`** in the sensor-local frame (z = depth/normal,
  x and y the in-plane pixel axes). This matches the CMS convention `cot(alpha) = cosx/cosz`,
  `cot(beta) = cosy/cosz` (CMS NOTE 2002/027; CMS IN 2004/014; CMSSW `SiPixelTemplate.h`), where
  α is the incidence angle in the x–z plane and β in the y–z (magnetic-bending) plane.
  **Caveat — the two lineages differ, so verify against the driver you run.** The *badeaa3* list
  drivers (and our patched real-entry driver, which inherits from them) apply the columns the
  other way around: `locdir[0] = cot_beta·locdir[2]`, `locdir[1] = cot_alpha·locdir[2]` — i.e.
  the **first** deck column steers the **y** (13-pixel, Lorentz) axis and the **second** steers
  **x** (21-pixel). **Verified against the driver source**
  (`analysis/pixelav/ppixelav2_list_trkpy_real_entry.c`) and confirmed with an
  end-to-end run: our deck writes `cot_alpha = p_u/p_w` (col 1) + `mody = u` (col 6) onto PIXELAV's
  y axis, and `cot_beta = p_v/p_w` (col 2) + `modx = v` (col 5) onto PIXELAV's x axis — angles and
  impact labels land on the same axes, self-consistently. The stock snippet quoted below (from
  `ppixelav2.c`'s randomizing wrapper) names the variables the Smart-Pixels way; only the
  fscanf-to-axis pairing of the list drivers is swapped.
- **`ppion`** — momentum magnitude in **GeV/c**.
- **`flipped`** (int 0/1) — selects the entry face and the sign of the depth direction: `flipped`
  picks z-entry = 0 with n_z > 0, or z-entry = thickness with n_z < 0.
- **`PID`** (Smart Pixels only) — 211 (π), 13 (µ), 11 (e). **Non-pion handling is approximate:**
  the wrapper only rescales the momentum by the pion/particle mass ratio so β ≈ matches; the
  Bichsel dE/dx model stays pion-like. Feeding e±/µ is therefore an approximation, not a separate
  energy-loss model. Tracks with |cot| > 10 are skipped.

### ⚠️ The entry point is **not** an input (important design point)

Stock PIXELAV does **not** read a local entry point. It **randomises the impact uniformly over
the central 3×3 pixel area** each event and projects it to the sensor face via the angles:

```c
vect[0] = 3.*xsize * (rvec[0] - 0.5) + (vect[2] - thick/2.)*cotalpha;   // rvec = RANLUX uniforms
vect[1] = 3.*ysize * (rvec[1] - 0.5) + (vect[2] - thick/2.)*cotbeta;
```

So what stock PIXELAV consumes per track is **angles + momentum (+ type)** — *not* the entry
point. Two ways to use it:

1. **Standard / template mode** — accept the internal randomisation. Our converter's precise
   `entry_u/entry_v` are then carried only as **labels** (in the `ylocal/zglobal` slots) for
   bookkeeping / matching, not as transport inputs.
2. **Truth-injection mode** — to make PIXELAV use *our* real per-crossing entry point, **patch
   the wrapper** (`vect[0]/vect[1]`) to read the entry from the deck instead of randomising. This
   is the route if the goal is the cluster for a *specific* shower crossing rather than a template.

This is a decision for the collaboration; the converter emits the entry point either way.
`analysis/pixelav/ppixelav2_list_trkpy_real_entry.c` is the truth-injection patch of the badeaa3
list driver: it reduces `modx`, `mody` modulo the pitch into the window's central pixel as the
mid-plane point and back-projects it to the entry face with the track angles, as upstream does
with its random point. The random numbers are still drawn, so the random stream is unchanged.

### Units

**Lengths = microns (µm); time = picoseconds (ps); E-field = V/cm** (converted internally to
V/m). Firmly sourced from the C comments. So CALOMAPS records (mm) scale by `LENGTH_UNIT_MM =
1000` when written into the deck's length label columns.

### 3c. Deposit jobs — the input of `ppixelav2_inject_deposits`

The deposits route does not call `deposit()`. `analysis/pixelav/deposit_jobs.py` writes jobs of
short straight segments, each with its energy, in the frame of one PIXELAV output window (§6,
"Frames and pixel grids"); `analysis/pixelav/ppixelav2_inject_deposits.c` makes the e–h pairs
along them and runs PIXELAV's `propag()` (drift, diffusion, trapping) and `detect()` (induced and
collected charge on the 21 × 13 window).

Jobs file: plain ASCII, repeated blocks; blank lines and lines starting with `#` are ignored.

```
J <job_id> <n_seg>
<x1> <y1> <z1> <x2> <y2> <z2> <edep_eV>        (n_seg lines)
```

`job_id` ≥ 0 (64-bit), `n_seg` ≥ 0, `edep_eV` ≥ 0. Coordinates in µm in the window frame: x along
the 21-pixel side (pitch `xsize`), y along the 13-pixel side (pitch `ysize`), pixel (10, 6)
centred on (0, 0), so the window is |x| < 10.5 `xsize`, |y| < 6.5 `ysize`; z from 0 to `thick`.

Per job the driver

- makes `floor(n)` + Bernoulli(`n − floor(n)`) pairs per segment, `n = edep_eV / 3.68`, placed
  uniformly along the segment. The deposit already carries Geant4's energy-loss fluctuation, so no
  further smearing is added;
- clips z to 1e-3 µm inside the faces (`propag()` and `detect()` take z ≤ 0 and z ≥ `thick` as
  having left the slab);
- counts the pairs created outside the 21 × 13 window (`n_out`) and does not transport them;
- transports every carrier (`Nscale = 1`);
- re-seeds RANLUX from (seed, `job_id`), so a job's cluster does not depend on the other jobs in
  the file or on their order.

It stops with a line starting `ERROR` and exit status 1, before any transport, on a malformed
block, a non-finite number, `edep_eV` < 0, a segment end farther outside the window than the
window's own size (|x| > 31.5 `xsize`, |y| > 19.5 `ysize`, z < −`thick`, z > 2 `thick`), or a job
whose pairs certainly reach `NEHSTORE` (1,000,000); during transport, on a job whose drawn pairs
reach `NEHSTORE` (the output is then incomplete). `deposit_jobs.py` splits every job above
100,000 pairs (`--max-pairs`), which is exact: PIXELAV's carriers do not interact, so clusters add
linearly.

How `deposit_jobs.py` builds the jobs from the steps:

- **Segments.** A step hit (`CALOMAPS_TRACKER_HITS=step`) holds the step midpoint, the mean
  momentum, the step length and the deposit. Each becomes a straight segment centred on the
  midpoint, along the mean momentum, as long as the step, clipped in depth to the slab, with its
  whole energy on the clipped part; then cut into equal pieces of at most `--piece-um` (5 µm),
  each with its share of the energy by length. Steps of neutral particles are segments like any
  other (0.01–0.04% of the deposited energy in 50 GeV photon showers).
- **Groups.** The steps of one particle in one sensor plane (event, cellID module, cellID layer)
  form a node. A node joins its parent's node in the same plane when the particle was created
  inside that slab and the parent is a charged particle with steps in that plane; the top of the
  chain is the group root. So delta rays join the track that made them, and a photon converting
  in the silicon gives two roots.
- **Tiles and jobs.** Each sensor plane has one pixel grid covered by tiles of 17 × 9 pixels: the
  21 × 13 window less a 2-pixel guard band on every side. A piece belongs to the tile holding its
  midpoint. A job is one (group, tile); job ids run from 0 in the order (group, tile, split).
- **Checks.** Nothing is written unless the jobs' energy equals the steps' (relative 1e-9) and
  every piece lies within its tile, give or take half a piece. The sensor (a run directory's
  `ppixel2.init`, or `--pitch-x-um`, `--pitch-y-um` and the npz's thickness) must be within 1 µm of
  the npz's silicon thickness, and a model with a magnetic field is refused.

---

## 4. Output

Both CALOMAPS drivers write one text file. Line 1 is the sensor model's title (line 1 of
`ppixel2.init`); line 2 is `xsize ysize thick stimstp` (pitches and thickness in µm, the
time-slice step in ps). Then one block per cluster:

```
<cluster>
<header line>
<time slice T ps>             T = k·stimstp
13 rows of 21 pixel charges   row j = PIXELAV y pixel j, column i = x pixel i; electrons
...                           (the next time slice)
```

The window is `TXSIZE` = 21 pixels along x by `TYSIZE` = 13 along y; pixel (10, 6) is centred on
(0, 0) of the window frame.

**The time slices are cumulative.** Slice k holds the charge collected or induced up to time
k·`stimstp`, so **a cluster's charge is its last time slice**. Never sum the slices: that
overstates the charge 10 to 18 times. The 20 slices divide the model's sampling time `samptim`
(line 4 of `ppixel2.init`), which must outlast the charge collection: the `decal320` model uses
12 ns, because a 320 µm sensor at 10 kV/cm needs about 9.5 ns; with 4 ns its last slice held 89%
of the pairs made. For the 100 µm, 50 × 25 µm model the injection driver gives collected / made
= 1.0000.

Header lines:

| Driver | Fields |
|---|---|
| `ppixelav2_list_trkpy_real_entry` | 10: `x y z px py pz neh mody pT deck_line` — x, y, z the entry point on the entry face (µm, window frame); px, py, pz the direction times `ppion` (GeV/c); `neh` the e–h pairs generated; `mody` and `pT` copied from the deck; `deck_line` the 1-based deck line, the record with the same `deck_line` in the converter's `.json`/`.csv`. 20 time slices. Files from earlier builds of the driver have 9 fields, without `deck_line`. |
| `ppixelav2_inject_deposits` | 5: `job_id neh n_seg edep_eV n_out` — `neh` the pairs transported, `n_out` the pairs created outside the window; `neh + n_out` = pairs created. The last time slice (default) or all 20 (`all`). |

The list driver also keeps a **seed file**: its RANLUX state and event counter, rewritten after
every deck line. A seed file at a non-zero counter, or an existing output file, stops a new run
unless `PIXELAV_RESUME=1`; the driver then continues from that counter and appends to the output,
so a finished deck resumed this way writes nothing. Deck lines skipped for |cot| > 10 or for
reaching `NEHSTORE` are printed and counted at the end of the run. The injection driver has no
seed file and overwrites its output.

[`analysis/pixelav/parse_cluster.py`](../analysis/pixelav/parse_cluster.py) reads both drivers'
files, plain or gzipped (`iter_clusters`), and summarises each cluster's charge (its last slice)
and charge-weighted extent. Byte-exact upstream layout: `docs/Pixelav_ac_file_format.pdf` and
`docs/Pixelav_coordinates.pdf` in the badeaa3 repo.

---

## 5. Downstream CMS template pipeline (context)

`OzAmram/PixelTemplateProduction` turns PIXELAV output into `SiPixelTemplate` / `SiPixelGenError`
payloads via `gen_xy_template` → `gen_xy_template2d` → `gen_zp_template` → `gen_zp_template2d`,
driven by a `pix_2t.proc` config (reco/threshold params + cot-α/cot-β bin edges) — distinct from
the Stage-B track list. Not needed to *run* PIXELAV, but it documents the conventions and is the
path if CALOMAPS ever wants CMS-style templates.

---

## 6. How this maps onto CALOMAPS

- **Track route.** `analysis/pixelav_converter.py` builds one record per (MCParticle, Si layer,
  face) crossing with local entry (u, v), `cot_alpha = p_u/p_w`, `cot_beta = p_v/p_w`, |p|, particle
  type — which is exactly the Stage-B driving set (our `cot_alpha/cot_beta` match the Smart Pixels
  columns 1–2, whose convention takes `u` as local x, `v` as local y and `w` as depth z; the
  badeaa3 list drivers pair them the other way, §3b caveat).
- `write_pixelav_deck()` emits the **7-column `badeaa3`** layout by default (the format our
  patched real-entry driver reads), with the Smart Pixels 9-column layout available via
  `--layout smartpix` (the longer-term collaboration target). Lengths are emitted in microns.
- **Deposits route.** `analysis/pixelav/deposit_jobs.py` turns the Geant4 steps in the silicon
  into jobs for `ppixelav2_inject_deposits` (§3c), and `analysis/pixelav/charge_maps.py` collects
  the per-job windows into the charge of every group on its sensor plane's pixel grid.
- **Stage A.** `setup/setup_pixelav.sh` installs one of two sensor models: `decal320`, generated by
  `analysis/pixelav/make_decal_stagea.py` (320 µm, 100 × 100 µm pixels, a uniform drift field and
  an analytic weighting potential, B = 0), or `bpix100_50x25`, an external TCAD-derived map of a
  100 µm sensor with 50 × 25 µm pixels, which is not in the repository. A TCAD map of the DECAL
  sensor (via `CodexForster/TCADtoPixelAV`) is open item 5 below.
  `analysis/pixelav/sensor_model.py --summary` describes a model and `--check-deck` checks a deck
  against it (thickness, axis pairing, tracks longer than the window).

### Frames and pixel grids

The records use the sensor-local frame **u** (across the stave, tangential), **v** (along the
barrel axis, = global z) and **w** (along the face's outward normal). A tracker hit's sensor plane
is the `module` (stave) and `layer` of its cellID: the barrel staves overlap past the 30° sector
boundaries, so the face a hit's azimuth points at is not always the stave it sits on.

Both drivers use the same PIXELAV frame:

- **x = v**, with the model's first pitch (`xsize`) and the 21-pixel side of the window;
- **y = u**, with the second pitch (`ysize`) and the 13-pixel side;
- **z rises with the outward normal**: 0 on the inner face, `thick` on the outer one. A deck line
  with `flipped = 1` (outward) runs from z = 0 to z = `thick`. Electrons drift against the E
  field; in the `decal320` and `bpix100_50x25` models E_z < 0, so with `ehole = 0` they are
  collected at z = `thick`.

(v, u, w) is the mirror image of the right-handed (u, v, w). A sensor model without a magnetic
field cannot tell the two apart; `deposit_jobs.py` refuses a model with a field.

The two routes put the pixel boundaries half a pixel apart, along both x and y:

- **Track route.** The list driver reduces `modx` and `mody` modulo the pitch into the central
  pixel, which is centred on (0, 0): pixels are centred on multiples of the pitch in v and u, so
  pixel k covers [k − ½, k + ½) × pitch.
- **Deposits route.** Each sensor plane has the grid `iv = floor(v / xsize)`,
  `iu = floor(u / ysize)`: pixel i covers [i, i + 1) × pitch. Tile `tv = floor(iv / 17)`,
  `tu = floor(iu / 9)`; window column 0 is global pixel `17·tv − 2` and row 0 is `9·tu − 2`
  (`job_win_iv0`, `job_win_iu0`), and window x = v − (17·tv + 8.5)·`xsize`,
  y = u − (9·tu + 4.5)·`ysize`.

In both routes the simulated pitch is the sensor model's, not the calorimeter readout's
(`geo_cell_size_mm`).

### The CALOMAPS per-crossing record — authoritative schema

`pixelav_converter.py` writes, next to the deck, a `.json`/`.csv` with one 20-field record per
crossing, every crossing whether or not it is in the deck (the provenance behind every deck line):
the 18 fields of `analysis/sensor_crossings.py` and the converter's `deck_line` and `deck_drop`.
Coordinates: **u** = across-pitch (tangential),
**v** = cylinder-z, **w** = sensor depth (outward face normal); all lengths in **mm** here (the
deck's length labels are in µm).

| Field | Unit | Meaning |
|---|---|---|
| `track_id` | – | index into the npz's MCParticle arrays (trace any line back to its particle); with several events these hold every event's particles in turn, and the npz `pid` is the index in the event's own `MCParticles` |
| `layer_id` | – | Si layer 0–29 (0 = innermost): the `layer` field of the hit's readout `cellID` when the npz has it (`tlayer`), else the layer nearest in depth |
| `pdg` | – | PDG code of the crossing particle |
| `p_GeV` | GeV | \|p\| — variant **C**: true Geant4 momentum *at the crossing*; **A**/**B**: momentum at production |
| `entry_u`, `entry_v` | mm | sensor-local impact point. Variant **C**: the `Geant4TrackerWeightedAction` hit position, the midpoint of the hit's first and last step points, which for a straight through-going track is where it crosses the Si mid-plane; **not** the entry face; **A**: earliest-in-time step position |
| `cot_alpha` | – | `p_u / p_w` (drives PIXELAV's 13-px/Lorentz **y** axis in our driver) |
| `cot_beta` | – | `p_v / p_w` (drives PIXELAV's 21-px **x** axis) |
| `flipped` | 0/1 | 1 = outward-going (`p_w ≥ 0`) → driver entry face `z = 0`; 0 → `z = thick` |
| `sensor_normal_phi` | rad | outward-normal azimuth of the dodecagon face the crossing is on: the face of the hit's `cellID` `module` (stave) when the npz has it (`tmodule`), else the face its azimuth points at; the staves overlap past the 30° sector boundaries |
| `depth_w_mm` | mm | radial depth (apothem) of the impact point |
| `energy_dep_GeV` | GeV | Geant4 energy deposited in the crossing — bookkeeping only; PIXELAV regenerates its own ionization |
| `n_steps` | – | Geant4 steps merged into this record (variant C: 1 combined hit) |
| `time_ns` | ns | crossing time (deck's `hittime` label is this ×1000, in ps, smartpix layout only) |
| `variant` | – | `C` (tracker truth), `A` (calo step truth), `B` (pixel centroid fallback) |
| `flags` | – | provenance markers: `tracker_hit`, `dir_from_momentum`, `pixel_centroid`, `grazing`; variant **C** also `outside_si` (hit position outside its Si slab), `starts_in_si` / `stops_in_si` (the particle's vertex / endpoint within the slab's depth range) |
| `evt_id` | – | event of the crossing: its entry number in the simulation's ROOT file (0 for an npz without event ids, which holds one event) |
| `path_length_mm` | mm | variant **C**: the hit's Geant4 path length in the Si (`pathLength`, summed over the hit's steps); **A**/**B**: NaN |
| `deck_line` | – | the record's line number in the deck, from 1; 0 when it has none (converter records only) |
| `deck_drop` | – | why it has no deck line, empty when it has one (converter records only; reasons below) |

Deck-column ↔ record mapping (badeaa3 7-col): `cot_alpha cot_beta ppion flipped modx mody pT` =
`cot_alpha`, `cot_beta`, `p·(m_π/m_particle)` (βγ-matched), `flipped`, `entry_v·1000` (µm),
`entry_u·1000` (µm), `p_GeV`.

Which crossings get a deck line (`pixelav_converter.deck_drop_reason`, first match wins).
PIXELAV simulates every deck line as a straight track through the full silicon thickness at
constant momentum, so a crossing is left out when it is

- `grazing`: a non-finite slope;
- `steep`: |cot_alpha| or |cot_beta| > 10, for which the driver writes no cluster;
- `outside_si`: flagged so;
- `path_short` / `path_long`: its `path_length_mm` differs from the straight chord
  `thickness·√(1 + cot_alpha² + cot_beta²)` by more than `--path-tolerance` (default 5%): it
  starts or stops in the silicon, or scatters or curls in it.

`--select through` (the default; `CALOMAPS_DECK_SELECT` in the wrapper) leaves all of these out,
`--select all` only the grazing and steep ones. Records without a path length (variants A and B)
pass the path tests and are counted as `path_unknown`. The deck legend carries the counts and the
share of the deposited energy per reason. In 50 GeV photon showers `through` keeps 65% of the
crossings, carrying 44% of the silicon energy, at 100 µm, and 54% carrying 41% at 320 µm. Most
crossings below 3 MeV/c are left out, and almost none above 30 MeV/c.

### Deposit-job tables (`<prefix>.jobs.npz`, from `deposit_jobs.py`)

| Keys | Meaning |
|---|---|
| `job_id` | job id, from 0 |
| `job_evt_id`, `job_module`, `job_layer` | the job's sensor plane: event (ROOT entry), cellID module and layer |
| `job_group_id` | its group |
| `job_tile_v`, `job_tile_u` | its tile |
| `job_win_iv0`, `job_win_iu0` | global pixel of window column 0 and row 0 |
| `job_n_seg`, `job_edep_keV` | segments and deposited energy |
| `job_n_split` | jobs its (group, tile) was split into |
| `job_part` | the part file that holds it |
| `group_id` | group id, from 0 |
| `group_evt_id`, `group_module`, `group_layer` | the group's sensor plane |
| `group_root`, `group_root_pdg`, `group_root_p_MeV` | root particle (index in the npz particle arrays), its PDG code, and its \|p\| at its first step in the plane |
| `group_n_particles`, `group_n_steps`, `group_edep_keV`, `group_path_um` | particles, steps, deposited energy, and summed step length |
| `group_w_min_um`, `group_w_max_um` | extent of its clipped segments in the window z (0 to `thick`) |
| `group_u_centroid_um`, `group_v_centroid_um` | energy-weighted centroid in the sensor plane |
| `group_u_min_um`, `group_u_max_um`, `group_v_min_um`, `group_v_max_um` | extent in the sensor plane |
| `group_t_first_ns` | time of its first step |
| `group_created_in_slab` | the root itself starts inside the slab (e.g. an electron from a photon interaction) |
| `group_n_tiles` | tiles it spans |
| scalars | `pitch_x_um`, `pitch_y_um`, `thick_um`, `sensor_source`, `window_x`, `window_y`, `guard`, `tile_x`, `tile_y`, `centre_x`, `centre_y`, `ev_per_pair`, `max_pairs`, `piece_um`, `parts`, `events_first`, `events_stop`, `source_npz` |
| provenance | the input npz's `sim_*` and `geo_*` keys, `hit_mode`, `particles_kept`, `num_events`, `tcellID_encoding` |

### Charge maps (`<prefix>.charges.npz`, from `charge_maps.py`)

`charge_maps.py <prefix> [--out FILE]` reads `<prefix>.jobs.npz` and the driver outputs next to it
(`<prefix>.partNNN.out`, or `<prefix>.out` for one part; each may be gzipped). A job's charge map
is the last time slice of its block, placed on the global grid at `job_win_iv0`, `job_win_iu0`.
Maps add, so a group's charge is the sum of the maps of its jobs; the guard bands of neighbouring
tiles overlap, and the jobs of a split tile share one window. It exits 1 and writes nothing unless
every job id appears in exactly one block, in the part its `job_part` names; every block has the
injection driver's header and 1 or 20 complete slices; each header's `n_seg` and deposited energy
match the table; the pairs made match `job_edep_keV / ev_per_pair` within `n_seg`; and every
output's title and pitch line agree with each other and with the table's pitches and thickness.

| Keys | Meaning |
|---|---|
| `pix_group_id`, `pix_iv`, `pix_iu`, `pix_q` | per (group, pixel), sorted by group then `iv`, `iu`, zero pixels dropped: the group's summed charge in electrons (float32) |
| `job_id`, `job_neh`, `job_n_out`, `job_q` | per job: pairs transported, pairs made outside the window, charge of its map |
| every `group_*` key of the jobs table, `group_q`, `group_n_pix` | per group: its charge and its number of `pix_*` rows |
| scalars and provenance | those of the jobs table |
| `n_jobs`, `n_groups`, `n_pix`, `total_q`, `total_neh`, `total_n_out`, `total_edep_keV`, `out_files`, `jobs_npz`, `pixelav_title`, `pixelav_stimstp_ps` | the collector's own |

### Limits

**Track route.** PIXELAV simulates each deck line as a straight track through the full thickness
at constant momentum, with the pion Bichsel model at the βγ-matched momentum, and generates its
own delta rays. The deck leaves out the crossings that are not such tracks (above). The list
driver puts the mid-plane point anywhere in the window's central pixel and drops charge outside
the 21 × 13 window, so a track whose length inside the silicon exceeds 20 pitches along x or 12
along y can be cut (`sensor_model.py --check-deck` counts them). Lines with |cot| > 10 give no cluster.

**Deposits route.** Not modelled: the spread of the deposits inside a segment beyond a uniform
line (the segments must be short: Geant4 steps, cut into pieces); the arrival times of the
deposits (all pairs of a job start drifting at t = 0); any interaction between carriers (PIXELAV
has none, so clusters add linearly). Each step is rebuilt as a straight segment of its path length
along its mean momentum, so the segments are only as good as the steps are short: the route
limits silicon steps to 5 µm, and without a limit 30% of the steps of a 50 GeV photon shower in
100 µm silicon, holding 46% of its energy, are clipped. Pairs made outside a job's window are
counted (`n_out`), not transported. The B field is taken from `ppixel2.init` as it is (the list
drivers flip its sign with `cot_beta`), and `deposit_jobs.py` refuses a model with a field. The
route costs about 15 µs of CPU per pair; a 50 GeV photon event at 100 µm holds about 157k steps
and 65 million pairs.

### Open decisions (need the collaboration / Swartz)

1. **Which fork/column layout** is canonical for our Smart Pixels target (we assume the 9-col
   `ppixelav2_custom.c`); confirm there isn't a newer internal version.
2. **Entry point:** standard randomisation vs truth injection. Our patched driver
   (`analysis/pixelav/ppixelav2_list_trkpy_real_entry.c`, built by `setup/setup_pixelav.sh`)
   already injects the deck's truth impact (reduced mod-pitch); the collaboration still has to
   choose which mode their production runs use. Note the "entry" we carry is the
   `Geant4TrackerWeightedAction` hit position, the midpoint of the hit's first and last step
   points — for a straight through-going track the point where it crosses the sensor
   **mid-plane** — not the entry-face point; the driver back-projects it to the entry face using
   the track angles, exactly as stock PIXELAV does with its randomized mid-plane impact.
3. ~~**cot axis/sign** for our sensor orientation (the public copies disagree on x↔y)~~ —
   **resolved** for the patched driver this project targets — the pairing is verified against
   the driver source (see the §3b caveat above). Re-verify if you switch to a different
   driver/lineage (e.g. the 9-column Smart Pixels wrapper).
4. **Non-pion dE/dx:** is the momentum-rescaling approximation acceptable for the e±-dominated
   shower, or does Swartz have a proper electron treatment? (The deposits route does not depend
   on it: its ionisation is Geant4's.)
5. The Stage-A field map for the 100 µm-pitch DECAL Si (TCAD → `TCADtoPixelAV`).

---

## 7. References

- M. Swartz, *A Detailed Simulation of the CMS Pixel Sensor*, CMS NOTE 2002/027 — <https://cds.cern.ch/record/687440> ([PDF](https://cds.cern.ch/record/687440/files/note02_027.pdf))
- M. Swartz, *CMS pixel simulations*, NIM A 511 (2003) 88, doi:10.1016/S0168-9002(03)01757-1 — <https://cds.cern.ch/record/726082>
- *Pixel Hit Reconstruction with the CMS Detector*, arXiv:0808.3804 — <https://arxiv.org/abs/0808.3804>
- Smart Pixels: arXiv:2312.11676; *Sensor Co-design for smartpixels*, arXiv:2510.06588 — <https://arxiv.org/html/2510.06588v1>; dataset doi:10.5281/zenodo.7331128
- Source: <https://github.com/badeaa3/pixelav> · <https://github.com/elizahoward/Muon_Collider_Smart_Pixels> · <https://github.com/CodexForster/TCADtoPixelAV> · <https://github.com/OzAmram/PixelTemplateProduction>
- Angle convention: CMS IN 2004/014 (referenced in CMSSW `CondFormats/SiPixelTransient/interface/SiPixelTemplate.h`)
