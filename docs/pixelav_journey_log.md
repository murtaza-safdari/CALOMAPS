# PIXELAV integration — build, run, and reproduce on EAF

This is the operational record of wiring **PIXELAV** (M. Swartz's silicon-pixel
charge-transport Monte Carlo) to the CALOMAPS DECAL pipeline. It exists so the work can
be reproduced or resumed from scratch — the build lives under EAF `/tmp`, which is
off-quota but **wiped on container restart**.

Conceptual background and source provenance are in
[`pixelav_reference.md`](pixelav_reference.md); this file is the *how it actually runs*.

Branch: `pixelav-integration`. Stable return point: tag `pre-pixelav-journey`
(`git fetch --tags && git checkout pre-pixelav-journey`).

---

## 0. Where things live

| Thing | Path | Notes |
|-------|------|-------|
| PIXELAV build | EAF `/tmp/pixelav_journey/pixelav/` | **ephemeral**; rebuild with `setup/setup_pixelav.sh` |
| Upstream source | `github.com/badeaa3/pixelav` | public mirror of Swartz's code, no CERN login |
| Our additions | repo `analysis/pixelav/` | patched driver, Stage-A config, deck writer — **durable** |
| Per-crossing input | repo `analysis/pixelav_converter.py` | builds the 7-column deck from the cascade `.npz` |
| Reference PDFs | `github.com/badeaa3/pixelav` `docs/` | `Pixelav_ac_file_format.pdf`, `Pixelav_coordinates.pdf` |

EAF has no SSH gateway; it is driven via a JupyterHub terminal over WebSocket. `/home` is
capped at **23 GB per user** (a 21 GB legacy dataset already sits there), so all PIXELAV
build artifacts and large outputs go under `/tmp` (4 TB overlay, off-quota). Writes from
EAF to `/nashome` are native NFS (reliable).

## 1. Reproduce the build (one command)

```bash
# on an EAF JupyterLab terminal, in the CALOMAPS repo root
bash setup/setup_pixelav.sh
```

This clones `badeaa3/pixelav` into `/tmp/pixelav_journey/`, builds the list driver
(`gcc -O2 ppixelav2_list_trkpy_n_2f.c -msse -lm`), installs our patched driver and DECAL
Stage-A config if present in `analysis/pixelav/`, and runs a one-track smoke test.

Toolchain confirmed on EAF: gcc 11.5, make 4.3, cmake 3.26 (only gcc is needed). The
`sse2neon.h` shim is ARM-only; x86-64 uses `<xmmintrin.h>`.

## 2. PIXELAV in two stages

- **Stage A — sensor/field model** (hand-authored per sensor, *external* to our sim): a 3-D
  E-field map, weighting potential, thickness, pixel pitch, bias, temperature, mobility/Hall
  model and B-field. Lives in `ppixel2.init` (E-field + scalar params) and a weighting-
  potential file named by `wgt_pot.init`. The bundled example is a CMS BPix "dot1" smart-pixel
  sensor (see §6).
- **Stage B — per-track event input** (this is what our converter produces): one ASCII line per
  charged-track sensor crossing in `track_list.txt`.

A run directory needs: `SIRUTH.SPR` (Bichsel straggling table), `ppixel2.init`, `wgt_pot.init`,
the weighting-potential file it points to, and `track_list.txt`. Invoke:

```bash
./ppixelav2_list_trkpy_n_2f 1 track_list.txt clusters.out seedfile.txt
```

Note the 4-argument (argc==5) form is required: `<first_run=1> <track_list> <output> <seedfile>`.
The 1- and 2-argument forms leave the output filename unset (an upstream bug).

## 3. Input deck — 7 columns (verified from the driver `fscanf`)

```
cot_alpha  cot_beta  ppion  flipped  modx  mody  pT
```

| Col | Meaning | Units |
|-----|---------|-------|
| `cot_alpha` | slope along the **13-px (y / Lorentz)** axis — see §4 | — |
| `cot_beta`  | slope along the **21-px (x)** axis | — |
| `ppion` | momentum magnitude (treated as a pion for dE/dx) | GeV |
| `flipped` | entry face: `1` → enters at z=0 (back), `0` → enters at z=thickness (ROC side) | int |
| `modx` | local impact across the pitch (label only upstream; **our patch consumes it**) | µm |
| `mody` | local impact (label only upstream; **our patch consumes it**) | µm |
| `pT` | written to the cluster header as `track_q_sign*pT` (label) | GeV |

Tracks with `|cot_alpha|>10` or `|cot_beta|>10` are silently skipped by the driver.

## 4. Coordinate / axis convention (the swap — important)

PIXELAV's local frame (from `Pixelav_coordinates.pdf`): array is **21 (x) × 13 (y)** pixels,
center `pix_06_10`; **x** = long axis, **y** = short axis carrying the **Lorentz drift**,
**z** = depth (0 at back, `thickness` at the ROC/collection side); **E**‖−z, **B**‖+x.

The driver **swaps the alpha/beta labels** relative to the PDF/CMS text. From the source:
`locdir[0] = cot_beta·n_z` (drives `vect[0]`, the x/21-px axis) and
`locdir[1] = cot_alpha·n_z` (drives `vect[1]`, the y/13-px Lorentz axis). So in the deck:

- **col-1 `cot_alpha` → the 13-px Lorentz (y) axis**
- **col-2 `cot_beta`  → the 21-px (x) axis**

Mapping to the DECAL sensor-local frame (u = across-pitch, v = cylinder-z, w = depth): with
B along the solenoid axis (our v) the Lorentz drift is along u, so **u → PIXELAV y (13-px)**,
**v → PIXELAV x (21-px)**, **w → PIXELAV z**. That makes
`col-1 = p_u/p_w` and `col-2 = p_v/p_w`. This must still be confirmed with a one-track
round-trip against the binary before trusting cluster orientation.

`flipped` ↔ entry face and the sign of `n_z`: flipped=1 → z-entry=0, n_z>0; flipped=0 →
z-entry=thickness, n_z<0.

## 5. Output cluster format (`clusters.out`)

```
<80-char header>
x-pitch  y-pitch  thickness  time-slice-step          # all µm, ps
<cluster>
x-entry y-entry z-entry  px py pz  neh  y_module  track_q_sign*pT
<time slice 1 ps> ... <time slice 20 ps>              # NCRRC=20 slices
  each slice: 13 rows × 21 cols of pixel charge
<cluster> ...
```

`(px,py,pz)` are momentum components (`direction · ppion`), not bare cosines. Pixel charges
must be ×10 to recover total charge = `neh` (a sampling factor). The track midplane position
is forced into the 3×3 about `pix_06_10`.

## 6. Bundled baseline config (sanity reference)

Header `dot1_50x13_phase3:BPix@-100V,3.8T@90deg,263K,rhe=1.10,rhh=0.7`; param line
`50.0  12.5  100.0  200.0` → x-pitch 50 µm, y-pitch 12.5 µm, thickness 100 µm, 200 ps slices.
A perpendicular 2 GeV track gives ~18 k e-h pairs in 100 µm Si. This is the working baseline
the DECAL Stage-A config (§7) replaces.

## 7. The entry-point patch (planned)

Upstream randomizes the sub-pixel impact uniformly over the central 3×3 pixels
(`ranlux_(rvec,&c__4)` then `vect[0]=3·xsize·(rvec[0]-0.5)+…`). Its job there is to *scan*
position dependence into a template, so the entry we pass is only a label. To make PIXELAV
simulate the cluster at **our real per-crossing impact**, the patched driver
(`analysis/pixelav/ppixelav2_list_trkpy_real_entry.c`) replaces the random term with the
deck's `modx/mody` (reduced to a sub-pixel offset near the center pixel), keeping the
back-projection from midplane to the entry face intact.

## 8. Status & next steps

- [x] Source obtained, driver built, baseline run verified on EAF.
- [x] DECAL Stage-A config (`ppixel2.init` + weighting potential) for our pitch/thickness with
      a simple analytic E-field. (§9)
- [x] Patched driver consuming real `modx/mody`. (§9)
- [x] `write_pixelav_deck()` finished (7-col, µm, axis map above) and a deck generated. (§9)
- [x] Round-trip to confirm the cot/axis/flip convention; full run + plots. (§9, and re-verified
      end-to-end 2026-07-09 — see §10.)

---

## 9. Built + verified (2026-06-05)

Everything in §8 is done; the full pipeline runs on EAF.

**What was built**
- `analysis/pixelav/ppixelav2_list_trkpy_real_entry.c` — the baseline driver with the random 3×3
  impact draw replaced by the **real per-crossing entry** (deck `modx`/`mody`, reduced mod-pitch
  to a sub-pixel midplane offset).
- `analysis/pixelav/make_decal_stagea.py` — generates the DECAL **Stage-A**: 320 µm thick, 100 µm
  square pitch, uniform `E_z = −10000 V/cm` (V_bias 320 V), **B = 0**, and an FFT-computed **Ramo
  weighting potential** (central-pixel weighting rises 0 → 0.046 → 1.0 from backplane to
  collection). Simple-field model, in lieu of a TCAD map (authorized).
- `analysis/pixelav_converter.py` — `write_pixelav_deck()` now defaults to the **7-column badeaa3**
  format and writes the entry with the correct axis pairing (`modx = v_entry`, `mody = u_entry`).
- `setup/setup_pixelav.sh` — one command rebuilds all of the above plus a ready `decal_run/` dir.

**Verified**
- *Baseline:* a 4-track synthetic deck → 4 clusters (BPix sensor). After the `wgt_pot.init` fix
  the weighting loads (no `match problem`) and clusters show real charge sharing.
- *Real-entry patch:* identical tracks give identical, **injected** entries (e.g. `modx=10,mody=3`
  → `x-entry=10, y-entry=3`) vs random entries from the upstream driver. The ionization stays
  stochastic (neh fluctuates) — correct physics.
- *Our data on our geometry:* the 2,188-crossing deck through the DECAL sensor gives clusters with
  **neh ≈ 24–29k e-h** (MIP for 320 µm Si, 3.2× the 100 µm BPix), at the injected real entries.
- *Caveat:* high-momentum crossings (the ~47 GeV primary products) can exceed the 150k-e-h
  `NEHSTORE` cap and are skipped by PIXELAV; the soft-track bulk is retained. Bump `NEHSTORE` +
  recompile to keep them. Per-crossing momentum is the MCParticle production momentum (EDM4hep has
  no per-step momentum) — a flagged approximation for deep crossings.

**Run our event**
```bash
bash setup/setup_pixelav.sh
RUN=/tmp/pixelav_journey/pixelav/decal_run
cp models/pixelav_segments_gamma50_1evt.pixelav.txt $RUN/track_list.txt
cd $RUN && ./ppixelav2_list_trkpy_real_entry 1 track_list.txt clusters.out seedfile.txt
```

**Next (fidelity)**
- A one-track round-trip to confirm the cot/axis mapping (our u↔y(13px), v↔x(21px)) against the
  binary's cluster orientation.
- Per-crossing momentum (needs per-step truth) and a TCAD field map (`TCADtoPixelAV`) to replace
  the simple uniform E-field.

---

## 10. Fixes pass (2026-06-05) — the §9 caveats addressed

All four limitations flagged in §8/§9 were fixed this journey (branches `shower-4vector-outputs`
for the sim/momentum work, `pixelav-integration` for the PIXELAV-side fixes, then merged):

1. **Per-crossing momentum is now real** (was: the producing particle's production momentum). The
   ECal Si is read out as a Geant4 tracker — `sim/run_sim_trackermom.py`,
   `SIM.action.mapActions['ECalBarrel'] = 'Geant4TrackerWeightedAction'` — so each `SimTrackerHit` is
   one crossing carrying the true Geant4 momentum. `analysis/extract_trackermom.py` pulls it and
   `pixelav_converter.build_segments_C` (Variant C, auto-selected) builds the crossings. nb05 §9
   validates it: |p| on/below the production diagonal, softening with depth, monotone loss along the
   leading track. (See handbook §10.1 for the run recipe.)

2. **dE/dx is species-correct** (was: every track treated as a pion at its face-value momentum).
   `write_pixelav_deck` feeds the **βγ-matched** pion momentum `ppion = p·m_π/m_particle` (~273× for
   e±) so a soft shower electron ionises like the relativistic particle it is (plateau ~MIP), not
   like a slow pion on the 1/β² rise. The `pT` column keeps the real |p| as a label.

3. **Overflow fixed** (was: the 150k-e-h `NEHSTORE` cap dropped ~66% of crossings). `NEHSTORE`
   150k→1M in the patched driver — the 1M static arrays need `ulimit -s unlimited` (the 8 MB default
   stack segfaults), now baked into `setup_pixelav.sh` and the run recipe. Combined with fix 2 (soft
   electrons no longer over-ionise into overflow), nearly all crossings are retained.

4. **B = 0 confirmed appropriate** (was: flagged as a simplification). The test-beam `<fields>` block
   in `SiD_TestBeam.xml` sets the solenoid to 0 T, so the field-free shower and the B=0 sensor are
   consistent — no Lorentz drift in either. The uniform drift field + analytic Ramo weighting remain
   the documented simple model (nb06 §6); a TCAD field map via `TCADtoPixelAV` is the next fidelity
   step (it would refine cluster *shapes*, not the charge scale).

Notebook 06 was rebuilt around these: it now explains what a cluster is, how its shape encodes the
track, the transport physics (time development, path-length charge growth), the downstream
observables (charge spectrum, cluster size, and the charge-weighted **position resolution**), and
justifies the Stage-A assumptions.

---

## 11. Readout complementarity, the in-shower digitisation question, and the 05a/05b split (2026-06-05)

After the per-crossing-momentum work (§10) we examined whether the tracker-SD readout is the right
and sufficient PIXELAV input, how it relates to the calorimeter readout, and the field's best practice.

**Two readouts, one shower.** The ECal Si is read out two ways: calorimeter SD -> `SimCalorimeterHit`
+ `CaloHitContribution` (energy per pixel + per-step deposits; NO momentum; used by nb01-04); tracker
SD (`Geant4TrackerWeightedAction`) -> `SimTrackerHit` (one per crossing: momentum, energy-weighted
~mid-plane position, `pathLength`, `eDep`, time, MCParticle link). The sensitive detector is a passive
observer -- verified by byte-identical MCParticle cascades (78,270 particles, identical momenta and
vertices) between the two same-seed runs.

**PIXELAV wants the mid-plane impact.** The patched driver reduces `modx/mody` mod-pitch as the sub-
pixel MID-PLANE impact and projects to the faces itself via the direction. The tracker energy-weighted
position IS the mid-plane, so it feeds PIXELAV directly; back-projecting to the entry face would
double-project. => the tracker SD alone is the natural, sufficient PIXELAV input. The combined
calo+tracker SD idea was evaluated and dropped (DD4hep needs a custom plugin / untested `collections=`
path, and it buys nothing for PIXELAV).

**Primary vs secondary = birth point, not generator status.** All crossings are Geant4 shower
secondaries (the 50 GeV photon does not ionize). The discriminator is the production vertex: born
OUTSIDE the sensor (entering track, ~83%, median KE ~8 MeV, ~96% full traversal) vs born INSIDE (delta
ray, ~16%, median KE ~0.1 MeV, mostly `pathLength` < 320 um). `pathLength` (a SimTrackerHit field)
identifies partials: < 320 um = not a full traversal.

**Best practice (literature pass).** The in-shower per-pixel response is a DEPOSIT-driven problem
(Allpix2: Geant4 per-step deposits -> drift/diffuse -> threshold), not track-driven; EPICAL-2 (the
FoCal demonstrator, 24 ALPIDE MAPS layers + W -- geometrically close to our DECAL) is the precedent.
PIXELAV is track-driven: it assumes a through-going primary and regenerates its own delta rays
(Bichsel), so feeding it every Geant4 crossing (a) mishandles partials and (b) double-counts ionization
(Geant4 deltas + PIXELAV's internal deltas).

**Quantified on our event.** Collected PIXELAV charge = ~0.8 GeV-equiv (Sum neh within the 13x21
window; ~1.07 GeV of raw generated e-h) vs the true Geant4 Si deposit 0.53 GeV -> ~1.6x over (2x on the
raw). The per-crossing charge is correct (perpendicular MIP -> 26,900 e-h); only the full-shower SUM
over-counts.

**Decisions (2026-06-05).**
1. Keep all crossings (no filter applied); document the filter fix -- feed only born-outside
   full-traversal primaries (~80%) and raise the Geant4 Si production (range) cut so Geant4 does not
   pre-produce the deltas PIXELAV regenerates (cuts-per-region, since the calo path wants a fine cut)
   -- in nb06 §5c.
2. Position PIXELAV as the per-crossing characterisation tool (cluster shape, charge sharing, eta,
   single-hit resolution -- nb06 §1/§2/§5b), NOT a full-shower digitiser. The full-shower digital
   readout stays the deposit-driven calorimeter path (nb01-03; Allpix2 is the citable higher-fidelity
   upgrade, validated against EPICAL-2/FoCal).
3. Split notebook 05 into two co-equal methods: `05a_pixelav_inputs_tracker.ipynb` (tracker SD,
   momentum) and `05b_pixelav_inputs_calo.ipynb` (calo SD, step-energy truth, entry-face). The
   readout complementarity is documented in 05a §2b and the 05b intro.

---

## 10. Full-chain re-verification + port to `main` (2026-07-09)

Everything in this log was re-exercised from scratch on the **updated EAF image** (AlmaLinux 9.7,
Key4hep pinned `2026-02-01`), starting from a fresh `git clone` of `main`:

1. `ddsim` (both steering files, seed 424242) → `extract_trackermom.py` → `pixelav_converter.py`
   reproduced the committed per-crossing CSV **byte-for-byte** (3217 crossings / 3241 tracker hits /
   24 neutral-skipped; deck: 3109 lines, 108 grazing skipped). The `free(): invalid pointer`
   finalization crash (§ earlier) did **not** reproduce on the new image under the pinned release.
2. `setup_pixelav.sh` rebuilt PIXELAV + the DECAL Stage-A model from scratch (one fix: the Stage-A
   generator must run with a clean `PYTHONPATH` from a Key4hep-sourced shell — now in the script);
   the patched driver on the fresh deck reproduced the committed `pixelav_clusters_full.out`
   **byte-identically** over the compared prefix.
3. The cot/axis/flip convention was verified a second, independent way: by direct read of the
   driver source. `locdir[0] = cot_beta·locdir[2]`, `locdir[1] = cot_alpha·locdir[2]`, so deck
   col 1 (`cot_alpha = p_u/p_w`) steers PIXELAV **y** (13 px, Lorentz) — the same axis as col 6
   (`mody = u`) — and col 2 (`cot_beta = p_v/p_w`) steers PIXELAV **x** (21 px), same axis as
   col 5 (`modx = v`). `flipped = 1` (outward, `p_w ≥ 0`) → `locdir[2] > 0` → entry face `z = 0`.
   Angles and impact labels are pairwise consistent; the convention is closed.
4. One geometry subtlety was root-caused while validating: the `ECalBarrel_o2_v03` driver starts
   the layer stack at `rmin + ecal_barrel_tolerance` (= `env_safety` = 0.1 mm), so the naive
   rmin-based Si mid-planes used by `si_layer_centers()` were 0.1 mm shallow. Layer *assignment*
   was never affected (verified against the readout `cellID` for every hit); the constant is now
   applied explicitly (`ECAL_STACK_OFFSET_MM`).

Scope split (settled 2026-07-10): `main` carries the input-preparation side only —
`sim/make_pixelav_inputs.sh`, the deck converter, notebook 05c (input inspection) and
`docs/pixelav_handoff.md`. The PIXELAV build/run tooling in this log (`analysis/pixelav/`,
`setup/setup_pixelav.sh`, notebook 06, this log itself) stays on this branch by design; the
2026-07-09 fixes (clean-PYTHONPATH Stage-A invocation, the verification results above) are
committed here.
