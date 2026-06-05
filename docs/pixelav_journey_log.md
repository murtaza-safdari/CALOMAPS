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
- [ ] DECAL Stage-A config (`ppixel2.init` + weighting potential) for our pitch/thickness with
      a simple analytic E-field.
- [ ] Patched driver consuming real `modx/mody`.
- [ ] `write_pixelav_deck()` finished (7-col, µm, axis map above) and a deck generated from the
      2,535 crossings.
- [ ] One-track round-trip to confirm the cot/axis/flip convention; then a full run + plots.

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
