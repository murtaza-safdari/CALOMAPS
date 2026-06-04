"""DD4hep shower cascade  ->  PIXELAV track-segment input  (SCAFFOLD).

PIXELAV (M. Swartz) is a detailed silicon-pixel charge-transport simulation behind the
CMS SiPixelTemplate / generic-error templates. KEY POINT (often misunderstood): PIXELAV
is NOT an energy-deposit consumer. Given a charged track's geometry + kinematics it
GENERATES its own ionization internally (Bichsel dE/dx + Landau straggling + delta-ray
production), then drifts/diffuses/traps the e-h pairs through the sensor's E and B fields
to produce the induced pixel signal. So we feed it TRACK SEGMENTS, never the Geant4
energy deposits. This dictates the converter's whole shape.

PIXELAV is effectively TWO stages — do not conflate them:

  STAGE A — sensor/field model (one-time, per sensor; authored by hand, NOT by this
  converter): 3-D electric-field map (or params to build it), sensor thickness, bias /
  depletion voltage, temperature, mobility/Hall model, magnetic-field vector, pixel
  pitch (x,y), trapping/radiation-damage params. This is the classic config deck.

  STAGE B — per-track event input (THE FILE THIS CONVERTER WRITES): one record per
  charged-track crossing of a sensor, in that sensor's LOCAL frame:
    - local entry point (u, v) on the entry face          [across-pitch, along-z]
    - direction as cot(alpha) = p_u/p_w and cot(beta) = p_v/p_w, w = sensor-normal/depth.
      (The standard CMS convention writes this as cot(alpha)=px/pz, cot(beta)=py/pz for a
      Z-NORMAL sensor. Our +y test wedge has the depth axis w=y, NOT z — see the geometry
      note below, where the implemented formula is cot(alpha)=px/py, cot(beta)=pz/py.)
      PIXELAV propagates the straight line across the FULL depletion depth itself, so it
      wants entry + 2 cot angles (NOT an arbitrary exit z).
    - momentum magnitude |p| (sets the betagamma / dE/dx regime: MIP vs soft e-)
    - particle type / charge (MEDIUM confidence it matters beyond |p|; confirm in source)

UNITS ARE UNCONFIRMED (cm vs µm vs mm): Swartz's code works largely in cm, while CMS
template tooling often quotes µm. Do NOT hardcode — see LENGTH_UNIT_MM below and confirm
against the real PIXELAV source before trusting absolute scales.

THE GAP (why this is a scaffold, not the finished converter):
  Experiment "A" gives us each shower particle's *production* 4-vector + vertex, and the
  calorimeter hits (energy deposits). PIXELAV wants the track's *local entry/exit and
  direction as it crosses a specific sensor*. Bridging that needs the per-sensor
  step-level truth (experiment "B"): associate each silicon hit with the track that made
  it (the Geant4Calorimeter::Hit `truth` MonteCarloContrib carries the depositing track
  id), and use the hit's entry/exit step points. Until "B" is extracted, this module:
    * implements the local-frame direction math for the +y test wedge (exact there),
    * builds an APPROXIMATE per-track record from the cascade (production vertex as a
      proxy entry point, production momentum as the direction),
    * emits an intermediate, well-defined track-segment table (CSV/JSON),
    * STUBS write_pixelav_deck() until the real deck format is confirmed from source.

Geometry note: in this single-photon test the beam is +y and the struck sensors' normal
is ~ +y, so the local depth axis w = y, the across-pitch axis u = x and the along-z axis
v = z. Hence cot(alpha) = px/py and cot(beta) = pz/py. For off-normal sensors (general
barrel position) a per-sensor rotation is required — stubbed in global_to_local().

Usage:
    python pixelav_converter.py [cascade.npz] [out_prefix]
"""
import os, sys, json
import numpy as np

# ==========================================================================================
# ⚠️  SCAFFOLD — NOT THE FINISHED CONVERTER.
#   * Output is an INTERMEDIATE track-segment table (JSON/CSV), NOT a runnable PIXELAV deck.
#   * write_pixelav_deck() raises NotImplementedError (deck format pending Swartz source).
#   * Track selection + entry point are APPROXIMATE (production vertex as proxy; one record
#     per charged track entering the ECal, NOT per silicon-layer crossing). The faithful
#     version needs experiment "B" (Geant4 step-level truth). See module docstring.
# ==========================================================================================
SCAFFOLD_STATUS = ("scaffold: intermediate track-segment table only; deck writer stubbed; "
                   "approximate production-vertex entry, +y-wedge geometry")

# ECal silicon radial extent for the +y wedge (mm); hits live at r in ~[1267, 1403].
ECAL_RMIN_MM = 1264.0
ECAL_RMAX_MM = 1410.0

# Output length unit relative to mm. PIXELAV's expected unit is UNCONFIRMED (cm vs µm vs
# mm) — set this once the real deck format is known (e.g. 0.1 for cm, 1000.0 for µm).
LENGTH_UNIT_MM = 1.0

# PROPER granularity (TODO): each (particle, silicon-LAYER) crossing is a separate
# PIXELAV track. The DECAL has 30 Si layers; a shower track pierces several. The faithful
# converter ("approx-rays" mode) ray-casts each charged track's production->end segment
# against every layer slab (straight lines; B=0) and emits one record per crossing, in
# that sensor's local frame. The fully-correct path ("stepB") uses Geant4 step-level
# truth (experiment "B"): per G4Step in a sensitive Si cell, the pre/post-step local
# points + momentum, associated to the cellID. This scaffold ships the simplest mode
# (one record per charged track entering the ECal, production vertex as proxy entry).


def load_cascade(npz_path):
    try:
        return np.load(npz_path, allow_pickle=True)
    except (FileNotFoundError, OSError, ValueError) as e:
        raise FileNotFoundError(
            f"Cannot load cascade .npz '{npz_path}': {e}. "
            f"Run analysis/extract_cascade.py first."
        ) from None


def select_silicon_tracks(d):
    """Charged tracks that plausibly cross the ECal silicon.

    Approximation (pending experiment 'B' hit-truth association): a charged particle
    (e+/e-) whose PRODUCTION vertex lies inside the ECal radial band is treated as a
    track that enters the silicon there. The proper version selects tracks that have an
    associated silicon hit and uses the hit entry/exit step points.

    WARNING: the radius check below uses the production-vertex Y only — valid ONLY for the
    +y pencil-beam wedge (where depth == y). For a general barrel sensor at another azimuth,
    replace `vsy` with the true radius and the per-sensor transform (see global_to_local).
    """
    pdg = d["pdg"]
    charged = (pdg == 11) | (pdg == -11)
    # for the +y wedge, "radius into the calorimeter" is the production-vertex y
    vsy = d["vsy"]
    in_ecal = (vsy > ECAL_RMIN_MM) & (vsy < ECAL_RMAX_MM)
    return np.where(charged & in_ecal)[0]


def global_to_local(px, py, pz, vsx, vsy, vsz):
    """Map a global momentum + vertex to the local sensor frame.

    EXACT for the +y test wedge (sensor normal w=y, u=x, v=z). For a general barrel
    sensor at azimuth phi this needs the sensor rotation R(phi); that lookup (from the
    DD4hep cellID / segmentation) is the main TODO for the production converter.

    Returns (u, v, cot_alpha, cot_beta) with depth axis = y.
    """
    # local entry point on the sensor face (proxy = production vertex transverse coords)
    u = vsx   # across pitch
    v = vsz   # along beam-z
    # direction cosines relative to the depth axis (y)
    with np.errstate(divide="ignore", invalid="ignore"):
        cot_alpha = np.where(py != 0, px / py, np.inf)
        cot_beta = np.where(py != 0, pz / py, np.inf)
    return u, v, cot_alpha, cot_beta


def build_track_segments(d):
    """Build the intermediate per-track-segment table from the cascade (approximate)."""
    idx = select_silicon_tracks(d)
    px, py, pz = d["px"][idx], d["py"][idx], d["pz"][idx]
    u, v, cot_a, cot_b = global_to_local(px, py, pz, d["vsx"][idx], d["vsy"][idx], d["vsz"][idx])
    pmag = np.sqrt(px**2 + py**2 + pz**2)
    segs = []
    for k, i in enumerate(idx):
        segs.append({
            "track_id": int(d["pid"][i]),
            "pdg": int(d["pdg"][i]),
            "p_GeV": float(pmag[k]),
            "entry_u": float(u[k] * LENGTH_UNIT_MM),   # local across-pitch (unit = LENGTH_UNIT_MM)
            "entry_v": float(v[k] * LENGTH_UNIT_MM),   # local along-z
            "cot_alpha": float(cot_a[k]),
            "cot_beta": float(cot_b[k]),
            "depth_y_mm": float(d["vsy"][i]),
        })
    return segs


def write_intermediate(segs, out_prefix):
    """Emit the well-defined intermediate track-segment table (JSON + CSV)."""
    with open(out_prefix + ".json", "w") as f:
        json.dump(segs, f, indent=2)
    if segs:
        cols = list(segs[0].keys())
        with open(out_prefix + ".csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for s in segs:
                f.write(",".join(str(s[c]) for c in cols) + "\n")
    return out_prefix + ".json", out_prefix + ".csv"


def write_pixelav_deck(segs, out_path):
    """STUB: emit a PIXELAV input deck.

    Deliberately not implemented — the exact PIXELAV deck syntax (field order, header,
    units) is not web-documented and must be confirmed against M. Swartz's PIXELAV
    source / a known-good example deck. Once confirmed, map each segment's
    (entry_u, entry_v, cot_alpha, cot_beta, p_GeV, pdg) onto the per-track lines and
    prepend the sensor/run configuration block. See module docstring + open questions.
    """
    raise NotImplementedError(
        "PIXELAV deck format pending Swartz source — see module docstring. "
        "Intermediate track-segment table is available via write_intermediate()."
    )


def main():
    home = os.environ.get("CALOMAPS_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(home, "models", "fullcascade_gamma50_1evt.npz")
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else os.path.join(home, "models", "pixelav_segments_gamma50_1evt")
    d = load_cascade(npz)
    n_charged = int(((d["pdg"] == 11) | (d["pdg"] == -11)).sum())
    segs = build_track_segments(d)
    j, c = write_intermediate(segs, out_prefix)
    print(f"[{SCAFFOLD_STATUS}]")
    print(f"cascade: {npz}")
    print(f"selected {len(segs)} / {n_charged} charged tracks in ECal band "
          f"[{ECAL_RMIN_MM}, {ECAL_RMAX_MM}] mm (approx, +y wedge)")
    if not segs:
        print("WARNING: zero tracks selected — check geometry band vs this event/detector.")
    print(f"wrote intermediate track-segment table:\n  {j}\n  {c}")
    print("PIXELAV deck writer is STUBBED (write_pixelav_deck) pending the deck format.")


if __name__ == "__main__":
    main()
