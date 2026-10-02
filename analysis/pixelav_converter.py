"""DD4hep shower cascade  ->  PIXELAV track-segment input.

PIXELAV (M. Swartz) is a detailed silicon-pixel charge-transport simulation behind the CMS
SiPixelTemplate / generic-error templates. KEY POINT: PIXELAV is NOT an energy-deposit
consumer. Given a charged track's geometry + kinematics it GENERATES its own ionization
internally (Bichsel dE/dx + Landau straggling + delta rays), then drifts/diffuses/traps the
e-h pairs through the sensor's E and B fields. So we feed it TRACK SEGMENTS, not energy.
Full background, sources and the input-deck spec: docs/pixelav_reference.md.

PIXELAV is effectively TWO stages:
  STAGE A -- sensor/field model (one-time, per sensor; authored by hand or via TCADtoPixelAV,
  NOT by this converter): 3-D E-field map, thickness, bias/depletion V, temperature,
  mobility/Hall model, B-field, pixel pitch, trapping/radiation-damage params.
  STAGE B -- per-track event input (THIS CONVERTER), one record per charged-track crossing of
  a sensor, in that sensor's LOCAL frame:
    - cot(alpha)=p_u/p_w, cot(beta)=p_v/p_w   (w = sensor radial normal / depth axis)
    - momentum magnitude |p|           (sets betagamma / dE/dx regime: MIP vs soft e-)
    - particle type
    - local impact point (u, v) -- for the tracker path this is the energy-weighted
      mid-crossing position (~sensor mid-plane), NOT the entry face. NOTE: stock PIXELAV
      randomizes the impact over the central 3x3 pixels internally, so the impact is carried
      only as a LABEL (for matching) unless the PIXELAV wrapper is patched to read it (ours
      is: analysis/pixelav/ppixelav2_list_trkpy_real_entry.c). See docs/pixelav_reference.md.

One record per charged-track sensor crossing. PRIMARY = Variant C (auto-selected):
run_sim_trackermom.py reads the Si out as a Geant4 tracker, so each SimTrackerHit is one crossing
carrying the REAL momentum; build_segments_C turns them into records (entry, cot a/b, |p|, type) with
no reconstruction. FALLBACK = Variant A (calorimeter only): run_sim_fullcascade.py sets
enableDetailedShowerMode, so CaloHitContributions carry stepPosition + time (but NO momentum);
build_segments_A time-orders the steps per (MCParticle, face) into per-layer crossings, taking the
direction from the entry->exit displacement and the momentum from the production 4-vector. Variant B
is the coarsest pixel-centroid fallback. main() selects C if tracker hits are present, else A, else B.

GEOMETRY: 12-sided Si-W barrel, axis along z; face centres at k*30 deg (verified from data:
the +y beam strikes the 90 deg face; all this event's deposits are on it). Si layers sit at
constant DEPTH (perpendicular face distance). The depth/normal axis w is the per-face radial
direction; the across-pitch axis u is tangential (x-y plane); v is the cylinder-z axis. A tracker
hit's face and layer are its cellID module and layer when the npz holds them, else the face its
azimuth points at and the layer nearest in depth (sensor_crossings.tracker_hit_frame).

DECK: write_pixelav_deck() emits the Stage-B per-track list. Default layout 'badeaa3' = the 7-column
ppixelav2_list_trkpy_n_2f.c format (our driver lineage; the driver is analysis/pixelav/, built by
setup/setup_pixelav.sh); 'smartpix' = the 9-column ppixelav2_custom.c format. ppion is the
betagamma-matched pion momentum p*(m_pi/m_particle) so PIXELAV's pion dE/dx reproduces the real
particle's ionisation. Lengths in MICRONS (PIXELAV's unit; LENGTH_UNIT_MM=1000). The
<deck>.columns.txt legend defines every column.

Outputs: <out_prefix>.json + .csv (the sensor_crossings records), <out_prefix>.pixelav.txt (the
deck) and its .columns.txt. The default prefix sits beside the input npz:
trackermom_X.npz -> pixelav_segments_X.

Usage:
    python pixelav_converter.py [cascade.npz] [out_prefix] [--variant C|A|B|auto] [--layout smartpix|badeaa3]
"""
import argparse, os, sys
import numpy as np

# sensor_crossings.py holds the one definition of the geometry and the crossing builders, so the
# deck and the notebook 06/07 records come from the same code. Names this module does not use
# itself are imported too, so pixelav_converter.<name> keeps resolving for existing callers.
from sensor_crossings import (
    ECAL_RMIN_MM, ECAL_STACK_OFFSET_MM, ECAL_RMAX_MM, SI_THICK_MM, N_FACES, FACE_PHI0_DEG,
    _NEUTRAL_PDGS, load_cascade, si_layer_centers, si_thickness_mm, face_phi, to_local, is_charged,
    _record, default_npz, default_out_prefix,
    build_segments_A, build_segments_B, build_segments_C, write_intermediate,
)

# PIXELAV length unit relative to mm. PIXELAV works in MICRONS (firmly sourced from the source;
# see docs/pixelav_reference.md), so mm -> um is x1000. Records are stored in mm; this factor is
# applied ONLY when serializing the deck, so the intermediate table stays mm-consistent.
LENGTH_UNIT_MM = 1000.0

# Map our PDG codes to the PIXELAV PID column (the ppixelav2_custom.c wrapper handles 211/13/11;
# for non-pions it only rescales momentum by the mass ratio -- the dE/dx model stays pion-like).
_PIXELAV_PID = {11: 11, -11: 11, 13: 13, -13: 13, 211: 211, -211: 211}


def pid_to_pixelav(pdg):
    return _PIXELAV_PID.get(int(pdg), 211)   # default to pion (mass-rescale approximation)


# Rest masses (GeV) for the betagamma-matched pion momentum below; _deck_legend names the species.
_MASS_GEV = {11: 0.000510999, 13: 0.105658, 211: 0.139570, 321: 0.493677, 2212: 0.938272}  # e, mu, pi, K, p
M_PION_GEV = 0.139570


def ppion_betagamma_matched(p_gev, pdg):
    """PIXELAV models every track as a PION for dE/dx (Bichsel pion cross-sections). Ionization
    depends on betagamma = p/m, not on the species, so to reproduce OUR particle's dE/dx we hand
    PIXELAV the pion momentum with the SAME betagamma: ppion = p * m_pion / m_particle. For an
    electron this is ~273*p (its true relativistic plateau); feeding p directly would treat a soft
    electron as a slow pion and hugely over-ionize via the 1/beta^2 rise. Unknown species -> pion."""
    m = _MASS_GEV.get(abs(int(pdg)), M_PION_GEV)
    return p_gev * (M_PION_GEV / m)


# What the records of each variant hold (sensor_crossings.build_segments_A/B/C), as the deck
# legend states it: the direction behind cot_alpha/cot_beta/flipped, the impact point, |p|, time.
_VARIANT_TERMS = {
    "C": dict(
        slope_u="p_u/p_w, the direction slope du/dw",
        slope_v="p_v/p_w, the direction slope dv/dw",
        outward="p_w >= 0",
        impact=("the impact point, the hit's energy-weighted mid-crossing position (about the Si "
                "mid-plane), not the entry face"),
        p="the total |p| at the crossing",
        time="crossing time",
        route=("# these are the tracker-route (variant C) definitions; analysis/sensor_crossings.py "
               "also describes the calorimeter-route variants A and B")),
    "A": dict(
        slope_u=("du/dw, the direction slope of the displacement from the crossing's earliest to its "
                 "latest Geant4 step (of the production momentum when those are within 0.02 mm)"),
        slope_v="dv/dw, the direction slope from the same displacement or momentum",
        outward="dw >= 0",
        impact="the impact point, the position of the crossing's earliest Geant4 step",
        p="the particle's total |p| at production, not at the crossing",
        time="time of the crossing's earliest Geant4 step",
        route=("# these are the calorimeter-route variant A definitions (time-ordered Geant4 steps, "
               "which carry no momentum); analysis/sensor_crossings.py also describes variants B and C")),
    "B": dict(
        slope_u="p_u/p_w of the particle's production momentum, the direction slope du/dw",
        slope_v="p_v/p_w of the production momentum, the direction slope dv/dw",
        outward="p_w >= 0",
        impact=("the impact point, the energy-weighted centroid of the pixel hits holding the "
                "particle's deposits in the layer, not a point on the track"),
        p="the particle's total |p| at production, not at the crossing",
        time="0, variant B has no crossing time",
        route=("# these are the calorimeter-route variant B definitions (pixel hits and production "
               "momenta, the coarsest route); analysis/sensor_crossings.py also describes variants A and C")),
}


def _deck_legend(layout, si_thick_mm=None, variant="C"):
    """Text of <deck>.columns.txt: one line per deck column with its unit and definition, as the
    records of `variant` (A, B or C) define them."""
    v = _VARIANT_TERMS[variant]
    frame = ("# frame: sensor-local; u = across the pitch (tangential, in the global x-y plane), "
             "v = along the barrel axis (global z), w = outward normal of the sensor face")
    impact = v["impact"]
    common = [
        f"# col 1 cot_alpha          = {v['slope_u']} [dimensionless]",
        f"# col 2 cot_beta           = {v['slope_v']} [dimensionless]",
        "# col 3 ppion [GeV/c]      = |p| * m_pi/m, the pion momentum with the particle's betagamma "
        "(masses known for e, mu, pi, K and p and their antiparticles; any other species is "
        "treated as a pion, ppion = |p|)",
        f"# col 4 flipped            = 1 when moving outward along the sensor normal ({v['outward']}), "
        f"0 inward",
    ]
    if layout == "smartpix":
        names = "cot_alpha cot_beta ppion flipped ylocal zglobal pT hittime PID"
        cols = common + [
            f"# col 5 ylocal [um]        = v of {impact}; labelled ylocal, it holds the coordinate "
            f"along the barrel axis (equal to global z)",
            "# col 6 zglobal [um]       = u of the same point; labelled zglobal, it holds the local "
            "across-pitch coordinate, not a global one",
            f"# col 7 pT [GeV/c]         = named pT by the driver, but holds {v['p']}",
            f"# col 8 hittime [ps]       = {v['time']}",
            "# col 9 PID                = 11 for e+-, 13 for mu+-, 211 for every other species",
        ]
        driver = "ppixelav2_custom.c"
        reader = ("# stock PIXELAV randomizes the impact; the position columns are labels for matching "
                  "(docs/pixelav_reference.md)")
    else:
        names = "cot_alpha cot_beta ppion flipped modx mody pT"
        cols = common + [
            f"# col 5 modx [um]          = v of {impact}",
            "# col 6 mody [um]          = u of the same point",
            f"# col 7 pT [GeV/c]         = named pT by the driver, but holds {v['p']}",
        ]
        driver = "ppixelav2_list_trkpy_n_2f.c"
        reader = ("# stock PIXELAV randomizes the impact and ignores the position columns; the patched "
                  "analysis/pixelav/ppixelav2_list_trkpy_real_entry.c reads them (docs/pixelav_reference.md)")
    out = [f"# PIXELAV track list, layout {layout} ({driver} format): one track per line, "
           f"whitespace-separated; PIXELAV does not read this legend",
           names, frame] + cols + [
           v["route"],
           reader,
           "# crossings with a non-finite slope or |cot_alpha| or |cot_beta| > 10 are left out, so "
           "deck line k is not record k of the .json/.csv"]
    if si_thick_mm is not None:
        out.append(f"# sensor: {si_thick_mm * 1e3:g} um silicon; the PIXELAV Stage-A model must "
                   f"have this thickness")
    return "\n".join(out) + "\n"


def write_pixelav_deck(segs, out_path, layout="badeaa3", si_thick_mm=None, variant=None):
    """Emit a PIXELAV Stage-B per-track 'track list' (one whitespace-separated track per line).

    layout='badeaa3' (default): the 7-column ppixelav2_list_trkpy_n_2f.c format (read by our patched
        real-entry driver, analysis/pixelav/ppixelav2_list_trkpy_real_entry.c)
        cot_alpha  cot_beta  ppion  flipped  modx  mody  pT
    layout='smartpix': the 9-column ppixelav2_custom.c format (the Smart Pixels lineage)
        cot_alpha  cot_beta  ppion  flipped  ylocal  zglobal  pT  hittime  PID

    Driving fields = cot_alpha, cot_beta, ppion (GeV/c), flipped, PID. The length-label columns
    (ylocal/zglobal or modx/mody) carry the record's impact point (entry_u/entry_v; what it is
    depends on the variant, see the legend) in MICRONS so it survives into the PIXELAV output for
    matching -- but NOTE stock PIXELAV randomizes the impact
    over the central 3x3 pixels and does NOT consume them unless the wrapper is patched (see
    docs/pixelav_reference.md). hittime is written in ps. Grazing tracks (non-finite or |cot|>10)
    are skipped, as PIXELAV itself skips them, so deck line k is not record k. Also writes
    <out_path>.columns.txt (_deck_legend; NOT read by PIXELAV's fscanf), which records si_thick_mm
    when given: the deck carries no thickness, so the PIXELAV Stage-A sensor model has to be built
    for it. The legend defines the columns as the records' variant (A, B or C) fills them;
    variant=None takes it from the records' own "variant" field.
    """
    if layout not in ("badeaa3", "smartpix"):
        raise ValueError(f"unknown layout {layout!r} (use 'smartpix' or 'badeaa3')")
    if variant is None:
        found = {s["variant"] for s in segs}
        if len(found) > 1:
            raise ValueError(f"records of variants {sorted(found)}: one deck legend describes one variant")
        variant = found.pop() if found else "C"
    U = LENGTH_UNIT_MM
    lines, n_skip = [], 0
    for s in segs:
        ca, cb = s["cot_alpha"], s["cot_beta"]
        if not (np.isfinite(ca) and np.isfinite(cb)) or abs(ca) > 10.0 or abs(cb) > 10.0:
            n_skip += 1
            continue
        eu_um, ev_um, p = s["entry_u"] * U, s["entry_v"] * U, s["p_GeV"]
        ppion = ppion_betagamma_matched(p, s["pdg"])   # betagamma-matched pion momentum for dE/dx; pT label keeps the real |p|
        if layout == "smartpix":
            lines.append("%.6f %.6f %.6f %d %.4f %.4f %.6f %.4f %d" %
                         (ca, cb, ppion, s["flipped"], ev_um, eu_um, p, s["time_ns"] * 1000.0,
                          pid_to_pixelav(s["pdg"])))
        else:
            # 7-col format read by ppixelav2_list_trkpy_n_2f.c / our patched real-entry driver
            # (analysis/pixelav/, built by setup/setup_pixelav.sh). Axis map checked against the
            # driver source: col1 cot_alpha pairs with y(13-px, Lorentz) = our u and col6 mody;
            # col2 cot_beta pairs with x(21-px) = our v and col5 modx. Impact is written
            # full-truth in um; the patched driver reduces it mod-pitch to the sub-pixel impact.
            # ppion is the betagamma-matched pion momentum (dE/dx); the pT column keeps the real |p|.
            # flipped = 1 (outward, dw >= 0) -> the driver sets locdir_z > 0, entry face z = 0;
            # flipped = 0 -> entry face z = thick.
            lines.append("%.6f %.6f %.6f %d %.4f %.4f %.6f" %
                         (ca, cb, ppion, s["flipped"], ev_um, eu_um, p))
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    with open(out_path + ".columns.txt", "w") as f:
        f.write(_deck_legend(layout, si_thick_mm, variant))
    return out_path, len(lines), n_skip


def main(argv=None):
    ap = argparse.ArgumentParser(description="Crossing records (.json + .csv) and a PIXELAV deck "
                                             "from an extracted npz.")
    ap.add_argument("npz", nargs="?", help="extract_trackermom.py or extract_cascade.py output "
                    "(default: $CALOMAPS_HOME/models/trackermom_gamma50_1evt.npz, else fullcascade_...)")
    ap.add_argument("out_prefix", nargs="?", help="output prefix (default: pixelav_segments_X beside "
                    "the input trackermom_X.npz / fullcascade_X.npz)")
    ap.add_argument("--variant", default="auto", choices=("C", "A", "B", "auto"),
                    help="C tracker hits, A calorimeter steps, B pixel centroids (default: auto)")
    ap.add_argument("--layout", default="badeaa3", choices=("badeaa3", "smartpix"),
                    help="deck layout (default: badeaa3)")
    args = ap.parse_intermixed_args(argv)
    variant, layout = args.variant, args.layout
    npz = args.npz or default_npz()
    out_prefix = args.out_prefix or default_out_prefix(npz, "pixelav_segments")
    d = load_cascade(npz)

    has_tracker = ("thx" in d) and len(np.atleast_1d(d["thx"])) > 0
    has_calo = ("cmc" in d) and ("cbeg" in d)   # calo-contribution arrays (extract_cascade.py)
    has_steps = has_calo and ("csx" in d) and bool(np.any((d["csx"] != 0) | (d["csy"] != 0) | (d["csz"] != 0)))
    if variant == "auto":
        variant = "C" if has_tracker else ("A" if has_steps else "B")   # C = real per-crossing momentum
    if variant == "C" and not has_tracker:
        print("WARNING: Variant C requested but tracker hits absent -> falling back.")
        variant = "A" if has_steps else "B"
    if variant == "A" and not has_steps:
        print("WARNING: Variant A requested but stepPosition is zero/absent -> falling back to B.")
        variant = "B"
    if variant in ("A", "B") and not has_calo:
        sys.exit(f"ERROR: variant {variant} needs the calorimeter-contribution arrays "
                 "(cmc/cbeg/cend/cE/hx...) which this npz does not have -- it looks like a "
                 "tracker-readout extraction (extract_trackermom.py). Use --variant C (or auto), "
                 "or point at a fullcascade npz from extract_cascade.py.")

    builder = {"A": build_segments_A, "B": build_segments_B, "C": build_segments_C}[variant]
    segs, stats = builder(d)
    j, c = write_intermediate(segs, out_prefix)
    # Only an npz that records its geometry gets a sensor line; one without it was simulated
    # at SI_THICK_MM.
    thick = si_thickness_mm(d) if "geo_si_thick_mm" in d else None
    deck, n_deck, n_skip = write_pixelav_deck(segs, out_prefix + ".pixelav.txt", layout=layout,
                                              si_thick_mm=thick, variant=variant)

    n_charged = int(is_charged(d["pdg"]).sum())
    n_src = len(np.atleast_1d(d["thx"])) if ("thx" in d) else (len(d["cpdg"]) if "cpdg" in d else 0)
    src_label = "tracker-hit crossings" if ("thx" in d) else "step-contributions"
    print(f"cascade: {npz}")
    print(f"variant {variant}: {len(segs)} per-sensor charged-track crossings "
          f"(from {n_charged} charged MCParticles, {n_src} {src_label})")
    if stats:
        print(f"  stats: {stats}")
    if segs:
        layers = sorted({s['layer_id'] for s in segs})
        per_layer = {L: sum(1 for s in segs if s['layer_id'] == L) for L in layers}
        print(f"  crossings span layers {layers[0]}..{layers[-1]}; layer counts: "
              + ", ".join(f"L{L}:{per_layer[L]}" for L in layers[:6]) + (" ..." if len(layers) > 6 else ""))
        print(f"  example record: {segs[0]}")
    else:
        print("WARNING: zero crossings -- check the cascade / geometry.")
    print(f"wrote per-crossing table:\n  {j}\n  {c}")
    print(f"wrote PIXELAV deck ({layout}, lengths in um): {deck}  "
          f"[{n_deck} tracks, {n_skip} grazing skipped]  (+ {deck}.columns.txt)")
    if thick is not None:
        print(f"  sensor: {thick * 1e3:g} um silicon (npz geometry provenance); build the PIXELAV "
              f"Stage-A model for this thickness")


if __name__ == "__main__":
    main()
