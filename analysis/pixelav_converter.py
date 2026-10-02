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
    - local impact point (u, v) -- for the tracker path this is the midpoint of the hit's
      first and last step points, which for a straight through-going track is where it
      crosses the sensor mid-plane, NOT the entry face. NOTE: stock PIXELAV
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

SELECTION: PIXELAV simulates every deck line as a straight track through the full silicon
thickness at constant momentum. deck_drop_reason() names what keeps a crossing from being one:
grazing or steep (the driver makes no cluster), outside_si, and a true path length (Geant4's)
more than --path-tolerance away from that straight chord (path_short, path_long). In 50 GeV
photon showers that is most crossings below 3 MeV/c and almost none above 30 MeV/c. --select
through (default) writes only the crossings with none of these; --select all writes everything
the driver processes. The counts go into the legend.

Outputs: <out_prefix>.json + .csv (the sensor_crossings records, every one of them, plus
deck_line, the record's 1-based line in the deck or 0, and deck_drop, why it is not there),
<out_prefix>.pixelav.txt (the deck) and its .columns.txt. The default prefix sits beside the
input npz: trackermom_X.npz -> pixelav_segments_X.

Usage:
    python pixelav_converter.py [cascade.npz] [out_prefix] [--variant C|A|B|auto] [--layout smartpix|badeaa3]
                                [--select through|all] [--path-tolerance TOL]
"""
import argparse, math, os, sys
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


# The driver writes no cluster for a deck line with |cot_alpha| or |cot_beta| above this
# (analysis/pixelav/ppixelav2_list_trkpy_real_entry.c: `if(fabsf(cotbeta) > 10.) goto incr;`,
# the same for cotalpha), so such a line would have no cluster to match.
DRIVER_MAX_COT = 10.0
PATH_TOLERANCE = 0.05      # --path-tolerance default
# deck_drop_reason's reasons in the order it tests them, and the ones each --select mode leaves out.
DROP_REASONS = ("grazing", "steep", "outside_si", "path_short", "path_long")
SELECT_DROPS = {"through": DROP_REASONS, "all": ("grazing", "steep")}
SELECT_MEANING = {
    "through": ("through = PIXELAV simulates every deck line as a straight track through the full "
                "silicon thickness at constant momentum; a crossing whose true path length in the "
                "silicon (Geant4's) differs from that straight line's by more than the path "
                "tolerance is not written, nor is a grazing, steep or outside_si one"),
    "all": ("all = every crossing the driver simulates is written (only grazing and steep ones are "
            "left out), also those whose true path differs from the straight line PIXELAV simulates"),
}


def deck_drop_reason(s, thick_mm, path_tol=PATH_TOLERANCE):
    """Why the crossing record s is not what PIXELAV simulates for a deck line, or None when it
    is. PIXELAV makes every line a straight track through the full thick_mm of silicon at
    constant momentum. The first test that fails names the reason:
      grazing     cot_alpha or cot_beta is not finite (a direction in the sensor plane)
      steep       |cot_alpha| or |cot_beta| > DRIVER_MAX_COT: the driver makes no cluster for it
      outside_si  the flags hold outside_si: the position lies outside its Si slab
      path_short  path_length_mm < (1 - path_tol) * chord, chord = thick_mm * sqrt(1 + cot_alpha^2
                  + cot_beta^2) being the straight line's length in the silicon: the particle
                  starts or stops in the silicon (flags starts_in_si, stops_in_si), or its
                  momentum (for C the mean of the momenta at the hit's first and last step
                  points) points more steeply than the line it took through the silicon
      path_long   path_length_mm > (1 + path_tol) * chord: multiple scattering, a curling soft
                  electron, a re-entry
    A record without a path length (NaN or absent: variants A and B) passes both path tests.
    SELECT_DROPS[mode] lists the reasons that keep a record out of a deck of --select mode."""
    ca, cb = s["cot_alpha"], s["cot_beta"]
    if not (math.isfinite(ca) and math.isfinite(cb)):
        return "grazing"
    if abs(ca) > DRIVER_MAX_COT or abs(cb) > DRIVER_MAX_COT:
        return "steep"
    if "outside_si" in s.get("flags", "").split("|"):
        return "outside_si"
    path = s.get("path_length_mm", math.nan)
    if math.isfinite(path):
        ratio = path / (thick_mm * math.sqrt(1.0 + ca * ca + cb * cb))
        if ratio < 1.0 - path_tol:
            return "path_short"
        if ratio > 1.0 + path_tol:
            return "path_long"
    return None


def selection_report(segs, reasons, select, path_tol, thick_mm):
    """Lines (no leading '#') counting the deck selection: segs are the records, reasons their
    deck_drop_reason (thick_mm, path_tol), select the --select mode."""
    n = len(segs)
    e = np.array([s["energy_dep_GeV"] for s in segs], float)
    e_tot = float(e.sum())
    dropped = SELECT_DROPS[select]

    def share(mask):
        k = int(np.count_nonzero(mask))
        return k, 100.0 * k / n if n else 0.0, 100.0 * float(e[mask].sum()) / e_tot if e_tot > 0 else 0.0

    why = np.array([r or "" for r in reasons], dtype=object)
    path = np.array([s.get("path_length_mm", math.nan) for s in segs], float)
    written = np.array([r not in dropped for r in why], bool)
    k, pc, pe = share(written)
    out = [f"deck selection: {select}, path tolerance {path_tol:g}, chord for {thick_mm * 1e3:g} um silicon",
           SELECT_MEANING[select],
           f"crossings {n}; written {k} ({pc:.1f}% of the crossings, {pe:.1f}% of their deposited energy)",
           f"  {'reason':<13}{'crossings':>10}{'% cross':>9}{'% energy':>10}  deck"]
    for r in DROP_REASONS + ("path_unknown",):
        mask = (why == "") & ~np.isfinite(path) if r == "path_unknown" else why == r
        k, pc, pe = share(mask)
        line = f"  {r:<13}{k:>10}{pc:>8.1f}%{pe:>9.1f}%  {'left out' if r in dropped else 'written'}"
        if r == "path_short" and k:
            fl = [set(s.get("flags", "").split("|")) for s, m in zip(segs, mask) if m]
            st = sum("starts_in_si" in f for f in fl)
            sp = sum("stops_in_si" in f for f in fl)
            nei = sum(not ({"starts_in_si", "stops_in_si"} & f) for f in fl)
            line += f"; starts_in_si {st}, stops_in_si {sp}, neither {nei}"
        if r == "path_unknown":
            line += "; no path length (variants A, B)"
        out.append(line)
    return out


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
        impact=("the impact point, the midpoint of the hit's first and last step points, which for "
                "a straight through-going track is where it crosses the Si mid-plane; not the "
                "entry face"),
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


def _deck_legend(layout, si_thick_mm=None, variant="C", selection=()):
    """Text of <deck>.columns.txt: one line per deck column with its unit and definition, as the
    records of `variant` (A, B or C) define them, then the lines of `selection` as comments."""
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
        axes = []
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
        # Pairing read from ppixelav2_list_trkpy_real_entry.c: locdir[0] = cotbeta*locdir[2] and
        # vect[0] from modx with pitch xsize; locdir[1] = cotalpha*locdir[2] and vect[1] from mody
        # with ysize; pixel[TXSIZE=21][TYSIZE=13]; modx, mody reduced mod xsize, ysize at z = thick/2.
        axes = [
            "# axes (ppixelav2_list_trkpy_real_entry.c): PIXELAV x, the sensor model's first pitch (xsize) "
            "and the 21-pixel side of the output window, is v (along the barrel axis), steered by cot_beta "
            "(col 2) and placed by modx (col 5); PIXELAV y, the second pitch (ysize) and the 13-pixel side, "
            "is u (across the stave), steered by cot_alpha (col 1) and placed by mody (col 6)",
            "# that driver folds modx and mody into one pixel of the sensor model and puts the track's "
            "mid-plane point there: the simulated pixel pitch is the sensor model's, not the calorimeter "
            "readout's (geo_cell_size_mm)"]
    out = [f"# PIXELAV track list, layout {layout} ({driver} format): one track per line, "
           f"whitespace-separated; PIXELAV does not read this legend",
           names, frame] + cols + [v["route"], reader] + axes + [
           "# deck line k is the record with deck_line == k in the converter's .json/.csv beside it; a "
           "record without a line has deck_line 0 and the reason in deck_drop"]
    out += [f"# {line}" for line in selection]
    if si_thick_mm is not None:
        out.append(f"# sensor: {si_thick_mm * 1e3:g} um silicon; the PIXELAV Stage-A model must "
                   f"have this thickness")
    return "\n".join(out) + "\n"


def write_pixelav_deck(segs, out_path, layout="badeaa3", si_thick_mm=None, variant=None,
                       select="through", path_tol=PATH_TOLERANCE):
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
    docs/pixelav_reference.md). hittime is written in ps.

    A record whose deck_drop_reason (thickness si_thick_mm, else SI_THICK_MM, the thickness of an
    npz without geometry provenance; tolerance path_tol) is in SELECT_DROPS[select] gets no line,
    so deck line k is not record k. Also writes <out_path>.columns.txt (_deck_legend; NOT read by
    PIXELAV's fscanf) with the selection counts, and with si_thick_mm when given: the deck carries
    no thickness, so the PIXELAV Stage-A sensor model has to be built for it. The legend defines
    the columns as the records' variant (A, B or C) fills them; variant=None takes it from the
    records' own "variant" field.

    Returns (out_path, number of deck lines, deck_drop, report): deck_drop holds, per record of
    segs, the reason it has no line or "" when it has one; report the selection_report lines.
    """
    if layout not in ("badeaa3", "smartpix"):
        raise ValueError(f"unknown layout {layout!r} (use 'smartpix' or 'badeaa3')")
    if select not in SELECT_DROPS:
        raise ValueError(f"unknown selection {select!r} (use {' or '.join(SELECT_DROPS)})")
    if not (math.isfinite(path_tol) and path_tol >= 0):
        raise ValueError(f"path tolerance must be a non-negative number, got {path_tol!r}")
    if variant is None:
        found = {s["variant"] for s in segs}
        if len(found) > 1:
            raise ValueError(f"records of variants {sorted(found)}: one deck legend describes one variant")
        variant = found.pop() if found else "C"
    thick = SI_THICK_MM if si_thick_mm is None else si_thick_mm
    reasons = [deck_drop_reason(s, thick, path_tol) for s in segs]
    deck_drop = [r if r in SELECT_DROPS[select] else "" for r in reasons]
    U = LENGTH_UNIT_MM
    lines = []
    for s, why in zip(segs, deck_drop):
        if why:
            continue
        ca, cb = s["cot_alpha"], s["cot_beta"]
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
    report = selection_report(segs, reasons, select, path_tol, thick)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    with open(out_path + ".columns.txt", "w") as f:
        f.write(_deck_legend(layout, si_thick_mm, variant, report))
    return out_path, len(lines), deck_drop, report


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
    ap.add_argument("--select", default="through", choices=tuple(SELECT_DROPS),
                    help="through: only crossings PIXELAV's straight full-thickness track reproduces; "
                         "all: every crossing the driver simulates (default: through)")
    ap.add_argument("--path-tolerance", type=float, default=PATH_TOLERANCE, metavar="TOL",
                    help="largest |path length / straight chord - 1| written with --select through "
                         f"(default: {PATH_TOLERANCE:g})")
    args = ap.parse_intermixed_args(argv)
    if not (math.isfinite(args.path_tolerance) and args.path_tolerance >= 0):
        ap.error(f"--path-tolerance must be a non-negative number, got {args.path_tolerance:g}")
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
    # Only an npz that records its geometry gets a sensor line; one without it was simulated
    # at SI_THICK_MM, which write_pixelav_deck then takes for the selection.
    thick = si_thickness_mm(d) if "geo_si_thick_mm" in d else None
    deck, n_deck, deck_drop, report = write_pixelav_deck(
        segs, out_prefix + ".pixelav.txt", layout=layout, si_thick_mm=thick, variant=variant,
        select=args.select, path_tol=args.path_tolerance)
    k = 0
    for s, why in zip(segs, deck_drop):
        k += not why
        s["deck_line"], s["deck_drop"] = (0 if why else k), why
    j, c = write_intermediate(segs, out_prefix)

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
          f"[{n_deck} tracks, {len(segs) - n_deck} crossings left out]  (+ {deck}.columns.txt)")
    for line in report:
        print(f"  {line}")
    if thick is not None:
        print(f"  sensor: {thick * 1e3:g} um silicon (npz geometry provenance); build the PIXELAV "
              f"Stage-A model for this thickness")


if __name__ == "__main__":
    main()
