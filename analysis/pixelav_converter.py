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
      is: analysis/pixelav/ppixelav2_list_trkpy_real_entry.c on the pixelav-integration
      branch). See docs/pixelav_reference.md.

EXPERIMENT "B": one record per charged-track sensor crossing. PRIMARY = Variant C (auto-selected):
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
direction; the across-pitch axis u is tangential (x-y plane); v is the cylinder-z axis. The
per-face normal azimuth phi_n is derived from the hit position (not +y-hardcoded).

DECK: write_pixelav_deck() emits the Stage-B per-track list. Default layout 'badeaa3' = the 7-column
ppixelav2_list_trkpy_n_2f.c format (our driver lineage; the driver itself lives on the
pixelav-integration branch: analysis/pixelav/ + setup/setup_pixelav.sh); 'smartpix' = the 9-column ppixelav2_custom.c
format. ppion is the betagamma-matched pion momentum p*(m_pi/m_particle) so PIXELAV's pion dE/dx
reproduces the real particle's ionisation. Lengths in MICRONS (PIXELAV's unit; LENGTH_UNIT_MM=1000).

Usage:
    python pixelav_converter.py [cascade.npz] [out_prefix] [--variant C|A|B|auto] [--layout smartpix|badeaa3]
"""
import os, sys, json
import numpy as np

ECAL_RMIN_MM = 1264.0
# The ECalBarrel_o2_v03 driver does NOT start the layer stack at rmin: it places the stave at
# mod_y_off = inner_r + ry + trd_z + tolerance (k4geo ECalBarrel_o2_v03_geo.cpp), where
# tolerance = ecal_barrel_tolerance = env_safety = 0.1 mm (SiD_TestBeam.xml) and ry = 0 (no
# support rails). So every layer sits 0.1 mm beyond the naive rmin-based depth. Verified
# empirically: tracker-hit depths cluster at +0.100 mm from the naive mid-planes and span
# exactly +/-0.16 mm (the Si half-thickness) around the corrected ones.
ECAL_STACK_OFFSET_MM = 0.1
# Outer apothem = inner apothem + stack offset + total radial stack depth (20 x 3.75 mm +
# 10 x 6.25 mm layers; pitches match si_layer_centers() / geometry/my_custom_ecal.xml).
# Used by the nb05b schematic.
ECAL_RMAX_MM = ECAL_RMIN_MM + ECAL_STACK_OFFSET_MM + 20 * 3.75 + 10 * 6.25   # = 1401.6 mm
SI_THICK_MM = 0.32   # silicon sensor thickness (320 um); matches geometry/my_custom_ecal.xml
N_FACES = 12
FACE_PHI0_DEG = 0.0           # face-centre azimuths at k*30 deg (verified: +y beam face is 90 deg)

# PIXELAV length unit relative to mm. PIXELAV works in MICRONS (firmly sourced from the source;
# see docs/pixelav_reference.md), so mm -> um is x1000. Records are stored in mm; this factor is
# applied ONLY when serializing the deck, so the intermediate table stays mm-consistent.
LENGTH_UNIT_MM = 1000.0

# Species PIXELAV should NOT process: neutrals (no primary ionization) and nuclei (slow recoils,
# not minimum-ionizing tracks). Nuclei use PDG |code| > 1e9, which is why they are excluded here
# even though they are formally charged -- intentional for a pixel MIP simulation.
_NEUTRAL_PDGS = {22, 2112, -2112, 130, 310, 311, -311, 12, -12, 14, -14, 16, -16,
                 111, 221,                      # pi0/eta (decay instantly; belt-and-braces)
                 3122, -3122, 3212, -3212, 3322, -3322}  # neutral hyperons (Lambda, Sigma0, Xi0)

# Map our PDG codes to the PIXELAV PID column (the ppixelav2_custom.c wrapper handles 211/13/11;
# for non-pions it only rescales momentum by the mass ratio -- the dE/dx model stays pion-like).
_PIXELAV_PID = {11: 11, -11: 11, 13: 13, -13: 13, 211: 211, -211: 211}


def pid_to_pixelav(pdg):
    return _PIXELAV_PID.get(int(pdg), 211)   # default to pion (mass-rescale approximation)


# Rest masses (GeV) for the betagamma-matched pion momentum below.
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


def load_cascade(npz_path):
    try:
        return np.load(npz_path, allow_pickle=True)
    except (FileNotFoundError, OSError, ValueError) as e:
        raise FileNotFoundError(
            f"Cannot load cascade .npz '{npz_path}': {e}. Run analysis/extract_cascade.py first."
        ) from None


def si_layer_centers():
    """Per-layer Si depths (mm): perpendicular (face-normal) distance from the barrel axis to
    each sensor mid-plane (the apothem, built from ECAL_RMIN_MM); compared against the local depth w.

    NOTE: the slice thicknesses below MUST match geometry/my_custom_ecal.xml -- the single source of
    truth that nb02's layer_radii() parses from the XML. They agree today; if you change the geometry
    (pitch / thickness scans), update them here too (ideally refactor layer_radii() into a shared module).
    The stack starts at rmin + ECAL_STACK_OFFSET_MM (driver placement tolerance, see above); the
    offset is uniform so it does not affect nearest-centre layer ASSIGNMENT, only absolute depths."""
    r, centers = ECAL_RMIN_MM + ECAL_STACK_OFFSET_MM, []
    for nrep, w in [(20, 2.5), (10, 5.0)]:
        pitch = w + 0.25 + 0.32 + 0.05 + 0.30 + 0.33   # W + air + Si + Cu + Kapton + air
        for _ in range(nrep):
            centers.append(r + w + 0.25 + 0.16)        # Si mid-plane = after W + air + half-Si
            r += pitch
    return np.array(centers)


def face_phi(x, y):
    """Outward-normal azimuth (rad) of the dodecagon face a global (x,y) point sits on."""
    phi = np.degrees(np.arctan2(y, x))
    k = np.round((phi - FACE_PHI0_DEG) / (360.0 / N_FACES))
    return np.radians(FACE_PHI0_DEG + k * (360.0 / N_FACES))


def to_local(x, y, z, phi_n):
    """Global (x,y,z) -> sensor-local (u across-pitch, v along-z, w radial/depth)."""
    c, s = np.cos(phi_n), np.sin(phi_n)
    w = x * c + y * s          # radial (depth / normal)
    u = -x * s + y * c         # tangential (across pitch)
    v = z                      # cylinder-z
    return u, v, w


def is_charged(pdg):
    pdg = np.asarray(pdg)
    return (~np.isin(pdg, list(_NEUTRAL_PDGS))) & (np.abs(pdg) < 1_000_000_000)


def _record(mc, lay, pdg, p_mag, eu, ev, ew, phi_n, du, dv, dw, edep, nstep, variant, flags, time_ns):
    cot_a, cot_b = (float(du / dw), float(dv / dw)) if abs(dw) > 1e-9 else (float("inf"), float("inf"))
    if abs(dw) <= 1e-9:
        flags = (flags + "|grazing").strip("|")
    return {
        "track_id": int(mc), "layer_id": int(lay), "pdg": int(pdg), "p_GeV": float(p_mag),
        "entry_u": float(eu), "entry_v": float(ev),           # mm (sensor-local IMPACT point; for
                                                              # variant C this is the tracker hit's
                                                              # energy-weighted mid-crossing position
                                                              # (~mid-plane), NOT the entry face)
        "cot_alpha": cot_a, "cot_beta": cot_b,
        "flipped": int(dw >= 0),       # 1 = outward-going (dw>=0). VERIFIED vs the patched driver
                                       # (pixelav-integration branch,
                                       # analysis/pixelav/ppixelav2_list_trkpy_real_entry.c):
                                       # flipped=1 -> locdir_z>0, entry face z=0; flipped=0 -> z=thick.
        "sensor_normal_phi": float(phi_n), "depth_w_mm": float(ew),
        "energy_dep_GeV": float(edep), "n_steps": int(nstep), "time_ns": float(time_ns),
        "variant": variant, "flags": flags,
    }


# ==========================================================================================
# Variant A -- per-sensor crossings from time-ordered Geant4 step-level truth (experiment "B")
# ==========================================================================================
def build_segments_A(d):
    cmc = np.asarray(d["cmc"]); ct = np.asarray(d["ctime"])
    csx, csy, csz = np.asarray(d["csx"]), np.asarray(d["csy"]), np.asarray(d["csz"]); cE = np.asarray(d["cE"])
    pdg_all = d["pdg"]; px_all, py_all, pz_all = d["px"], d["py"], d["pz"]
    assert cmc.size == 0 or (cmc.min() >= 0 and cmc.max() < len(pdg_all)), "contribution->MCParticle index out of range"

    centers = si_layer_centers()
    phin = face_phi(csx, csy)
    wdep = csx * np.cos(phin) + csy * np.sin(phin)               # DEPTH = projection on face normal
    layer = np.argmin(np.abs(wdep[:, None] - centers[None, :]), axis=1)  # layers sit at constant depth, not r
    face = np.round((np.degrees(phin) - FACE_PHI0_DEG) / (360.0 / N_FACES)).astype(int) % N_FACES

    groups = {}
    for j in range(len(cmc)):
        groups.setdefault((int(cmc[j]), int(face[j])), []).append(j)

    segs, n_neutral, n_single = [], 0, 0
    for (mc, fc), js in groups.items():
        pdg = int(pdg_all[mc])
        if not bool(is_charged(pdg)):
            n_neutral += 1
            continue
        js = sorted(js, key=lambda k: ct[k])                     # time order
        phi_n = float(np.radians(FACE_PHI0_DEG + fc * (360.0 / N_FACES)))
        p_mag = float(np.sqrt(px_all[mc]**2 + py_all[mc]**2 + pz_all[mc]**2))
        # split the time-ordered steps into maximal runs of constant Si layer = one crossing each
        runs, cur = [], [js[0]]
        for k in js[1:]:
            if layer[k] == layer[cur[-1]]:
                cur.append(k)
            else:
                runs.append(cur)
                cur = [k]
        runs.append(cur)
        for run in runs:
            lay = int(layer[run[0]])
            e, x = run[0], run[-1]                               # entry = earliest time, exit = latest
            disp = np.array([csx[x] - csx[e], csy[x] - csy[e], csz[x] - csz[e]])
            flags = ""
            if np.linalg.norm(disp) > 0.02:                      # time-ordered traversal vector
                du, dv, dw = to_local(disp[0], disp[1], disp[2], phi_n)
            else:                                                # single/co-located steps -> production momentum
                du, dv, dw = to_local(px_all[mc], py_all[mc], pz_all[mc], phi_n)
                flags = "dir_from_momentum"; n_single += 1
            eu, ev, ew = to_local(csx[e], csy[e], csz[e], phi_n)
            segs.append(_record(mc, lay, pdg, p_mag, eu, ev, ew, phi_n, du, dv, dw,
                                float(cE[run].sum()), len(run), "A", flags, float(ct[e])))
    segs.sort(key=lambda s: (s["track_id"], s["layer_id"]))
    return segs, {"neutral_groups_skipped": n_neutral, "dir_from_momentum": n_single}


# ==========================================================================================
# Variant B -- fallback when no step positions: pixel hits + MCParticle momentum (coarser)
# ==========================================================================================
def build_segments_B(d):
    cbeg, cend, cmc, cE = np.asarray(d["cbeg"]), np.asarray(d["cend"]), np.asarray(d["cmc"]), np.asarray(d["cE"])
    hx, hy, hz = np.asarray(d["hx"]), np.asarray(d["hy"]), np.asarray(d["hz"])
    pdg_all = d["pdg"]; px_all, py_all, pz_all = d["px"], d["py"], d["pz"]
    centers = si_layer_centers()
    hit_of = np.full(len(cmc), -1, dtype=np.int64)               # contribution -> hit (pixel)
    for h in range(len(cbeg)):
        hit_of[cbeg[h]:cend[h]] = h
    phiH = face_phi(hx, hy); wH = hx * np.cos(phiH) + hy * np.sin(phiH)   # hit depth (face normal)
    layerH = np.argmin(np.abs(wH[:, None] - centers[None, :]), axis=1)
    groups = {}
    for j in range(len(cmc)):
        h = int(hit_of[j])
        if h < 0:                                                # contribution outside any hit range -> skip
            continue
        groups.setdefault((int(cmc[j]), int(layerH[h])), []).append(j)
    segs = []
    for (mc, lay), js in groups.items():
        pdg = int(pdg_all[mc])
        if not bool(is_charged(pdg)):
            continue
        js = np.array(js); hh = hit_of[js]; wgt = cE[js]; wsum = wgt.sum() if wgt.sum() > 0 else 1.0
        ex = float((hx[hh]*wgt).sum()/wsum); ey = float((hy[hh]*wgt).sum()/wsum); ez = float((hz[hh]*wgt).sum()/wsum)
        phi_n = float(face_phi(ex, ey))
        eu, ev, ew = to_local(ex, ey, ez, phi_n)
        du, dv, dw = to_local(px_all[mc], py_all[mc], pz_all[mc], phi_n)
        segs.append(_record(mc, lay, pdg, float(np.sqrt(px_all[mc]**2+py_all[mc]**2+pz_all[mc]**2)),
                            eu, ev, ew, phi_n, du, dv, dw, float(wgt.sum()), len(js), "B", "pixel_centroid", 0.0))
    segs.sort(key=lambda s: (s["track_id"], s["layer_id"]))
    return segs, {}


# ==========================================================================================
# Variant C -- per-crossing records from tracker-SD SimTrackerHits (REAL Geant4 momentum)
# ==========================================================================================
def build_segments_C(d):
    """One record per Si crossing from the tracker-readout sim (sim/run_sim_trackermom.py): the
    ECal Si is read out as a Geant4 tracker, so each SimTrackerHit is one combined sensor crossing
    carrying the TRUE Geant4 momentum at that crossing. So |p|, the direction (cot a/b) and the
    impact point are all real per-crossing truth -- no production-momentum fallback, no step
    time-ordering needed (Geant4TrackerWeightedAction already combines the crossing's steps).
    NOTE: the hit position is the action's energy-weighted COMBINED position (~sensor mid-plane
    for a through-going track), not the entry-face point; the record keeps the historical
    entry_u/entry_v field names, but treat them as the mid-crossing impact."""
    thx, thy, thz = np.asarray(d["thx"]), np.asarray(d["thy"]), np.asarray(d["thz"])
    tpx, tpy, tpz = np.asarray(d["tpx"]), np.asarray(d["tpy"]), np.asarray(d["tpz"])
    tedep, ttime, tmc = np.asarray(d["tedep"]), np.asarray(d["ttime"]), np.asarray(d["tmc"]).astype(int)
    pdg_all = d["pdg"]
    assert tmc.size == 0 or (tmc.min() >= 0 and tmc.max() < len(pdg_all)), "trackerhit->MCParticle index out of range"
    centers = si_layer_centers()
    phin = face_phi(thx, thy)
    wdep = thx * np.cos(phin) + thy * np.sin(phin)                       # depth = projection on face normal
    layer = np.argmin(np.abs(wdep[:, None] - centers[None, :]), axis=1)  # layers sit at constant depth, not r
    segs, n_neutral = [], 0
    for j in range(len(thx)):
        mc = int(tmc[j]); pdg = int(pdg_all[mc])
        if not bool(is_charged(pdg)):
            n_neutral += 1
            continue
        phi_n = float(phin[j])
        eu, ev, ew = to_local(float(thx[j]), float(thy[j]), float(thz[j]), phi_n)   # weighted mid-crossing impact (local)
        du, dv, dw = to_local(float(tpx[j]), float(tpy[j]), float(tpz[j]), phi_n)    # momentum -> local frame
        p_mag = float(np.sqrt(tpx[j]**2 + tpy[j]**2 + tpz[j]**2))                     # REAL |p| at the crossing
        segs.append(_record(mc, int(layer[j]), pdg, p_mag, eu, ev, ew, phi_n, du, dv, dw,
                            float(tedep[j]), 1, "C", "tracker_hit", float(ttime[j])))
    segs.sort(key=lambda s: (s["track_id"], s["layer_id"]))
    return segs, {"neutral_hits_skipped": n_neutral, "n_tracker_hits": int(len(thx))}


def write_intermediate(segs, out_prefix):
    with open(out_prefix + ".json", "w") as f:
        json.dump(segs, f, indent=2)
    if segs:
        cols = list(segs[0].keys())
        with open(out_prefix + ".csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for s in segs:
                f.write(",".join(str(s[c]) for c in cols) + "\n")
    return out_prefix + ".json", out_prefix + ".csv"


def write_pixelav_deck(segs, out_path, layout="badeaa3"):
    """Emit a PIXELAV Stage-B per-track 'track list' (one whitespace-separated track per line).

    layout='badeaa3' (default): the 7-column ppixelav2_list_trkpy_n_2f.c format (read by our patched
        real-entry driver, analysis/pixelav/ppixelav2_list_trkpy_real_entry.c on the
        pixelav-integration branch)
        cot_alpha  cot_beta  ppion  flipped  modx  mody  pT
    layout='smartpix': the 9-column ppixelav2_custom.c format (the Smart Pixels lineage)
        cot_alpha  cot_beta  ppion  flipped  ylocal  zglobal  pT  hittime  PID

    Driving fields = cot_alpha, cot_beta, ppion (GeV/c), flipped, PID. The length-label columns
    (ylocal/zglobal or modx/mody) carry the per-crossing truth entry point in MICRONS so it
    survives into the PIXELAV output for matching -- but NOTE stock PIXELAV randomizes the impact
    over the central 3x3 pixels and does NOT consume them unless the wrapper is patched (see
    docs/pixelav_reference.md). hittime is written in ps. Grazing tracks (non-finite or |cot|>10)
    are skipped, as PIXELAV itself skips them. Also writes <out_path>.columns.txt (a legend; NOT
    read by PIXELAV's fscanf).
    """
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
        elif layout == "badeaa3":
            # 7-col format read by ppixelav2_list_trkpy_n_2f.c / our patched real-entry driver
            # (pixelav-integration branch: analysis/pixelav/, built by setup/setup_pixelav.sh).
            # Axis map VERIFIED against the
            # driver source: col1 cot_alpha pairs with y(13-px, Lorentz) = our u and col6 mody;
            # col2 cot_beta pairs with x(21-px) = our v and col5 modx. Impact is written
            # full-truth in um; the patched driver reduces it mod-pitch to the sub-pixel impact.
            # ppion is the betagamma-matched pion momentum (dE/dx); the pT column keeps the real |p|.
            lines.append("%.6f %.6f %.6f %d %.4f %.4f %.6f" %
                         (ca, cb, ppion, s["flipped"], ev_um, eu_um, p))
        else:
            raise ValueError(f"unknown layout {layout!r} (use 'smartpix' or 'badeaa3')")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    legend = {"smartpix": "cot_alpha cot_beta ppion flipped ylocal[um] zglobal[um] pT hittime[ps] PID",
              "badeaa3":  "cot_alpha cot_beta ppion flipped modx[um]=v_entry mody[um]=u_entry pT"}[layout]
    with open(out_path + ".columns.txt", "w") as f:
        f.write(f"# PIXELAV track-list ({layout}); lengths in microns; modx/mody carry the truth "
                f"mid-plane impact as a LABEL (stock PIXELAV randomizes the impact; our patched "
                f"real-entry driver consumes it -- see docs/pixelav_reference.md)\n{legend}\n")
    return out_path, len(lines), n_skip


def main():
    home = os.environ.get("CALOMAPS_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    argv = sys.argv[1:]
    variant, layout, pos = "auto", "badeaa3", []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--variant"):
            variant = a.split("=", 1)[1] if "=" in a else (argv[i + 1] if i + 1 < len(argv) else variant)
            i += 0 if "=" in a else 1
        elif a.startswith("--layout"):
            layout = a.split("=", 1)[1] if "=" in a else (argv[i + 1] if i + 1 < len(argv) else layout)
            i += 0 if "=" in a else 1
        else:
            pos.append(a)
        i += 1
    default_npz = os.path.join(home, "models", "trackermom_gamma50_1evt.npz")   # prefer real-momentum source
    if not os.path.exists(default_npz):
        default_npz = os.path.join(home, "models", "fullcascade_gamma50_1evt.npz")
    npz = pos[0] if len(pos) > 0 else default_npz
    out_prefix = pos[1] if len(pos) > 1 else os.path.join(home, "models", "pixelav_segments_gamma50_1evt")
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
    deck, n_deck, n_skip = write_pixelav_deck(segs, out_prefix + ".pixelav.txt", layout=layout)

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


if __name__ == "__main__":
    main()
