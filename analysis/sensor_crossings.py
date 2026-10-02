"""Per-sensor charged-track crossings from the DECAL shower simulation.

Every charged particle in a shower that traverses one of the 30 silicon layers is one
"crossing". This module reduces the simulation output to one record per crossing, holding
exactly what a detailed sensor-level simulation needs to re-simulate that traversal:

  - WHERE:  the impact point in the sensor's local frame (u across-pitch, v along the barrel
            axis, w through the thickness) and the azimuth of the face it is on. Records hold
            no global coordinates; x = w cos(phi) - u sin(phi), y = w sin(phi) + u cos(phi), z = v
  - WHICH WAY: the direction of traversal, as local direction slopes du/dw and dv/dw
            (the cotangents of the track's inclination to the sensor plane -- the standard
            pixel-sensor convention) plus the sign along the sensor normal
  - HOW FAST: the true momentum magnitude |p| at that crossing (sets the ionization regime:
            a relativistic MIP deposits very differently from a soft delta-ray-like electron)
  - WHO:    the PDG particle type, the crossing time, and the deposited energy as a label

Provenance of the numbers (two independent routes, cross-checked in notebooks 06/07):
  PRIMARY (variant C): sim/run_sim_trackermom.py reads the ECal silicon out as a Geant4
  TRACKER (Geant4TrackerWeightedAction), so each SimTrackerHit is one combined sensor
  crossing carrying the TRUE Geant4 momentum at that crossing -- no reconstruction at all.
  NOTE: the action's hit position is the midpoint of the hit's first and last step points
  (HitPositionCombination 2), which for a straight through-going track is the point where it
  crosses the Si mid-plane, NOT the entry face; the record keeps the entry_u/entry_v field
  names, but read them as that point.
  CROSS-CHECK (variant A): sim/run_sim_fullcascade.py keeps the silicon as a CALORIMETER
  with per-step truth (enableDetailedShowerMode); steps carry position + time but no
  momentum, so build_segments_A time-orders the steps of each (particle, face) into
  per-layer crossings, taking the direction from the entry->exit displacement and the
  momentum from the particle's production 4-vector. Variant B is a coarser pixel-centroid
  fallback used only when neither source is available.

GEOMETRY: 12-sided Si-W barrel, axis along z; face centres at k*30 deg (verified from
data: the +y beam strikes the 90-deg face). Si layers sit at constant DEPTH -- the
perpendicular distance from the barrel axis to the flat face -- so the local depth axis w
is the per-face outward normal, u is tangential (in the x-y plane) and v is the cylinder
z. The staves overlap past the 30-degree sector boundaries, so near a boundary a hit can
sit on the neighbouring stave. A tracker hit's face and layer are its cellID module and layer
when the npz holds them (tracker_hit_frame); otherwise, and for the calorimeter route (A, B),
the face is the one the position's azimuth points at (face_phi) and the layer the one nearest
in depth, which misassigns hits in those overlap regions.

RECORDS: one dict per crossing, fields in this order (build_segments_A/B/C via _record):
  field              unit   meaning
  track_id           -      index of the producing particle in the npz MCParticle arrays (pdg, px...)
  layer_id           -      Si layer 0..29: the hit's cellID layer (C, when the npz has it), else
                            the layer whose mid-plane is nearest in depth
  pdg                -      PDG code of the particle
  p_GeV              GeV/c  raw momentum magnitude |p|: at the crossing (C, the mean of the momenta at
                            the hit's first and last step points), the particle's production |p|
                            (A, B). Not the betagamma-matched ppion of a PIXELAV deck
  entry_u, entry_v   mm     impact point in the sensor-local frame (u across the pitch, v along the
                            barrel axis = global z). C: the midpoint of the hit's first and last
                            step points, which for a straight through-going track is where it
                            crosses the Si mid-plane, not the entry face; A: the crossing's
                            earliest step; B: the energy-weighted pixel centroid
  cot_alpha          -      du/dw of the direction: of the momentum at the crossing (C, = p_u/p_w),
                            of the entry->exit displacement (A; the production momentum when flags
                            holds dir_from_momentum), of the production momentum (B)
  cot_beta           -      dv/dw, from the same direction
  flipped            -      1 = moving outward along the face normal (dw >= 0), 0 = inward
  sensor_normal_phi  rad    azimuth of the face's outward normal in the global x-y plane
  depth_w_mm         mm     w of the impact point: distance from the barrel axis along that normal
  energy_dep_GeV     GeV    energy the crossing deposited in the Si (a label)
  n_steps            -      Geant4 steps combined into the crossing (1 for C)
  time_ns            ns     crossing time: the hit time (C), the earliest step's (A), 0 (B)
  variant            -      "A", "B" or "C"
  flags              -      "|"-separated: tracker_hit (C), dir_from_momentum (A),
                            pixel_centroid (B), grazing
  evt_id             -      event of the crossing: its entry number in the simulation's ROOT file
                            (the npz's evt_id_hit (C) or evt_id_particle (A, B)); 0 for an npz
                            without event ids, which holds one event
  path_length_mm     mm     C: the hit's Geant4 path length in the silicon (pathLength, summed
                            over the hit's steps); A, B: NaN, no true path length is taken
A grazing crossing (direction parallel to the sensor plane, |dw| <= 1e-9) has cot_alpha =
cot_beta = inf, written as Infinity in the .json and inf in the .csv, and the flag grazing. NaN is
written as NaN in the .json and nan in the .csv.

An npz of several events (extract_trackermom.py) concatenates their particles and hits; track_id
indexes the concatenated arrays, so records sorted by track_id come grouped by event, in event order.

The records are written as <out_prefix>.json + .csv; by default beside the input npz
(trackermom_X.npz -> sensor_crossings_X). analysis/pixelav_converter.py builds the PIXELAV deck
(PIXELAV: the silicon-pixel charge-transport simulation our collaborators run) from the same
records. Build decks with it rather than by hand: it converts to microns, swaps |p| for the
betagamma-matched pion momentum and leaves out the grazing crossings.

Usage:
    python sensor_crossings.py [crossings.npz] [out_prefix] [--variant C|A|B|auto]
"""
import argparse, json, math, os, re, sys
import numpy as np

import decal_geometry

ECAL_RMIN_MM = 1264.0
# The ECalBarrel_o2_v03 driver does NOT start the layer stack at rmin: it places the stave at
# mod_y_off = inner_r + ry + trd_z + tolerance (k4geo ECalBarrel_o2_v03_geo.cpp), where
# tolerance = ecal_barrel_tolerance = env_safety = 0.1 mm (SiD_TestBeam.xml) and ry = 0 (no
# support rails). So every layer sits 0.1 mm beyond the naive rmin-based depth. Verified
# empirically on the 320 um baseline: tracker-hit depths cluster at +0.100 mm from the naive
# mid-planes and span exactly +/-0.16 mm (half of 320 um) around the corrected ones.
ECAL_STACK_OFFSET_MM = 0.1
# Outer apothem = inner apothem + stack offset + total radial stack depth (20 x 3.75 mm +
# 10 x 6.25 mm layers; pitches match si_layer_centers() / geometry/my_custom_ecal.xml).
ECAL_RMAX_MM = ECAL_RMIN_MM + ECAL_STACK_OFFSET_MM + 20 * 3.75 + 10 * 6.25   # = 1401.6 mm
SI_THICK_MM = 0.32   # baseline sensor thickness (320 um); si_thickness_mm(d) prefers the npz's own
N_FACES = 12
FACE_PHI0_DEG = 0.0           # face-centre azimuths at k*30 deg (verified: +y beam face is 90 deg)
# The face of a hit's stave is the cellID field `module`: module m has its outward normal at
# FACE_PHI0_DEG + (MODULE0_FACE - m) * 360/N_FACES, i.e. 300 - 30*m deg. This map is measured
# from the data (module 7 holds the 90-deg +y beam face; modules 0, 2-7, 9, 10 seen), not taken
# from the driver; validate_crossings.py tests it on every run, since a wrong face puts hits
# outside their slab.
MODULE0_FACE = 10
CELLID_ENCODING_KEY = "{}__CellIDEncoding"   # metadata-tree key of a collection's cellID layout
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# The sim wrappers append _si<T>um to every output name of a silicon-thickness variant run.
_SI_TAG = re.compile(r"_si([0-9]+(?:[.][0-9]+)?)um(?![0-9A-Za-z])")

# Species that leave no reconstructable track in silicon: neutrals (no primary ionization)
# and nuclei (slow recoils, not minimum-ionizing tracks; PDG |code| > 1e9), which is why the
# nuclei are excluded even though they are formally charged.
_NEUTRAL_PDGS = {22, 2112, -2112, 130, 310, 311, -311, 12, -12, 14, -14, 16, -16,
                 111, 221,                      # pi0/eta (decay instantly; belt-and-braces)
                 3122, -3122, 3212, -3212, 3322, -3322}  # neutral hyperons (Lambda, Sigma0, Xi0)


def load_cascade(npz_path):
    try:
        return np.load(npz_path, allow_pickle=True)
    except (FileNotFoundError, OSError, ValueError) as e:
        raise FileNotFoundError(
            f"Cannot load .npz '{npz_path}': {e}. Run analysis/extract_trackermom.py "
            "(or extract_cascade.py) first."
        ) from None


def si_layer_centers():
    """Per-layer Si depths (mm): perpendicular (face-normal) distance from the barrel axis to
    each sensor mid-plane (the apothem, built from ECAL_RMIN_MM); compared against the local depth w.

    The slices below are the 320 um baseline stack. The mid-planes they give hold for any
    ECal_Si_thickness, because sim/make_geometry_variant.py moves the thickness change into the
    air pads ECal_Si_air_in/out, half on each side. Other changes to the layer stack generally
    move them; the extractors refuse a geometry whose mid-planes differ (geometry_provenance),
    so update the slices here first.
    The stack starts at rmin + ECAL_STACK_OFFSET_MM (driver placement tolerance, see above); the
    offset is uniform so it does not affect nearest-centre layer ASSIGNMENT, only absolute depths."""
    r, centers = ECAL_RMIN_MM + ECAL_STACK_OFFSET_MM, []
    for nrep, w in [(20, 2.5), (10, 5.0)]:
        pitch = w + 0.25 + 0.32 + 0.05 + 0.30 + 0.33   # W + air + Si + Cu + Kapton + air
        for _ in range(nrep):
            centers.append(r + w + 0.25 + 0.16)        # Si mid-plane = after W + air + half-Si
            r += pitch
    return np.array(centers)


def si_thickness_mm(d):
    """Sensor silicon thickness (mm) the npz `d` was simulated with: its geo_si_thick_mm, or
    SI_THICK_MM for an npz extracted before the extractors stored geometry provenance."""
    return float(d["geo_si_thick_mm"]) if "geo_si_thick_mm" in d else SI_THICK_MM


def geometry_provenance(home):
    """The geo_* keys the extractors store in their npz, read from the config root `home` whose
    geometry/ was simulated (the repo, or a sim/make_geometry_variant.py copy):
      geo_si_thick_mm    sensor silicon thickness
      geo_si_centers_mm  per-layer Si mid-plane apothems as the compact defines them (the driver
                         places the stack ECAL_STACK_OFFSET_MM further out; si_layer_centers()
                         includes that offset)
      geo_cell_size_mm   pixel pitch
      geo_home           the config root, as an absolute path
    Aborts (SystemExit) when the compact's mid-planes are not the ones si_layer_centers()
    assumes, because every crossing would then be assigned against the wrong layer centres."""
    try:
        centers = decal_geometry.layer_radii(home)
        thick = decal_geometry.si_thickness_mm(home)
        cell = decal_geometry.cell_size_mm(home)
    except FileNotFoundError:
        raise SystemExit(f"--home {home!r} has no geometry/SiD_TestBeam.xml + my_custom_ecal.xml -- "
                         f"point it at the config root that was simulated")
    except ValueError as e:
        raise SystemExit(f"cannot read the geometry under --home {home!r}: {e}")
    expected = si_layer_centers() - ECAL_STACK_OFFSET_MM
    if centers.shape != expected.shape:
        raise SystemExit(f"{home}/geometry has {centers.size} Si layers; the crossing code "
                         f"(sensor_crossings.si_layer_centers) assumes {expected.size}")
    off = ~(np.abs(centers - expected) <= 1e-6)
    if off.any():
        k = int(np.argmax(off))
        raise SystemExit(f"Si mid-planes in {home}/geometry differ from the ones the crossing code "
                         f"assumes (sensor_crossings.si_layer_centers): layer {k} at {centers[k]:.6f} mm "
                         f"vs {expected[k]:.6f} mm. Only the Si thickness, changed with "
                         f"sim/make_geometry_variant.py, may differ without updating si_layer_centers().")
    return {"geo_si_thick_mm": np.float64(thick), "geo_si_centers_mm": centers,
            "geo_cell_size_mm": np.float64(cell), "geo_home": np.array(os.path.abspath(home))}


def thickness_tag_um(path):
    """The silicon thickness [um] an _si<T>um tag in path's file name names, or None."""
    tags = {float(t) for t in _SI_TAG.findall(os.path.basename(path))}
    if len(tags) > 1:
        raise SystemExit(f"{os.path.basename(path)}: more than one _si<T>um thickness tag")
    return tags.pop() if tags else None


def check_thickness_label(in_path, geo, home):
    """Exit unless the silicon thickness in_path's name implies equals the thickness of the --home
    geometry `home` (geo = geometry_provenance(home)). An _si<T>um tag names T; an untagged name
    means this checkout's own geometry (REPO). Every thickness variant has the baseline's Si
    mid-planes, so the hit positions cannot tell the two apart."""
    base_um = decal_geometry.si_thickness_mm(REPO) * 1e3
    tag_um = thickness_tag_um(in_path)
    named_um = base_um if tag_um is None else tag_um
    home_um = float(geo["geo_si_thick_mm"]) * 1e3
    if math.isclose(named_um, home_um, rel_tol=1e-5):
        return
    if tag_um is None:
        named = f"has no _si<T>um tag, which means this checkout's {base_um:g} um geometry"
    else:
        named = f"is tagged _si{tag_um:g}um"
    if math.isclose(named_um, base_um, rel_tol=1e-5):
        match = REPO
    else:
        match = (f"{decal_geometry.variant_home(named_um, REPO)} (the {named_um:g} um variant "
                 f"`python {REPO}/sim/make_geometry_variant.py --si-thickness-um {named_um:g}` writes)")
    raise SystemExit(f"{os.path.basename(in_path)} {named}, but --home {home} has {home_um:g} um "
                     f"silicon; nothing written. The --home for this input is {match}")


_RUN_TEXT_KEYS = {"sim_seed": "random.seed", "sim_dd4hep_version": "DD4hepVersion",
                  "sim_geant4_version": "GEANT4Version", "sim_date_utc": "DateUTC"}


def sim_provenance(root_file, in_path):
    """The sim_* keys the extractors store in their npz, from the ddsim run header of the open
    EDM4hep file root_file (uproot): its `runs` tree, GPStringKeys / GPStringValues.
      sim_seed            random.seed, as ddsim recorded it
      sim_n_events        numberOfEvents: the events ddsim was asked to simulate (-1 when unknown)
      sim_gun_particle    gun.particle
      sim_gun_energy_GeV  gun.energy, which ddsim records in MeV (NaN when unknown)
      sim_dd4hep_version, sim_geant4_version, sim_date_utc   DD4hepVersion, GEANT4Version, DateUTC
    The header is read only when it holds exactly one run. Without it, particle and energy come
    from in_path's name as the steering files write it (<prefix>_<tag><E>_..., the tag spelling
    pi+ as piplus); every other missing text field is "unknown". Returns (fields, source), source
    saying where particle and energy came from."""
    params = {}
    try:
        runs = root_file["runs"]
        if runs.num_entries == 1:
            keys = runs["GPStringKeys"].array(library="ak").tolist()[0]
            values = runs["GPStringValues"].array(library="ak").tolist()[0]
            params = {k: v[0] for k, v in zip(keys, values) if len(v) == 1}
    except (KeyError, IndexError, TypeError, ValueError):   # no run header, or not ddsim's layout
        params = {}
    out = {key: np.array(params.get(name, "unknown")) for key, name in _RUN_TEXT_KEYS.items()}
    try:
        n_events = int(params["numberOfEvents"])
    except (KeyError, ValueError):
        n_events = -1
    try:
        energy = float(params["gun.energy"]) / 1000.0
    except (KeyError, ValueError):
        energy = math.nan
    particle = params.get("gun.particle")
    source = "ddsim run header"
    if particle is None or not math.isfinite(energy):
        m = re.search(r"(?:trackermom|fullcascade)_([a-z]+?)([0-9.]+)_", os.path.basename(in_path))
        if particle is None:
            particle = m.group(1) if m else "unknown"
        if not math.isfinite(energy) and m:
            energy = float(m.group(2))
        source = "file name" if m else "unknown, no run header and no particle/energy in the file name"
    out.update(sim_n_events=np.int64(n_events), sim_gun_particle=np.array(particle),
               sim_gun_energy_GeV=np.float64(energy))
    return out, source


def default_out_prefix(npz, kind):
    """<kind>_X beside npz, X being its file name without .npz and without a leading trackermom_
    or fullcascade_: trackermom_gamma50_1evt_si100um.npz -> sensor_crossings_gamma50_1evt_si100um."""
    stem = os.path.basename(npz)
    stem = stem[:-4] if stem.endswith(".npz") else stem
    for lead in ("trackermom_", "fullcascade_"):
        if stem.startswith(lead):
            stem = stem[len(lead):]
            break
    return os.path.join(os.path.dirname(npz), f"{kind}_{stem}")


def face_phi(x, y):
    """Outward-normal azimuth (rad) of the dodecagon face a global (x,y) point sits on."""
    phi = np.degrees(np.arctan2(y, x))
    k = np.round((phi - FACE_PHI0_DEG) / (360.0 / N_FACES))
    return np.radians(FACE_PHI0_DEG + k * (360.0 / N_FACES))


def cellid_encoding(root_file, collection="ECalBarrelHits"):
    """The cellID layout string of `collection` (DD4hep's "name:offset:width,..."), from the
    `metadata` tree of the open EDM4hep file root_file (uproot), or None when it has none."""
    try:
        meta = root_file["metadata"]
        if meta.num_entries != 1:
            return None
        keys = meta["GPStringKeys"].array(library="ak").tolist()[0]
        values = meta["GPStringValues"].array(library="ak").tolist()[0]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    for k, v in zip(keys, values):
        if k == CELLID_ENCODING_KEY.format(collection) and len(v) == 1:
            return v[0]
    return None


def decode_cellid(cellid, encoding, names):
    """{name: int64 array} of the fields `names` of the uint64 cellIDs, laid out by `encoding`
    ("name:offset:width" or "name:width" per field, the latter starting where the previous field
    ends; a negative width is a signed field). Raises ValueError for a field the encoding lacks."""
    fields, nxt = {}, 0
    for item in encoding.split(","):
        parts = item.strip().split(":")
        if len(parts) == 3:
            offset, width = int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            offset, width = nxt, int(parts[1])
        else:
            raise ValueError(f"cellID encoding field {item!r} is not name:offset:width")
        fields[parts[0]] = (offset, width)
        nxt = offset + abs(width)
    c = np.asarray(cellid).astype(np.uint64)
    out = {}
    for name in names:
        if name not in fields:
            raise ValueError(f"cellID encoding {encoding!r} has no field {name!r}")
        offset, width = fields[name]
        n = abs(width)
        v = ((c >> np.uint64(offset)) & np.uint64((1 << n) - 1)).astype(np.int64)
        if width < 0:
            v = np.where(v >= (1 << (n - 1)), v - (1 << n), v)
        out[name] = v
    return out


def module_face_phi(module):
    """Outward-normal azimuth (rad) of the face of cellID module `module` (MODULE0_FACE), on the
    face_phi grid: the face index is taken into (-N_FACES/2, N_FACES/2], the range face_phi gives."""
    half = N_FACES // 2
    k = (MODULE0_FACE - np.asarray(module, np.int64) + half - 1) % N_FACES - (half - 1)
    return np.radians(FACE_PHI0_DEG + k * (360.0 / N_FACES))


def has_cellid_frame(d):
    """True when the tracker-route npz (or dict) d holds the hits' cellID module and layer."""
    return "tmodule" in d and "tlayer" in d


def tracker_hit_frame(d):
    """(face-normal azimuth [rad], depth w [mm], layer index) per tracker hit of the npz d.

    The one frame of the tracker hits: build_segments_C and validate_crossings.py both take it
    from here. Face and layer are the hit's cellID module and layer (tmodule, tlayer of
    extract_trackermom.py) when d has them. An npz without them falls back to the face the
    hit's azimuth points at (face_phi) and the layer nearest in depth, which puts a hit on the
    wrong face where neighbouring staves overlap past the 30-degree sector boundary. w is the
    hit's projection on the face normal."""
    thx, thy = np.asarray(d["thx"], float), np.asarray(d["thy"], float)
    centers = si_layer_centers()
    if has_cellid_frame(d):
        module, layer = np.asarray(d["tmodule"], np.int64), np.asarray(d["tlayer"], np.int64)
        bad_m = (module < 0) | (module >= N_FACES)
        bad_l = (layer < 0) | (layer >= len(centers))
        if bad_m.any() or bad_l.any():
            raise ValueError(f"tracker-hit cellIDs outside this barrel: modules "
                             f"{sorted(set(module[bad_m].tolist()))} (expected 0..{N_FACES - 1}), "
                             f"layers {sorted(set(layer[bad_l].tolist()))} (expected 0..{len(centers) - 1})")
        phin = module_face_phi(module)
        w = thx * np.cos(phin) + thy * np.sin(phin)
    else:
        phin = face_phi(thx, thy)
        w = thx * np.cos(phin) + thy * np.sin(phin)
        layer = np.argmin(np.abs(w[:, None] - centers[None, :]), axis=1)   # layers sit at constant depth, not r
    return phin, w, layer


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


def _event_ids(d, key):
    """The npz's per-hit or per-particle event ids (key), or None for an npz of one event
    extracted without them."""
    return np.asarray(d[key]) if key in d else None


def _record(mc, lay, pdg, p_mag, eu, ev, ew, phi_n, du, dv, dw, edep, nstep, variant, flags, time_ns,
            evt_id=0, path_mm=math.nan):
    cot_a, cot_b = (float(du / dw), float(dv / dw)) if abs(dw) > 1e-9 else (float("inf"), float("inf"))
    if abs(dw) <= 1e-9:
        flags = (flags + "|grazing").strip("|")
    return {
        "track_id": int(mc), "layer_id": int(lay), "pdg": int(pdg), "p_GeV": float(p_mag),
        "entry_u": float(eu), "entry_v": float(ev),           # mm (sensor-local IMPACT point; for
                                                              # variant C the midpoint of the tracker
                                                              # hit's first and last step points, the
                                                              # mid-plane point of a straight
                                                              # through-going track, NOT the entry face)
        "cot_alpha": cot_a, "cot_beta": cot_b,                # direction slopes du/dw, dv/dw
        "flipped": int(dw >= 0),       # direction sign along the sensor normal: 1 = outward-going
        "sensor_normal_phi": float(phi_n), "depth_w_mm": float(ew),
        "energy_dep_GeV": float(edep), "n_steps": int(nstep), "time_ns": float(time_ns),
        "variant": variant, "flags": flags,
        "evt_id": int(evt_id), "path_length_mm": float(path_mm),
    }


# ==========================================================================================
# Variant A -- per-sensor crossings from time-ordered Geant4 step-level truth (calo route)
# ==========================================================================================
def build_segments_A(d):
    cmc = np.asarray(d["cmc"]); ct = np.asarray(d["ctime"])
    csx, csy, csz = np.asarray(d["csx"]), np.asarray(d["csy"]), np.asarray(d["csz"]); cE = np.asarray(d["cE"])
    pdg_all = d["pdg"]; px_all, py_all, pz_all = d["px"], d["py"], d["pz"]
    evt_p = _event_ids(d, "evt_id_particle")
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
                                float(cE[run].sum()), len(run), "A", flags, float(ct[e]),
                                0 if evt_p is None else evt_p[mc]))
    segs.sort(key=lambda s: (s["track_id"], s["layer_id"]))
    return segs, {"neutral_groups_skipped": n_neutral, "dir_from_momentum": n_single}


# ==========================================================================================
# Variant B -- fallback when no step positions: pixel hits + MCParticle momentum (coarser)
# ==========================================================================================
def build_segments_B(d):
    cbeg, cend, cmc, cE = np.asarray(d["cbeg"]), np.asarray(d["cend"]), np.asarray(d["cmc"]), np.asarray(d["cE"])
    hx, hy, hz = np.asarray(d["hx"]), np.asarray(d["hy"]), np.asarray(d["hz"])
    pdg_all = d["pdg"]; px_all, py_all, pz_all = d["px"], d["py"], d["pz"]
    evt_p = _event_ids(d, "evt_id_particle")
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
                            eu, ev, ew, phi_n, du, dv, dw, float(wgt.sum()), len(js), "B", "pixel_centroid", 0.0,
                            0 if evt_p is None else evt_p[mc]))
    segs.sort(key=lambda s: (s["track_id"], s["layer_id"]))
    return segs, {}


# ==========================================================================================
# Variant C -- per-crossing records from tracker-SD SimTrackerHits (REAL Geant4 momentum)
# ==========================================================================================
def build_segments_C(d):
    """One record per Si crossing from the tracker-readout sim (sim/run_sim_trackermom.py): the
    ECal Si is read out as a Geant4 tracker, so each SimTrackerHit is one combined sensor crossing
    carrying the TRUE Geant4 momentum at that crossing. So |p|, the direction slopes and the
    impact point are all real per-crossing truth -- no reconstruction, no production-momentum
    fallback, no step time-ordering needed (Geant4TrackerWeightedAction already combines the
    crossing's steps). NOTE: the hit position is the midpoint of the hit's first and last step
    points, which for a straight through-going track is where it crosses the Si mid-plane, not
    the entry-face point; the record keeps the entry_u/entry_v field names, but treat them as
    that point."""
    thx, thy, thz = np.asarray(d["thx"]), np.asarray(d["thy"]), np.asarray(d["thz"])
    tpx, tpy, tpz = np.asarray(d["tpx"]), np.asarray(d["tpy"]), np.asarray(d["tpz"])
    tedep, ttime, tmc = np.asarray(d["tedep"]), np.asarray(d["ttime"]), np.asarray(d["tmc"]).astype(int)
    tpath = np.asarray(d["tpath"])
    evt_h = _event_ids(d, "evt_id_hit")
    pdg_all = d["pdg"]
    assert tmc.size == 0 or (tmc.min() >= 0 and tmc.max() < len(pdg_all)), "trackerhit->MCParticle index out of range"
    phin, _, layer = tracker_hit_frame(d)     # face and layer: the validator's frame
    segs, n_neutral = [], 0
    for j in range(len(thx)):
        mc = int(tmc[j]); pdg = int(pdg_all[mc])
        if not bool(is_charged(pdg)):
            n_neutral += 1
            continue
        phi_n = float(phin[j])
        eu, ev, ew = to_local(float(thx[j]), float(thy[j]), float(thz[j]), phi_n)   # hit position (local)
        du, dv, dw = to_local(float(tpx[j]), float(tpy[j]), float(tpz[j]), phi_n)    # momentum -> local frame
        p_mag = float(np.sqrt(tpx[j]**2 + tpy[j]**2 + tpz[j]**2))                     # REAL |p| at the crossing
        segs.append(_record(mc, int(layer[j]), pdg, p_mag, eu, ev, ew, phi_n, du, dv, dw,
                            float(tedep[j]), 1, "C", "tracker_hit", float(ttime[j]),
                            0 if evt_h is None else evt_h[j], float(tpath[j])))
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


def default_npz():
    """$CALOMAPS_HOME/models (else this checkout's models/): the tracker-route gamma 50 GeV npz,
    else the calorimeter-route one."""
    home = os.environ.get("CALOMAPS_HOME", REPO)
    npz = os.path.join(home, "models", "trackermom_gamma50_1evt.npz")   # prefer real-momentum source
    return npz if os.path.exists(npz) else os.path.join(home, "models", "fullcascade_gamma50_1evt.npz")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-sensor charged-track crossing records (.json + .csv) "
                                             "from an extracted npz.")
    ap.add_argument("npz", nargs="?", help="extract_trackermom.py or extract_cascade.py output "
                    "(default: $CALOMAPS_HOME/models/trackermom_gamma50_1evt.npz, else fullcascade_...)")
    ap.add_argument("out_prefix", nargs="?", help="output prefix (default: sensor_crossings_X beside "
                    "the input trackermom_X.npz / fullcascade_X.npz)")
    ap.add_argument("--variant", default="auto", choices=("C", "A", "B", "auto"),
                    help="C tracker hits, A calorimeter steps, B pixel centroids (default: auto)")
    args = ap.parse_intermixed_args(argv)
    variant = args.variant
    npz = args.npz or default_npz()
    out_prefix = args.out_prefix or default_out_prefix(npz, "sensor_crossings")
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

    n_charged = int(is_charged(d["pdg"]).sum())
    n_src = len(np.atleast_1d(d["thx"])) if ("thx" in d) else (len(d["cpdg"]) if "cpdg" in d else 0)
    src_label = "tracker-hit crossings" if ("thx" in d) else "step-contributions"
    print(f"input: {npz}")
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
        print("WARNING: zero crossings -- check the input .npz / geometry.")
    print(f"wrote per-crossing table:\n  {j}\n  {c}")


if __name__ == "__main__":
    main()
