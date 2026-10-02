#!/usr/bin/env python3
"""Geant4 step deposits in the ECal silicon -> PIXELAV deposit jobs.

    python analysis/pixelav/deposit_jobs.py <step npz> <prefix> --run-dir <PIXELAV run dir>
    python analysis/pixelav/deposit_jobs.py <step npz> <prefix> --pitch-x-um 50 --pitch-y-um 25
                                            [--thick-um T] [--piece-um 5] [--max-pairs 100000]
                                            [--parts N] [--events A:B]
    python analysis/pixelav/deposit_jobs.py --check-sensor <config root> (--run-dir DIR | --pitch-x-um X
                                            --pitch-y-um Y [--thick-um T])

The input is an npz of step hits from analysis/extract_trackermom.py (hit_mode step:
sim/run_sim_trackermom.py with CALOMAPS_TRACKER_HITS=step makes every Geant4 step with a deposit
in the silicon one hit, at the midpoint of its pre- and post-step points, with the mean of their
momenta, the step length as pathLength and the step's deposit as EDep). A job hands such
deposits to PIXELAV's drift and readout stages in place of its own track ionisation: a list of
short straight segments, each with its energy, in the frame of one PIXELAV output window.

SEGMENTS: every step is a straight segment centred on the hit position, along the unit mean
momentum, as long as the step's pathLength, clipped in depth to the slab of silicon; its energy
stays whole on the clipped segment, whose ends are clipped into 0..thick in the window z. The
hit holds neither the pre- and post-step points nor the chord between them, which multiple
scattering makes shorter than the path, so a long curved step becomes a straight segment of its
path length, and part of it may be clipped away. The segments are therefore only as good as the
steps are short. The deposits route of sim/make_pixelav_inputs.sh limits Geant4 steps in the
silicon to 5 um (CALOMAPS_SI_STEP_UM, a limitset of the silicon's own); there, consecutive
segments of a charged particle chain end to start within 0.3 um for 90% of its steps. Without a
limit, steps run up to the calorimeter's 5 mm, and in 100 um silicon 30% of the steps of a 50
GeV photon shower, holding 46% of its energy, are clipped. The summary gives the share of steps,
and of their energy, longer than twice the piece length, and the npz's silicon step limit. The
segment is cut into equal pieces of at most --piece-um (default 5 um), each with its share of
the energy by length. Steps of neutral particles (photons absorbed in the silicon, neutrons) are
segments like any other; in 50 GeV photon showers they hold 0.01-0.04% of the deposited energy.

FRAME: sensor-local u (across the stave), v (along the barrel axis) and w (outward normal) of the
hit's cellID module and layer (sensor_crossings.tracker_hit_frame), in um. The window frame is
the stock deck's (pixelav_converter.py): PIXELAV x = v with the sensor model's first pitch
(xsize) and the WINDOW_X side of the window, y = u with the second pitch (ysize) and the WINDOW_Y
side, z = w - w_centre + thick/2, from 0 on the inner face to thick on the outer one. (v, u, w)
is the mirror image of the right-handed (u, v, w), which a sensor model without a magnetic field
cannot tell apart; a model with a field is refused.

TILES: every sensor plane has one pixel grid, pixel iv = floor(v/xsize), iu = floor(u/ysize),
covered by tiles of TILE_X x TILE_Y pixels: the WINDOW_X x WINDOW_Y window less a GUARD-pixel band
on every side, 17 x 9 for 21 x 13 and 2. Tile tv = floor(iv/TILE_X), tu = floor(iu/TILE_Y). A
piece belongs to the tile that holds its midpoint, so it reaches at most half a piece past the
tile; the guard band holds that and the charge that diffuses out. Window column 0 is global
pixel tv*TILE_X - GUARD and row 0 is tu*TILE_Y - GUARD, and window pixel (10, 6) is centred on
(0, 0):
    x = v - (tv*17 + 8 + 0.5)*xsize,   y = u - (tu*9 + 4 + 0.5)*ysize.

GROUPS: the independent charged particles of each sensor plane (event, module, layer). The steps
of one particle in one plane form a node. A node joins its parent's node in the same plane when
the particle was created inside that slab (its creation vertex within thickness/2 +
sensor_crossings.SLAB_TOL_MM of the mid-plane, along the plane's normal) and the parent is a
charged particle with steps in that plane; the top of such a chain is the group root. So delta
rays join the track that made them, and a photon converting in the silicon gives two roots.
created_in_slab marks a root that itself starts inside the slab (an electron from a photon
interaction, say).

JOBS: one per (group, tile). A job whose energy would make more than --max-pairs electron-hole
pairs at EV_PER_PAIR (default MAX_PAIRS) is split into jobs with the same window, each at most
--max-pairs: the driver's transport time grows faster than the pair count above about 1e5 pairs
per job, and it refuses a job that reaches DRIVER_PAIRS (its NEHSTORE). Job ids run from 0 in the
order (group, tile, split).

OUTPUTS:
  <prefix>.jobs.txt, or with --parts N the files <prefix>.part000.txt ... of roughly equal energy,
      each holding whole jobs in job_id order:
        J <job_id> <n_seg>
        <x1> <y1> <z1> <x2> <y2> <z2> <edep_eV>      n_seg lines, um in the window frame
  <prefix>.jobs.npz   per job (job_*): job_id, evt_id, module, layer, group_id, tile_v, tile_u,
      win_iv0, win_iu0 (the global pixel of window column 0 and row 0), n_seg, edep_keV, n_split
      (jobs its group and tile were split into), part; per group (group_*): group_id, evt_id,
      module, layer, root (the root particle's index in the npz particle arrays), root_pdg,
      root_p_MeV (|p| at the root's first step in the plane), n_particles, n_steps, edep_keV,
      path_um (sum of the step lengths), w_min_um, w_max_um (extent of the clipped segments in
      the window z, 0 to thick), u_centroid_um, v_centroid_um (energy-weighted), u_min_um,
      u_max_um, v_min_um, v_max_um (sensor-plane u, v), t_first_ns, created_in_slab, n_tiles;
      the scalars of the frame and the options; and the input npz's sim_* and geo_* keys,
      hit_mode, particles_kept, num_events and tcellID_encoding.
  <prefix>.summary.txt   the summary this script prints.
Nothing is written unless the energy of the jobs equals that of the steps (relative 1e-9) and
every piece lies within its window's tile, give or take half a piece.

Sensor facts come from a PIXELAV run directory's ppixel2.init and wgt_pot.init
(sensor_model.read_run_dir: thick, xsize, ysize), or from --pitch-x-um and --pitch-y-um with the
npz's silicon thickness (or --thick-um). A sensor thickness more than 1 um from the npz's is
refused. --check-sensor checks the sensor facts against the silicon of a config root and exits,
so a wrapper can stop before it simulates.
"""
import argparse, math, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import sensor_model                          # noqa: E402
import sensor_crossings as sc                # noqa: E402
import decal_geometry                        # noqa: E402

WINDOW_X, WINDOW_Y = 21, 13       # PIXELAV output window, pixels along x (v) and y (u)
GUARD = 2                         # pixels of the window around its tile, on every side
TILE_X, TILE_Y = WINDOW_X - 2 * GUARD, WINDOW_Y - 2 * GUARD
CENTRE_X, CENTRE_Y = WINDOW_X // 2, WINDOW_Y // 2   # the window pixel centred on (0, 0)
EV_PER_PAIR = 3.68                # the driver's EPAIR
MAX_PAIRS = 100_000               # default largest job, in pairs
DRIVER_PAIRS = 1_000_000          # the driver's NEHSTORE: a job must make fewer pairs
PIECE_UM = 5.0
THICK_TOL_UM = 1.0
CHUNK_EVENTS = 10                 # events per block; the block's pieces and text are held at once
N_LAYERS = len(sc.si_layer_centers())
N_SLABS = sc.N_FACES * N_LAYERS
_STEP_KEYS = ("thx", "thy", "thz", "tpx", "tpy", "tpz", "tpath", "ttime", "tedep", "tmc",
              "tmodule", "tlayer", "evt_id_hit")
_PARTICLE_KEYS = ("pdg", "vsx", "vsy", "parent", "parent_charged", "evt_id_particle")
_COPIED = ("hit_mode", "particles_kept", "num_events", "tcellID_encoding")


def sensor_facts(run_dir=None, pitch_x_um=None, pitch_y_um=None, thick_um=None):
    """{"xsize", "ysize", "thick", "source"} in um: from the PIXELAV run directory run_dir, or
    from the explicit pitches (thick None unless thick_um is given). ValueError for a run
    directory sensor_model finds a problem in, a model with a magnetic field, or missing,
    contradictory or non-positive facts."""
    if run_dir:
        if pitch_x_um is not None or pitch_y_um is not None or thick_um is not None:
            raise ValueError("give the run directory or the pitches and thickness, not both")
        ef, _, problems = sensor_model.read_run_dir(run_dir)
        if problems:
            raise ValueError("; ".join(problems))
        if any(b != 0 for b in ef["bfield"]):
            raise ValueError(f"{run_dir}: B field {ef['bfield']} T; the window frame (v, u, w) is "
                             f"mirrored, which reverses the Lorentz drift of a sensor in a field")
        return {"xsize": ef["xsize"], "ysize": ef["ysize"], "thick": ef["thick"],
                "source": f"run directory {os.path.abspath(run_dir)}: {sensor_model.describe_efield(ef)}"}
    if pitch_x_um is None or pitch_y_um is None:
        raise ValueError("the sensor needs a PIXELAV run directory (--run-dir), or both pitches "
                         "(--pitch-x-um along v, --pitch-y-um along u)")
    for name, x in (("--pitch-x-um", pitch_x_um), ("--pitch-y-um", pitch_y_um), ("--thick-um", thick_um)):
        if x is not None and not (math.isfinite(x) and x > 0):
            raise ValueError(f"{name} must be positive, got {x!r}")
    return {"xsize": pitch_x_um, "ysize": pitch_y_um, "thick": thick_um,
            "source": f"--pitch-x-um {pitch_x_um:g} --pitch-y-um {pitch_y_um:g}"
                      + ("" if thick_um is None else f" --thick-um {thick_um:g}")}


def sensor_thickness(sensor, si_thick_um, what):
    """The sensor's thickness [um]: its own, or si_thick_um (the simulated silicon's, of `what`)
    when it has none. ValueError when the two differ by more than THICK_TOL_UM."""
    if sensor["thick"] is None:
        return si_thick_um
    if not abs(sensor["thick"] - si_thick_um) <= THICK_TOL_UM:
        raise ValueError(f"the sensor is {sensor['thick']:g} um thick ({sensor['source']}), the "
                         f"silicon of {what} {si_thick_um:g} um; they may differ by {THICK_TOL_UM:g} um")
    return sensor["thick"]


def check_piece(piece_um, sensor):
    """ValueError unless pieces of piece_um stay inside the window: half a piece past the tile
    must fit in the guard band."""
    if not (math.isfinite(piece_um) and piece_um > 0):
        raise ValueError(f"--piece-um must be positive, got {piece_um!r}")
    if piece_um / 2 > GUARD * min(sensor["xsize"], sensor["ysize"]):
        raise ValueError(f"--piece-um {piece_um:g}: half a piece reaches past the {GUARD}-pixel "
                         f"guard band of {GUARD * min(sensor['xsize'], sensor['ysize']):g} um")


def _event_range(spec, evt):
    """[first, stop) of the event ids --events A:B selects (default: every event of evt)."""
    lo, hi = (int(evt.min()), int(evt.max()) + 1) if evt.size else (0, 0)
    if spec is None:
        return lo, hi
    a, sep, b = spec.partition(":")
    try:
        if not sep:
            raise ValueError
        first = int(a) if a.strip() else lo
        stop = int(b) if b.strip() else hi
    except ValueError:
        raise SystemExit(f"--events {spec!r}: expected A:B, the event ids A..B-1") from None
    if not first < stop:
        raise SystemExit(f"--events {spec!r} selects no event")
    return first, stop


def segments(S, thick_um):
    """Per step of the step arrays S: the clipped straight segment, in um, as (u, v, z) of its
    centre, the unit direction (du, dv, dz), and s_lo <= 0 <= s_hi, its extent along the
    direction; plus layer and the energy [eV] and path length [um]."""
    phin, w, layer = sc.tracker_hit_frame(S)
    u, v, _ = sc.to_local(S["thx"], S["thy"], S["thz"], phin)
    pu, pv, pw = sc.to_local(S["tpx"], S["tpy"], S["tpz"], phin)
    p = np.sqrt(pu**2 + pv**2 + pw**2)
    moving = p > 0
    safe = np.where(moving, p, 1.0)
    du, dv, dz = (np.where(moving, c / safe, f) for c, f in ((pu, 0.0), (pv, 0.0), (pw, 1.0)))
    z = np.clip((w - sc.si_layer_centers()[layer]) * 1e3 + thick_um / 2, 0.0, thick_um)
    half = S["tpath"] * 1e3 / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        s_in, s_out = (0.0 - z) / dz, (thick_um - z) / dz
    flat = dz == 0
    s_lo = np.where(flat, -half, np.maximum(-half, np.minimum(s_in, s_out)))
    s_hi = np.where(flat, half, np.minimum(half, np.maximum(s_in, s_out)))
    s_lo, s_hi = np.minimum(s_lo, 0.0), np.maximum(s_hi, 0.0)
    return {"u": u * 1e3, "v": v * 1e3, "z": z, "du": du, "dv": dv, "dz": dz, "s_lo": s_lo,
            "s_hi": s_hi, "layer": layer, "e_eV": S["tedep"] * 1e9, "path_um": S["tpath"] * 1e3,
            "p_MeV": p * 1e3}


def groups(S, P, si_thick_mm):
    """Per step of S: its group (0..G-1), and per group the root node's particle and whether the
    root was created in its slab; the grouping of the module docstring. P are the particle
    arrays of the npz."""
    slab = S["tmodule"].astype(np.int64) * N_LAYERS + S["tlayer"]
    nodes, step_node = np.unique(S["tmc"] * N_SLABS + slab, return_inverse=True)
    part, nslab = nodes // N_SLABS, nodes % N_SLABS
    module, layer = nslab // N_LAYERS, nslab % N_LAYERS
    parent = P["parent"][part]
    want = parent * N_SLABS + nslab
    at = np.minimum(np.searchsorted(nodes, want), len(nodes) - 1)
    parent_here = (parent >= 0) & (nodes[at] == want)
    phi = sc.module_face_phi(module)
    w_vtx = P["vsx"][part] * np.cos(phi) + P["vsy"][part] * np.sin(phi)
    created = np.abs(w_vtx - sc.si_layer_centers()[layer]) <= si_thick_mm / 2 + sc.SLAB_TOL_MM
    up = np.where(parent_here & created & P["parent_charged"][part], at, np.arange(len(nodes)))
    for _ in range(64):                      # pointer jumping: depth halves every pass
        nxt = up[up]
        if np.array_equal(nxt, up):
            break
        up = nxt
    else:
        raise RuntimeError("particle chains did not converge to roots in 64 passes")
    roots, node_group = np.unique(up, return_inverse=True)
    return {"step_group": node_group[step_node], "root": part[roots], "module": module[roots],
            "layer": layer[roots], "created": created[roots],
            "n_particles": np.bincount(node_group, minlength=len(roots))}


def build_block(S, P, sensor, thick_um, si_thick_mm, piece_um, max_pairs=MAX_PAIRS):
    """Pieces, jobs and groups of the steps S (all events of the block), ids from 0. Returns
    (pieces, jobs, grp): pieces in job order with window-frame endpoints x0..z1 [um] and e [eV];
    jobs with start/stop into the pieces; grp the per-group fields."""
    xs, ys = sensor["xsize"], sensor["ysize"]
    seg = segments(S, thick_um)
    G = groups(S, P, si_thick_mm)
    step_group = G["step_group"]
    n_groups = len(G["root"])

    # pieces of at most piece_um, each with its share of the energy by length
    length = seg["s_hi"] - seg["s_lo"]
    n = np.maximum(1, np.ceil(length / piece_um - 1e-9)).astype(np.int64)
    step = np.repeat(np.arange(len(n)), n)
    k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    dl = (length / n)[step]
    s0 = seg["s_lo"][step] + k * dl
    s1 = s0 + dl
    e = (seg["e_eV"] / n)[step]
    # a piece too large for one job is shared, geometry repeated, so a split can always fit
    cap = max_pairs * EV_PER_PAIR
    copies = np.where(e > cap / 2, np.ceil(e / (cap / 2)), 1).astype(np.int64)
    if (copies > 1).any():
        step, s0, s1, e = (np.repeat(a, copies) for a in (step, s0, s1, e))
        e = e / np.repeat(copies, copies)
    end0 = {c: seg[c][step] + seg["d" + c][step] * s0 for c in ("u", "v", "z")}
    end1 = {c: seg[c][step] + seg["d" + c][step] * s1 for c in ("u", "v", "z")}
    end0["z"], end1["z"] = np.clip(end0["z"], 0.0, thick_um), np.clip(end1["z"], 0.0, thick_um)
    tv = np.floor((end0["v"] + end1["v"]) / 2 / xs).astype(np.int64) // TILE_X
    tu = np.floor((end0["u"] + end1["u"]) / 2 / ys).astype(np.int64) // TILE_Y
    pg = step_group[step]

    # jobs: (group, tile), in piece order inside; split those over the pair cap
    order = np.lexsort((np.arange(len(e)), tu, tv, pg))
    step, s0, s1, e, tv, tu, pg = (a[order] for a in (step, s0, s1, e, tv, tu, pg))
    end0 = {c: a[order] for c, a in end0.items()}
    end1 = {c: a[order] for c, a in end1.items()}
    new = np.ones(len(e), bool)
    new[1:] = (pg[1:] != pg[:-1]) | (tv[1:] != tv[:-1]) | (tu[1:] != tu[:-1])
    tile_start = np.flatnonzero(new)
    tile_e = np.add.reduceat(e, tile_start) if len(e) else np.zeros(0)
    tile_max = np.maximum.reduceat(e, tile_start) if len(e) else np.zeros(0)
    n_split = np.where(tile_e > cap, np.ceil(tile_e / (cap - tile_max)), 1).astype(np.int64)
    tile_of = np.cumsum(new) - 1
    before = np.cumsum(e) - e - np.repeat(np.cumsum(tile_e) - tile_e, np.diff(np.append(tile_start, len(e))))
    over = (n_split > 1)[tile_of]
    target = np.where(over, (tile_e / n_split)[tile_of], 1.0)
    sub = np.where(over, np.minimum(np.floor((before + e / 2) / target), n_split[tile_of] - 1), 0).astype(np.int64)
    new[1:] |= sub[1:] != sub[:-1]
    start = np.flatnonzero(new)
    stop = np.append(start[1:], len(e))
    job_e = np.add.reduceat(e, start) if len(e) else np.zeros(0)
    if (job_e > cap * (1 + 1e-12)).any():
        raise RuntimeError(f"a split job holds {job_e.max() / EV_PER_PAIR:.0f} pairs, over {max_pairs}")
    if (job_e / EV_PER_PAIR + (stop - start) >= DRIVER_PAIRS).any():
        raise RuntimeError(f"a job may make {DRIVER_PAIRS} pairs, which the driver refuses")
    jt = tile_of[start]

    # window frame of each piece's job
    cx = (tv * TILE_X - GUARD + CENTRE_X + 0.5) * xs
    cy = (tu * TILE_Y - GUARD + CENTRE_Y + 0.5) * ys
    pieces = {"x0": end0["v"] - cx, "y0": end0["u"] - cy, "z0": end0["z"],
              "x1": end1["v"] - cx, "y1": end1["u"] - cy, "z1": end1["z"], "e": e}
    root = G["root"]
    jg = pg[start]
    jobs = {"evt_id": P["evt_id_particle"][root][jg], "module": G["module"][jg],
            "layer": G["layer"][jg], "group_id": jg, "tile_v": tv[start], "tile_u": tu[start],
            "win_iv0": tv[start] * TILE_X - GUARD, "win_iu0": tu[start] * TILE_Y - GUARD,
            "n_seg": stop - start, "edep_keV": job_e / 1e3, "n_split": n_split[jt],
            "start": start, "stop": stop, "e_eV": job_e}

    # per group: totals over its steps and the extent of its clipped segments
    def reduce(op, values, by_order, starts):
        return op.reduceat(values[by_order], starts) if len(values) else np.zeros(0)
    gorder = np.lexsort((S["ttime"], S["tmc"] != root[step_group], step_group))
    gstart = np.flatnonzero(np.r_[True, step_group[gorder][1:] != step_group[gorder][:-1]])
    lo_pt = {c: seg[c] + seg["d" + c] * seg["s_lo"] for c in ("u", "v", "z")}
    hi_pt = {c: seg[c] + seg["d" + c] * seg["s_hi"] for c in ("u", "v", "z")}
    lo = {c: np.minimum(lo_pt[c], hi_pt[c]) for c in lo_pt}
    hi = {c: np.maximum(lo_pt[c], hi_pt[c]) for c in lo_pt}
    mid = {c: (lo_pt[c] + hi_pt[c]) / 2 for c in ("u", "v")}
    ge = np.bincount(step_group, seg["e_eV"], n_groups)
    gw = np.where(ge > 0, ge, 1.0)
    grp = {"evt_id": P["evt_id_particle"][root], "module": G["module"], "layer": G["layer"],
           "root": root, "root_pdg": P["pdg"][root],
           "root_p_MeV": seg["p_MeV"][gorder][gstart],
           "n_particles": G["n_particles"], "n_steps": np.bincount(step_group, minlength=n_groups),
           "edep_keV": ge / 1e3, "path_um": np.bincount(step_group, seg["path_um"], n_groups),
           "w_min_um": reduce(np.minimum, lo["z"], gorder, gstart),
           "w_max_um": reduce(np.maximum, hi["z"], gorder, gstart),
           "u_centroid_um": np.bincount(step_group, mid["u"] * seg["e_eV"], n_groups) / gw,
           "v_centroid_um": np.bincount(step_group, mid["v"] * seg["e_eV"], n_groups) / gw,
           "u_min_um": reduce(np.minimum, lo["u"], gorder, gstart),
           "u_max_um": reduce(np.maximum, hi["u"], gorder, gstart),
           "v_min_um": reduce(np.minimum, lo["v"], gorder, gstart),
           "v_max_um": reduce(np.maximum, hi["v"], gorder, gstart),
           "t_first_ns": reduce(np.minimum, S["ttime"], gorder, gstart),
           "created_in_slab": G["created"],
           "n_tiles": np.bincount(pg[tile_start], minlength=n_groups)}
    return pieces, jobs, grp


def check_block(pieces, jobs, step_e_eV, thick_um, sensor, piece_um):
    """(energy difference [eV], pieces outside their window's tile): the hard checks. The energy
    of the jobs must equal step_e_eV, the steps' (relative 1e-9); every piece end must lie within
    the tile, give or take half a piece, and within 0..thick in z."""
    diff = float(jobs["e_eV"].sum()) - step_e_eV
    tol = piece_um / 2 + 1e-6
    bx = TILE_X / 2 * sensor["xsize"] + tol
    by = TILE_Y / 2 * sensor["ysize"] + tol
    out = np.zeros(len(pieces["e"]), bool)
    for end in "01":
        out |= ~(np.abs(pieces["x" + end]) <= bx) | ~(np.abs(pieces["y" + end]) <= by)
        out |= ~((pieces["z" + end] >= 0.0) & (pieces["z" + end] <= thick_um))
    return diff, int(np.count_nonzero(out))


_FMT = "%.3f %.3f %.3f %.3f %.3f %.3f %.2f"


def job_text(pieces, jobs, job_id, a, b):
    """The text of jobs a..b-1 of a block (ids job_id), header line first for each."""
    if a == b:
        return ""
    p0, p1 = jobs["start"][a], jobs["stop"][b - 1]
    rows = np.column_stack([pieces[c][p0:p1] for c in ("x0", "y0", "z0", "x1", "y1", "z1", "e")])
    lines = np.empty(p1 - p0 + (b - a), dtype=object)
    head = jobs["start"][a:b] - p0 + np.arange(b - a)
    is_head = np.zeros(len(lines), bool)
    is_head[head] = True
    lines[head] = ["J %d %d" % (j, n) for j, n in zip(job_id[a:b].tolist(), jobs["n_seg"][a:b].tolist())]
    lines[~is_head] = [_FMT % tuple(r) for r in rows.tolist()]
    return "\n".join(lines.tolist()) + "\n"


def _load(npz, events):
    """(step arrays, particle arrays, event range, provenance) of the step npz."""
    d = sc.load_cascade(npz)
    if sc.hit_mode(d) != "step":
        raise SystemExit(f"{npz}: holds one tracker hit per sensor crossing (hit_mode "
                         f"{sc.hit_mode(d)}); deposit jobs need one per Geant4 step "
                         f"(sim/run_sim_trackermom.py with CALOMAPS_TRACKER_HITS=step)")
    missing = [k for k in _STEP_KEYS + _PARTICLE_KEYS + ("geo_si_thick_mm",) if k not in d]
    if missing:
        raise SystemExit(f"{npz}: no {', '.join(missing)}; extract it with "
                         f"analysis/extract_trackermom.py, which stores them for a step file")
    S = {k: np.asarray(d[k]) for k in _STEP_KEYS}
    for k in ("thx", "thy", "thz", "tpx", "tpy", "tpz", "tpath", "ttime", "tedep"):
        S[k] = S[k].astype(np.float64)
    S["tmc"] = S["tmc"].astype(np.int64)
    P = {k: np.asarray(d[k]) for k in _PARTICLE_KEYS}
    P["parent"] = P["parent"].astype(np.int64)
    P["parent_charged"] = P["parent_charged"].astype(bool)
    if S["tmc"].size and (S["tmc"].min() < 0 or S["tmc"].max() >= len(P["pdg"])):
        raise SystemExit(f"{npz}: tracker-hit particle links outside the particle arrays")
    if np.any(np.diff(S["evt_id_hit"]) < 0):
        raise SystemExit(f"{npz}: the step hits are not in event order")
    first, stop = _event_range(events, S["evt_id_hit"])
    keep = [k for k in d.files if k.startswith(("sim_", "geo_")) or k in _COPIED]
    return S, P, (first, stop), {k: d[k] for k in keep}


def _percentiles(x):
    return (f"median {np.median(x):.0f}, p99 {np.percentile(x, 99):.0f}, max {x.max():.0f}"
            if len(x) else "none")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz", nargs="?", help="npz of step hits from analysis/extract_trackermom.py")
    ap.add_argument("prefix", nargs="?", help="output prefix: <prefix>.jobs.txt (or .partNNN.txt), "
                    ".jobs.npz, .summary.txt")
    ap.add_argument("--run-dir", metavar="DIR", help="PIXELAV run directory (ppixel2.init, wgt_pot.init)")
    ap.add_argument("--pitch-x-um", type=float, metavar="X", help="pixel pitch along PIXELAV x = v (xsize)")
    ap.add_argument("--pitch-y-um", type=float, metavar="Y", help="pixel pitch along PIXELAV y = u (ysize)")
    ap.add_argument("--thick-um", type=float, metavar="T",
                    help="sensor thickness with the explicit pitches (default: the npz's silicon)")
    ap.add_argument("--piece-um", type=float, default=PIECE_UM, metavar="L",
                    help=f"longest piece of a segment (default {PIECE_UM:g})")
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS, metavar="N",
                    help=f"split jobs above N electron-hole pairs (default {MAX_PAIRS}; below {DRIVER_PAIRS})")
    ap.add_argument("--parts", type=int, default=1, metavar="N",
                    help="write the jobs in N files of roughly equal energy (default 1: <prefix>.jobs.txt)")
    ap.add_argument("--events", metavar="A:B", help="the event ids A..B-1 (default: every event)")
    ap.add_argument("--check-sensor", metavar="CONFIG_ROOT",
                    help="check the sensor facts against the silicon of CONFIG_ROOT's geometry and exit")
    args = ap.parse_args(argv)
    try:
        sensor = sensor_facts(args.run_dir, args.pitch_x_um, args.pitch_y_um, args.thick_um)
        if args.check_sensor:
            si_um = decal_geometry.si_thickness_mm(args.check_sensor) * 1e3
            thick_um = sensor_thickness(sensor, si_um, f"{args.check_sensor}/geometry")
            check_piece(args.piece_um, sensor)
            print(f"sensor: {thick_um:g} um thick, pitch {sensor['xsize']:g} um along x = v, "
                  f"{sensor['ysize']:g} um along y = u ({sensor['source']}); the geometry has "
                  f"{si_um:g} um silicon")
            return 0
    except (OSError, ValueError) as err:
        sys.exit(f"deposit_jobs: {err}")
    if not (args.npz and args.prefix):
        ap.error("give the step npz and the output prefix (or --check-sensor)")
    if not 1 <= args.parts <= 1000:
        ap.error(f"--parts must be 1..1000, got {args.parts}")
    if not 1 <= args.max_pairs < DRIVER_PAIRS:
        ap.error(f"--max-pairs must be 1..{DRIVER_PAIRS - 1}, got {args.max_pairs}")

    S_all, P, (first, stop), prov = _load(args.npz, args.events)
    si_thick_mm = float(prov["geo_si_thick_mm"])
    try:
        thick_um = sensor_thickness(sensor, si_thick_mm * 1e3, args.npz)
        check_piece(args.piece_um, sensor)
    except ValueError as err:
        sys.exit(f"deposit_jobs: {err}")

    evt = S_all["evt_id_hit"]
    lo, hi = np.searchsorted(evt, first), np.searchsorted(evt, stop)
    e_total = float(S_all["tedep"][lo:hi].sum()) * 1e9
    if hi == lo or not e_total > 0:
        sys.exit(f"deposit_jobs: {args.npz}: no step deposits in events {first}..{stop - 1}")
    names = ([f"{args.prefix}.jobs.txt"] if args.parts == 1 else
             [f"{args.prefix}.part{i:03d}.txt" for i in range(args.parts)])
    os.makedirs(os.path.dirname(os.path.abspath(args.prefix)), exist_ok=True)
    files = [open(n + ".partial", "w") for n in names]
    J, Gr, n_pieces, e_done, e_steps, n_out, n_groups, n_jobs_done = [], [], 0, 0.0, 0.0, 0, 0, 0
    ok = False
    try:
        for b0 in range(first, stop, CHUNK_EVENTS):
            a, b = np.searchsorted(evt, b0), np.searchsorted(evt, min(b0 + CHUNK_EVENTS, stop))
            if a == b:
                continue
            S = {k: v[a:b] for k, v in S_all.items()}
            pieces, jobs, grp = build_block(S, P, sensor, thick_um, si_thick_mm, args.piece_um,
                                            args.max_pairs)
            step_e = float(S["tedep"].sum()) * 1e9
            diff, out = check_block(pieces, jobs, step_e, thick_um, sensor, args.piece_um)
            if not abs(diff) <= 1e-9 * step_e or out:
                raise SystemExit(f"deposit_jobs: events {b0}..{min(b0 + CHUNK_EVENTS, stop) - 1}: "
                                 f"jobs hold {diff:+.6g} eV more than the steps' {step_e:.6g} eV, and "
                                 f"{out} pieces lie outside their window's tile; nothing written")
            ids = n_jobs_done + np.arange(len(jobs["e_eV"]))
            before = e_done + np.cumsum(jobs["e_eV"]) - jobs["e_eV"]
            part = np.minimum(np.floor(args.parts * (before + jobs["e_eV"] / 2) / e_total),
                              args.parts - 1).astype(np.int64)
            for i in np.unique(part):
                ja, jb = np.searchsorted(part, i), np.searchsorted(part, i, side="right")
                files[i].write(job_text(pieces, jobs, ids, ja, jb))
            jobs["part"] = part
            jobs["group_id"] = jobs["group_id"] + n_groups
            J.append({k: v for k, v in jobs.items() if k not in ("start", "stop")})
            Gr.append(grp)
            n_groups += len(grp["root"])
            n_jobs_done += len(ids)
            n_pieces += len(pieces["e"])
            n_out += out
            e_done += float(jobs["e_eV"].sum())
            e_steps += step_e
        ok = True
    finally:
        for f in files:
            f.close()
        for n in names:
            if ok:
                os.replace(n + ".partial", n)
            elif os.path.exists(n + ".partial"):
                os.remove(n + ".partial")

    jobs = {k: np.concatenate([j[k] for j in J]) for k in J[0]}
    grp = {k: np.concatenate([g[k] for g in Gr]) for k in Gr[0]}
    n_jobs, n_grp = len(jobs["e_eV"]), len(grp["root"])
    if not abs(e_done - e_total) <= 1e-9 * e_total:
        sys.exit(f"deposit_jobs: jobs hold {e_done:.9g} eV, the steps {e_total:.9g} eV")
    np.savez_compressed(
        args.prefix + ".jobs.npz",
        job_id=np.arange(n_jobs, dtype=np.int64),
        **{"job_" + k: jobs[k] for k in ("evt_id", "module", "layer", "group_id", "tile_v", "tile_u",
                                         "win_iv0", "win_iu0", "n_seg", "edep_keV", "n_split", "part")},
        group_id=np.arange(n_grp, dtype=np.int64),
        **{"group_" + k: v for k, v in grp.items()},
        pitch_x_um=np.float64(sensor["xsize"]), pitch_y_um=np.float64(sensor["ysize"]),
        thick_um=np.float64(thick_um), sensor_source=np.array(sensor["source"]),
        window_x=np.int64(WINDOW_X), window_y=np.int64(WINDOW_Y), guard=np.int64(GUARD),
        tile_x=np.int64(TILE_X), tile_y=np.int64(TILE_Y),
        centre_x=np.int64(CENTRE_X), centre_y=np.int64(CENTRE_Y),
        ev_per_pair=np.float64(EV_PER_PAIR), max_pairs=np.int64(args.max_pairs),
        piece_um=np.float64(args.piece_um), parts=np.int64(args.parts),
        events_first=np.int64(first), events_stop=np.int64(stop),
        source_npz=np.array(os.path.abspath(args.npz)),
        **prov)

    n_steps = int(grp["n_steps"].sum())
    step_um = float(prov["geo_si_step_mm"]) * 1e3 if "geo_si_step_mm" in prov else math.nan
    is_long = S_all["tpath"][lo:hi] * 1e3 > 2 * args.piece_um
    n_long, e_long = int(np.count_nonzero(is_long)), float(S_all["tedep"][lo:hi][is_long].sum()) * 1e9
    pairs = jobs["e_eV"] / EV_PER_PAIR
    split = jobs["n_split"] > 1
    n_split_tiles = int(round(float((1.0 / jobs["n_split"][split]).sum())))
    pdg, count = np.unique(grp["root_pdg"], return_counts=True)
    top = np.argsort(-count, kind="stable")[:6]
    evts = np.unique(evt[lo:hi])
    lines = [
        f"deposit jobs: {os.path.abspath(args.npz)}",
        f"  sensor: {thick_um:g} um thick, pitch {sensor['xsize']:g} um along x = v, "
        f"{sensor['ysize']:g} um along y = u ({sensor['source']})",
        f"  window {WINDOW_X} x {WINDOW_Y} pixels, tile {TILE_X} x {TILE_Y}, guard {GUARD}; pieces "
        f"<= {args.piece_um:g} um; {EV_PER_PAIR:g} eV per pair, at most {args.max_pairs} pairs per job",
        f"  events: {len(evts)} with steps, ids {first}..{stop - 1}",
        f"  steps: {n_steps}, {e_total / 1e3:.3f} keV; silicon step limit "
        + (f"{step_um:g} um" if math.isfinite(step_um) else "none of its own (the calorimeter's 5 mm)"),
        f"  steps longer than {2 * args.piece_um:g} um (twice the piece): {n_long} ({100.0 * n_long / n_steps:.2f}%), "
        f"holding {100.0 * e_long / e_total:.2f}% of the energy",
        f"  groups: {n_grp}; created in their slab {int(grp['created_in_slab'].sum())}; spanning more "
        f"than one tile {int(np.count_nonzero(grp['n_tiles'] > 1))} "
        f"({100.0 * np.count_nonzero(grp['n_tiles'] > 1) / n_grp:.1f}%)",
        "  group roots by PDG: " + ", ".join(f"{int(pdg[i])}: {int(count[i])}" for i in top),
        f"  jobs: {n_jobs}; {n_split_tiles} (group, tile) over the pair cap, split into "
        f"{int(np.count_nonzero(split))} jobs",
        f"  pieces: {n_pieces}; per job {_percentiles(jobs['n_seg'])}",
        f"  pairs per job: {_percentiles(pairs)}",
        f"  energy closure: jobs {e_done / 1e3:.6f} keV, steps {e_steps / 1e3:.6f} keV, relative "
        f"difference {(e_done - e_steps) / e_steps:+.2e}",
        f"  pieces outside their window's tile (half a piece allowed): {n_out}",
        "  jobs: " + ", ".join(names) + f" ({args.parts} part{'s' if args.parts > 1 else ''})",
        f"  tables: {args.prefix}.jobs.npz",
    ]
    text = "\n".join(lines) + "\n"
    with open(args.prefix + ".summary.txt", "w") as f:
        f.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
