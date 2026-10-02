#!/usr/bin/env python3
"""Read a PIXELAV Stage-A sensor model and check it against a track deck.

    python analysis/pixelav/sensor_model.py --summary <run dir | ppixel2.init | wgt_pot.init>
    python analysis/pixelav/sensor_model.py --check-deck <deck.pixelav.txt> <run dir>
                                            [--assume-deck-thickness-um T]

A run directory holds the two model files that pixinit() in upstream ppixelav2.c reads (the
field order below follows that function at the commit setup/setup_pixelav.sh pins):

  ppixel2.init  line 1: free-text title, copied into the driver's char header[80], so at most
                79 characters; then, read with fscanf (line breaks carry no meaning):
                  pimom filebase
                  Bx By Bz                                                  [T]
                  thick xsize ysize temp flux_e flux_h rhe rhh peaktim samptim
                  ehole new_drde npixx npixy npixz                          [um, K, ps]
                then npixx*npixy*npixz records "ix iy iz Ex Ey Ez" [V/cm], iz outermost and ix
                innermost; pixinit ignores the index columns and relies on that order.
  wgt_pot.init  thick xsize ysize npixx npixy npixz, then as many records "ix iy iz w00 .. w22"
                in the same order. pixinit stops reading it, printing "match problem", unless
                these six numbers equal ppixel2.init's.

The grid samples one quadrant of a pixel: npixx nodes span xsize/2, npixy span ysize/2, npixz
span the thickness (fieldsav() and wgtpotsav() in ppixelav2.c).

--summary prints one line per model and exits non-zero when a file is malformed or incomplete or
the two files disagree; with --require-zero-bfield it also refuses a non-zero B field.
--check-deck compares the model thickness with the "# sensor: <T> um silicon" line that
analysis/pixelav_converter.py writes into <deck>.columns.txt, and exits non-zero on a mismatch,
on a legend without that line (unless --assume-deck-thickness-um gives it), or when the two model
files disagree. It then prints, without changing the exit status, how the driver beside this
script (ppixelav2_list_trkpy_real_entry.c, which reads the badeaa3 layout) pairs the deck with
the model: x has the model's first pitch (xsize) and TXSIZE pixels and is v, steered by cot_beta;
y has the second pitch (ysize) and TYSIZE pixels and is u, steered by cot_alpha. And it counts
the deck lines whose track inside the silicon, thick*|cot_beta| in x or thick*|cot_alpha| in y,
is longer than TXSIZE-1 or TYSIZE-1 pixels: the driver puts a track's mid-plane point anywhere in
the window's central pixel, so the window may not hold such a track.

Standard library only, so it runs under any python3.
"""
import argparse
import os
import re
import struct
import sys

EFIELD_FILE, WEIGHTING_FILE = "ppixel2.init", "wgt_pot.init"
TITLE_MAX = 79
_EF_PARAMS = ("thick", "xsize", "ysize", "temp", "flux_e", "flux_h", "rhe", "rhh",
              "peaktim", "samptim")
_SENSOR_LINE = re.compile(r"^#\s*sensor:\s*([0-9.eE+-]+)\s*um silicon")
_LAYOUT_LINE = re.compile(r"^#\s*PIXELAV track list, layout (\S+)")
# The output window of the driver is pixel[TXSIZE][TYSIZE]; the sizes are read from its source.
DRIVER_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "ppixelav2_list_trkpy_real_entry.c")
_WINDOW_DEFINE = re.compile(r"^#define\s+(TXSIZE|TYSIZE)\s+(\d+)\s*$", re.M)


def _f32(x):
    """x rounded to float32, as fscanf("%f") stores it in the driver."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _number(tok, what, kind=float):
    try:
        return kind(tok)
    except ValueError:
        raise ValueError(f"{what}: expected {'an integer' if kind is int else 'a number'}, "
                         f"found {tok!r}") from None


def _check_records(toks, grid, width, path):
    """Problems with the grid records: count, and node indices in pixinit's read order."""
    nx, ny, nz = grid
    n = nx * ny * nz
    if len(toks) != n * width:
        return [f"{path}: {len(toks) / width:g} grid records for a {nx} x {ny} x {nz} grid "
                f"({n} expected)"]
    k = 0
    for iz in range(1, nz + 1):
        for iy in range(1, ny + 1):
            for ix in range(1, nx + 1):
                if toks[k] != str(ix) or toks[k + 1] != str(iy) or toks[k + 2] != str(iz):
                    return [f"{path}: record {k // width + 1} is node "
                            f"({' '.join(toks[k:k + 3])}), expected ({ix} {iy} {iz}); pixinit "
                            f"reads records in iz, iy, ix order and ignores these indices"]
                k += width
    return []


def read_efield(path):
    """ppixel2.init header as a dict, with a 'problems' list (empty when the file is usable)."""
    with open(path, encoding="ascii", errors="replace") as f:
        text = f.read()
    title, sep, rest = text.partition("\n")
    if not sep:
        raise ValueError(f"{path}: no newline after the title line")
    toks = rest.split()
    if len(toks) < 20:
        raise ValueError(f"{path}: {len(toks)} numbers after the title, the header needs 20")
    m = {"path": path, "title": title,
         "pimom": _number(toks[0], f"{path} pimom"),
         "filebase": _number(toks[1], f"{path} filebase", int),
         "bfield": tuple(_number(t, f"{path} B field") for t in toks[2:5])}
    for name, tok in zip(_EF_PARAMS, toks[5:15]):
        m[name] = _number(tok, f"{path} {name}")
    m["ehole"] = _number(toks[15], f"{path} ehole", int)
    m["new_drde"] = _number(toks[16], f"{path} new_drde", int)
    m["grid"] = tuple(_number(t, f"{path} grid size", int) for t in toks[17:20])
    problems = []
    if len(title) > TITLE_MAX:
        problems.append(f"{path}: title is {len(title)} characters; the driver's header[80] "
                        f"holds {TITLE_MAX}")
    if title.endswith("\r"):
        problems.append(f"{path}: CRLF line ending; the title would carry a carriage return")
    if min(m["thick"], m["xsize"], m["ysize"]) <= 0:
        problems.append(f"{path}: non-positive thickness or pixel size")
    if min(m["grid"]) < 2:
        problems.append(f"{path}: grid {m['grid']} needs at least 2 nodes per axis")
    else:
        problems += _check_records(toks[20:], m["grid"], 6, path)
    m["problems"] = problems
    return m


def read_weighting(path):
    """wgt_pot.init header as a dict, with a 'problems' list (empty when the file is usable)."""
    with open(path, encoding="ascii", errors="replace") as f:
        toks = f.read().split()
    if len(toks) < 6:
        raise ValueError(f"{path}: {len(toks)} tokens; the header is "
                         f"'thick xsize ysize npixx npixy npixz' and pixinit reads the grid "
                         f"from this file itself")
    try:
        w = {"path": path,
             "thick": _number(toks[0], f"{path} thick"),
             "xsize": _number(toks[1], f"{path} xsize"),
             "ysize": _number(toks[2], f"{path} ysize"),
             "grid": tuple(_number(t, f"{path} grid size", int) for t in toks[3:6])}
    except ValueError as e:
        raise ValueError(f"{e} (the header is 'thick xsize ysize npixx npixy npixz'; pixinit "
                         f"reads the grid from this file itself, not from a file it names)") from None
    w["problems"] = ([f"{path}: grid {w['grid']} needs at least 2 nodes per axis"]
                     if min(w["grid"]) < 2 else _check_records(toks[6:], w["grid"], 12, path))
    return w


def pair_mismatch(ef, wp):
    """Differences pixinit's comparison would find, as text; empty when the files agree."""
    out = [f"{k} {ef[k]:g} vs {wp[k]:g}" for k in ("thick", "xsize", "ysize")
           if _f32(ef[k]) != _f32(wp[k])]
    if ef["grid"] != wp["grid"]:
        out.append("grid %s vs %s" % ("x".join(map(str, ef["grid"])), "x".join(map(str, wp["grid"]))))
    return out


def is_weighting_file(path):
    """True for a wgt_pot* file name, False for ppixel2*; otherwise True when the first line
    reads as a wgt_pot.init header (three numbers, three integers)."""
    name = os.path.basename(path)
    if name.startswith(("wgt_pot", "ppixel2")):
        return name.startswith("wgt_pot")
    with open(path, encoding="ascii", errors="replace") as f:
        first = f.readline().split()
    if len(first) != 6:
        return False
    try:
        for t in first[:3]:
            float(t)
        for t in first[3:]:
            int(t)
    except ValueError:
        return False
    return True


def describe_efield(ef):
    bx, by, bz = ef["bfield"]
    return (f'{os.path.basename(ef["path"])}: "{ef["title"]}" | {ef["thick"]:g} um thick | '
            f'pixels {ef["xsize"]:g} x {ef["ysize"]:g} um | {ef["temp"]:g} K | '
            f'B ({bx:g}, {by:g}, {bz:g}) T | grid {"x".join(map(str, ef["grid"]))}')


def describe_weighting(wp):
    return (f'{os.path.basename(wp["path"])}: {wp["thick"]:g} um thick | pixels {wp["xsize"]:g} x '
            f'{wp["ysize"]:g} um | grid {"x".join(map(str, wp["grid"]))}')


def read_run_dir(run_dir):
    """(ppixel2.init dict, wgt_pot.init dict, problems) for a PIXELAV run directory."""
    ef = read_efield(os.path.join(run_dir, EFIELD_FILE))
    wp = read_weighting(os.path.join(run_dir, WEIGHTING_FILE))
    problems = ef["problems"] + wp["problems"]
    mis = pair_mismatch(ef, wp)
    if mis:
        problems.append(f"{EFIELD_FILE} and {WEIGHTING_FILE} disagree ({'; '.join(mis)}): pixinit "
                        f"prints 'match problem' and leaves the weighting potential unread")
    return ef, wp, problems


def deck_thickness_um(deck):
    """Silicon thickness from the deck legend's sensor line, or None when there is none."""
    legend = deck + ".columns.txt"
    if not os.path.exists(legend):
        return None
    with open(legend) as f:
        for line in f:
            m = _SENSOR_LINE.match(line)
            if m:
                return float(m.group(1))
    return None


def driver_window(path=None):
    """(TXSIZE, TYSIZE): the pixels of the driver's output window along x and y, from the
    #defines of its source file (default DRIVER_SOURCE)."""
    path = path or DRIVER_SOURCE
    with open(path) as f:
        sizes = {name: int(n) for name, n in _WINDOW_DEFINE.findall(f.read())}
    if set(sizes) != {"TXSIZE", "TYSIZE"}:
        raise ValueError(f"{path}: no '#define TXSIZE <n>' and '#define TYSIZE <n>' lines")
    return sizes["TXSIZE"], sizes["TYSIZE"]


def deck_layout(deck):
    """The deck layout its legend names (badeaa3, smartpix), or None without a legend line."""
    legend = deck + ".columns.txt"
    if not os.path.exists(legend):
        return None
    with open(legend) as f:
        for line in f:
            m = _LAYOUT_LINE.match(line)
            if m:
                return m.group(1)
    return None


def _window(deck, ef):
    """Print the driver's pairing of the deck with the model ef, and count the deck lines whose
    track is too long for the output window. Informational; returns nothing."""
    layout = deck_layout(deck)
    if layout not in (None, "badeaa3"):
        print(f"axes:  not checked; layout {layout} is not the badeaa3 layout "
              f"{os.path.basename(DRIVER_SOURCE)} reads")
        return
    try:
        nx, ny = driver_window()
    except (OSError, ValueError) as e:
        print(f"axes:  not checked ({e})")
        return
    t, px, py = ef["thick"], ef["xsize"], ef["ysize"]
    print(f"axes:  x: {px:g} um pitch, {nx} pixels = {nx * px:g} um, along v (cot_beta, modx); "
          f"y: {py:g} um pitch, {ny} pixels = {ny * py:g} um, along u (cot_alpha, mody) "
          f"({os.path.basename(DRIVER_SOURCE)})")
    n = n_x = n_y = n_any = 0
    with open(deck) as f:
        for k, line in enumerate(f, 1):
            tok = line.split()
            if not tok:
                continue
            try:
                cot_a, cot_b = float(tok[0]), float(tok[1])
            except (IndexError, ValueError):
                print(f"window: not checked; deck line {k} does not start with cot_alpha cot_beta")
                return
            over_x = t * abs(cot_b) > (nx - 1) * px
            over_y = t * abs(cot_a) > (ny - 1) * py
            n += 1
            n_x += over_x
            n_y += over_y
            n_any += over_x or over_y
    print(f"window: {n_any} of {n} deck lines are longer inside the {t:g} um silicon than {nx - 1} "
          f"pixels in x ({(nx - 1) * px:g} um; {n_x} lines) or {ny - 1} in y ({(ny - 1) * py:g} um; "
          f"{n_y} lines); the window may not hold such a track")


def _summary(path, require_zero_b):
    if os.path.isdir(path):
        ef, wp, problems = read_run_dir(path)
        print(f"{path}: {describe_efield(ef)} | "
              f"{WEIGHTING_FILE} {'disagrees' if pair_mismatch(ef, wp) else 'agrees'}")
    elif is_weighting_file(path):
        ef, wp = None, read_weighting(path)
        problems = wp["problems"]
        print(describe_weighting(wp))
    else:
        ef = read_efield(path)
        problems = ef["problems"]
        print(describe_efield(ef))
    if require_zero_b and ef is not None and any(b != 0 for b in ef["bfield"]):
        problems.append(f"B field {ef['bfield']} T is not zero; the CALOMAPS simulation has no "
                        f"magnetic field")
    return problems


def _check_deck(deck, run_dir, assumed_um):
    if not os.path.isfile(deck):
        return [f"no deck {deck}"]
    ef, wp, problems = read_run_dir(run_dir)
    print(f"model: {describe_efield(ef)}")
    legend_um = deck_thickness_um(deck)
    if legend_um is None and assumed_um is None:
        return problems + [f"{deck}.columns.txt is missing or has no '# sensor: <T> um silicon' "
                           f"line; give the deck's thickness with --assume-deck-thickness-um"]
    if legend_um is not None and assumed_um is not None and abs(legend_um - assumed_um) > 1e-3:
        return problems + [f"--assume-deck-thickness-um {assumed_um:g} contradicts the legend's "
                           f"{legend_um:g} um"]
    deck_um = legend_um if legend_um is not None else assumed_um
    print(f"deck:  {deck}: {deck_um:g} um silicon "
          f"({'legend' if legend_um is not None else '--assume-deck-thickness-um'})")
    if abs(deck_um - ef["thick"]) > 1e-3:
        problems.append(f"deck is {deck_um:g} um silicon, the sensor model is {ef['thick']:g} um thick")
    _window(deck, ef)
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--summary", metavar="PATH",
                      help="run directory, or a single ppixel2.init / wgt_pot.init file")
    mode.add_argument("--check-deck", nargs=2, metavar=("DECK", "RUN_DIR"),
                      help="deck .pixelav.txt (its .columns.txt legend beside it) and run directory")
    ap.add_argument("--assume-deck-thickness-um", type=float, metavar="T",
                    help="deck thickness when its legend has no sensor line")
    ap.add_argument("--require-zero-bfield", action="store_true",
                    help="with --summary: refuse a model whose B field is not zero")
    args = ap.parse_args(argv)
    try:
        if args.summary:
            problems = _summary(args.summary, args.require_zero_bfield)
        else:
            problems = _check_deck(*args.check_deck, args.assume_deck_thickness_um)
    except (OSError, ValueError) as e:
        problems = [str(e)]
    sys.stdout.flush()
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if args.check_deck and not problems:
        print("OK: deck and sensor model thicknesses agree; "
              f"{EFIELD_FILE} and {WEIGHTING_FILE} agree")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
