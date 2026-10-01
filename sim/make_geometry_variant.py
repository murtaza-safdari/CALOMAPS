#!/usr/bin/env python3
"""Write a copy of the repo's geometry/ with a different ECal sensor (silicon) thickness.

    python sim/make_geometry_variant.py --si-thickness-um 100 --out $CALOMAPS_DATA_BASE/geometry_si100um

OUT/geometry/ is byte-for-byte the repo's geometry/ except for the value of the
ECal_Si_thickness constant (geometry/SiD_TestBeam.xml). The air pads either side of the
silicon (ECal_Si_air_in/out) are defined from that constant, so every layer keeps its depth
and its Si mid-plane; only the sensor thickness changes. The variant compact path,
OUT/geometry/SiD_TestBeam.xml, is the only thing printed on stdout.

OUT is a config root: pass it as --home to analysis/extract_readouts.py,
extract_trackermom.py and extract_cascade.py. sim/geometry_variant.sh calls this script when
CALOMAPS_SI_THICKNESS_UM is set, so the sim wrappers rarely need it directly.

Refused: a thickness <= 0, one that leaves either air pad without positive thickness, a source
tree without exactly one ECal_Si_thickness constant, an OUT/geometry holding files the repo's
geometry/ does not have, and a variant whose Si mid-planes or layer depths differ from the
repo's. Re-running with the same arguments rewrites nothing; a changed file is replaced
atomically, so a ddsim already reading the variant never sees a partial file.
"""
import argparse, math, os, re, shutil, sys, tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import decal_geometry as dg   # noqa: E402

CONSTANT = "ECal_Si_thickness"
# Thinner than this is round-off of a zero pad, which Geant4 cannot build as a solid.
MIN_PAD_MM = 1e-6
_ELEMENT = re.compile(r'<constant\b[^>]*?\bname\s*=\s*"' + CONSTANT + r'"[^>]*>')
_VALUE = re.compile(r'(\bvalue\s*=\s*")([^"]*)(")')
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _fail(msg):
    sys.exit(f"make_geometry_variant: {msg}")


def _source_files(src):
    """Relative paths of the files under src, skipping hidden entries (.DS_Store, ._*, .nfs*)."""
    out = []
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        out += [os.path.relpath(os.path.join(root, f), src) for f in sorted(files) if not f.startswith(".")]
    return out


def _rewrite(text, value):
    """text with the one live ECal_Si_thickness constant's value replaced, and the match count."""
    comments = [m.span() for m in _COMMENT.finditer(text)]
    live = [m for m in _ELEMENT.finditer(text)
            if not any(a <= m.start() < b for a, b in comments)]
    if len(live) != 1:
        return text, len(live)
    m = live[0]
    element, n = _VALUE.subn(lambda v: v.group(1) + value + v.group(3), m.group(0), count=1)
    if n != 1:
        return text, 0
    return text[:m.start()] + element + text[m.end():], 1


def _validate(home, t_um):
    """Fail unless the tree at `home` has the requested Si thickness, positive air pads, and the
    repo's Si mid-planes and layer depths."""
    consts = dg.compact_constants(home)
    for pad in ("ECal_Si_air_in", "ECal_Si_air_out"):
        if pad not in consts:
            _fail(f"SiD_TestBeam.xml has no {pad} constant to absorb the thickness change")
        mm = dg._pv(consts[pad], consts)
        if mm < MIN_PAD_MM:
            _fail(f"{t_um:g} um silicon leaves {pad} = {mm:.6g} mm; a slice needs a positive "
                  f"thickness (>= {MIN_PAD_MM:g} mm) for Geant4 to build it")
    t_mm = dg.si_thickness_mm(home)
    if abs(t_mm - t_um / 1000.0) > 1e-12:
        _fail(f"variant Si thickness is {t_mm} mm, expected {t_um / 1000.0} mm -- the layer "
              f"slices do not use {CONSTANT}")
    r_src, r_var = dg.layer_radii(REPO), dg.layer_radii(home)
    d_src = [sum(t for _, t in sl) for sl in dg.layer_stack(REPO)]
    d_var = [sum(t for _, t in sl) for sl in dg.layer_stack(home)]
    if r_src.shape != r_var.shape or abs(r_src - r_var).max() > 1e-9 or \
            max(abs(a - b) for a, b in zip(d_src, d_var)) > 1e-9:
        _fail("the variant moves the Si mid-planes or the layer depths -- the layer slices "
              "do not use ECal_Si_air_in/ECal_Si_air_out around the silicon")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--si-thickness-um", type=float, required=True, metavar="T",
                    help="ECal sensor silicon thickness in microns (baseline 320)")
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="config root to write; the variant goes to DIR/geometry/")
    args = ap.parse_args()

    t_um = args.si_thickness_um
    if not (math.isfinite(t_um) and t_um > 0):
        _fail(f"--si-thickness-um must be a positive thickness, got {t_um!r}")
    src = os.path.join(REPO, "geometry")
    out = os.path.abspath(args.out)
    dst = os.path.join(out, "geometry")
    if os.path.realpath(dst) == os.path.realpath(src) or \
            os.path.realpath(dst).startswith(os.path.realpath(src) + os.sep):
        _fail(f"--out {args.out!r} would write into the repo's own geometry/")

    files = _source_files(src)
    value = "%.12g*mm" % (t_um / 1000.0)
    contents, n_found = {}, 0
    for rel in files:
        with open(os.path.join(src, rel), "rb") as f:
            data = f.read()
        if rel.endswith(".xml"):
            text, n = _rewrite(data.decode("utf-8"), value)
            if n > 1:
                _fail(f"{rel} defines {CONSTANT} {n} times")
            if n == 1:
                n_found += 1
                data = text.encode("utf-8")
        contents[rel] = data
    if n_found != 1:
        _fail(f"found {n_found} {CONSTANT} constants under {src} (expected exactly 1)")

    if os.path.isdir(dst):
        extra = sorted(set(_source_files(dst)) - set(files))
        if extra:
            _fail(f"{dst} holds files the repo's geometry/ does not have: {', '.join(extra)}; "
                  f"remove them or choose another --out")

    with tempfile.TemporaryDirectory(prefix="geometry-variant-") as tmp:
        for rel, data in contents.items():
            p = os.path.join(tmp, "geometry", rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
        _validate(tmp, t_um)

    for rel, data in contents.items():
        target = os.path.join(dst, rel)
        if os.path.isfile(target):
            with open(target, "rb") as f:
                if f.read() == data:
                    continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd, part = tempfile.mkstemp(prefix=".part-", dir=os.path.dirname(target))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            shutil.copymode(os.path.join(src, rel), part)
            os.replace(part, target)
        except BaseException:
            os.unlink(part)
            raise

    print(os.path.join(dst, "SiD_TestBeam.xml"))


if __name__ == "__main__":
    main()
