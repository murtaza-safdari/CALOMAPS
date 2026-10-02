#!/usr/bin/env python3
"""Write a copy of the repo's geometry/ with a different ECal sensor (silicon) thickness.

    python sim/make_geometry_variant.py --si-thickness-um 100
    python sim/make_geometry_variant.py --si-thickness-um 100 --out DIR

The variant goes to OUT/geometry/, and OUT/geometry/SiD_TestBeam.xml, its compact, is the only
thing printed on stdout. Without --out, OUT is $CALOMAPS_DATA_BASE/geometry_si<T>um_<hash>
(decal_geometry.variant_home), <hash> being the first 10 hex digits of a content hash of the
repo's geometry/: checkouts with the same geometry share one variant, and a checkout whose
geometry differs gets its own. Without --out, a thickness equal to the repo's own prints the
repo's compact and writes nothing.

OUT/geometry/ is byte-for-byte the repo's geometry/ except in SiD_TestBeam.xml: the values of
three constants, ECal_Si_thickness and the air pads either side of the silicon,
ECal_Si_air_in/out, and a comment after the XML declaration that names the thickness and the
sha256 of the source geometry/. The thickness change is split equally between the two pads, so
every layer keeps its depth and its Si mid-plane; only the sensor thickness changes. The
constants stay plain values (a number times a unit), which the notebooks' own compact parsers
can also evaluate.

OUT is a config root: pass it as --home to analysis/extract_readouts.py,
extract_trackermom.py and extract_cascade.py. sim/geometry_variant.sh calls this script when
CALOMAPS_SI_THICKNESS_UM is set, so the sim wrappers rarely need it directly.

An existing OUT/geometry is never rewritten, since another run may be simulating it: if it
already holds exactly this variant nothing is written, otherwise the script stops. A new variant
is written to a temporary directory beside OUT, validated, and renamed into place, so no reader
sees a partial tree and a refused variant leaves nothing behind.

Refused: a thickness <= 0, one that leaves either air pad without positive thickness, a source
tree without exactly one definition of each of the three constants (in SiD_TestBeam.xml), and a
variant whose Si mid-planes or layer depths differ from the repo's.
"""
import argparse, math, os, re, shutil, sys, tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import decal_geometry as dg   # noqa: E402

SI, AIR_IN, AIR_OUT = "ECal_Si_thickness", "ECal_Si_air_in", "ECal_Si_air_out"
# Thinner than this is round-off of a zero pad, which Geant4 cannot build as a solid.
MIN_PAD_MM = 1e-6
_VALUE = re.compile(r'(\bvalue\s*=\s*")([^"]*)(")')
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_DECLARATION = re.compile(r"\A﻿?<\?xml[^>]*\?>[ \t]*(\r?\n)?")
_MARKER = re.compile(r"<!-- ECal silicon-thickness variant written by sim/make_geometry_variant\.py: "
                     r"ECal_Si_thickness (\S+) um, source geometry/ sha256 ([0-9a-f]+) -->")


def _fail(msg):
    sys.exit(f"make_geometry_variant: {msg}")


def _definitions(text, name):
    """The <constant name="name" ...> elements of text that are not inside an XML comment."""
    comments = [m.span() for m in _COMMENT.finditer(text)]
    element = re.compile(r'<constant\b[^>]*?\bname\s*=\s*"' + re.escape(name) + r'"[^>]*>')
    return [m for m in element.finditer(text) if not any(a <= m.start() < b for a, b in comments)]


def _set_value(text, name, value):
    """text with the value of its one definition of `name` replaced."""
    m, = _definitions(text, name)
    element, n = _VALUE.subn(lambda v: v.group(1) + value + v.group(3), m.group(0), count=1)
    if n != 1:
        _fail(f"the {name} constant has no value attribute")
    return text[:m.start()] + element + text[m.end():]


def _mark(text, t_um, source_sha):
    """text with the variant marker comment after its XML declaration (or first)."""
    note = (f"<!-- ECal silicon-thickness variant written by sim/make_geometry_variant.py: "
            f"ECal_Si_thickness {t_um:g} um, source geometry/ sha256 {source_sha} -->\n")
    m = _DECLARATION.match(text)
    if not m:
        return note + text
    return text[:m.end()] + ("" if m.group(1) else "\n") + note + text[m.end():]


def _render(t_um):
    """{relative path: bytes} of the t_um variant of the repo's geometry/."""
    src = os.path.join(REPO, "geometry")
    contents = {}
    for rel in dg.geometry_files(REPO):
        with open(os.path.join(src, rel), "rb") as f:
            contents[rel] = f.read()
    xml = {rel: data.decode("utf-8") for rel, data in contents.items() if rel.endswith(".xml")}
    for name in (SI, AIR_IN, AIR_OUT):
        where = [rel for rel, text in xml.items() for _ in _definitions(text, name)]
        if where != ["SiD_TestBeam.xml"]:
            _fail(f"found {len(where)} definitions of {name} under {src} "
                  f"({', '.join(where) or 'none'}); expected exactly one, in SiD_TestBeam.xml")

    consts = dg.compact_constants(REPO)
    si_src, in_src, out_src = (dg._pv(consts[n], consts) for n in (SI, AIR_IN, AIR_OUT))
    si = t_um / 1000.0
    text = xml["SiD_TestBeam.xml"]
    pad_shift = (si_src - si) / 2
    for name, mm in ((SI, si), (AIR_IN, in_src + pad_shift), (AIR_OUT, out_src + pad_shift)):
        text = _set_value(text, name, "%.12g*mm" % mm)
    contents["SiD_TestBeam.xml"] = _mark(text, t_um, dg.geometry_hash(REPO)).encode("utf-8")
    return contents


def _validate(home, t_um):
    """Fail unless the tree at `home` has the requested Si thickness, positive air pads, and the
    repo's Si mid-planes and layer depths."""
    consts = dg.compact_constants(home)
    for pad in (AIR_IN, AIR_OUT):
        mm = dg._pv(consts[pad], consts)
        if not mm >= MIN_PAD_MM:
            _fail(f"{t_um:g} um silicon leaves {pad} = {mm:.6g} mm; a slice needs a positive "
                  f"thickness (>= {MIN_PAD_MM:g} mm) for Geant4 to build it")
    t_mm = dg.si_thickness_mm(home)
    if not abs(t_mm - t_um / 1000.0) <= 1e-12:
        _fail(f"variant Si thickness is {t_mm} mm, expected {t_um / 1000.0} mm -- the layer "
              f"slices do not use {SI}")
    r_src, r_var = dg.layer_radii(REPO), dg.layer_radii(home)
    d_src = [sum(t for _, t in sl) for sl in dg.layer_stack(REPO)]
    d_var = [sum(t for _, t in sl) for sl in dg.layer_stack(home)]
    if r_src.shape != r_var.shape or len(d_src) != len(d_var) or \
            not abs(r_src - r_var).max() <= 1e-9 or \
            not max(abs(a - b) for a, b in zip(d_src, d_var)) <= 1e-9:
        _fail("the variant moves the Si mid-planes or the layer depths -- the layer slices "
              "do not use ECal_Si_air_in/ECal_Si_air_out around the silicon")


def _require(dst, contents):
    """Return if dst holds exactly `contents`; otherwise stop without touching it."""
    have = set(dg.geometry_files(os.path.dirname(dst)))
    differ = sorted(have ^ set(contents))
    for rel in sorted(have & set(contents)):
        with open(os.path.join(dst, rel), "rb") as f:
            if f.read() != contents[rel]:
                differ.append(rel)
    if not differ:
        return
    try:
        with open(os.path.join(dst, "SiD_TestBeam.xml"), encoding="utf-8") as f:
            m = _MARKER.search(f.read())
    except (OSError, UnicodeDecodeError):
        m = None
    what = (f"the {m.group(1)} um variant of a geometry/ with sha256 {m.group(2)[:10]}..." if m
            else "no variant marker")
    _fail(f"{dst} already exists and is not this variant (it has {what}; differing: "
          f"{', '.join(differ[:5])}{' ...' if len(differ) > 5 else ''}). It is left untouched, since "
          f"a run may be using it: remove it or choose another --out.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--si-thickness-um", type=float, required=True, metavar="T",
                    help="ECal sensor silicon thickness in microns (repo geometry: 320)")
    ap.add_argument("--out", metavar="DIR",
                    help="config root to write; the variant goes to DIR/geometry/ "
                         "(default: $CALOMAPS_DATA_BASE/geometry_si<T>um_<hash>)")
    args = ap.parse_args()

    t_um = args.si_thickness_um
    if not (math.isfinite(t_um) and t_um > 0):
        _fail(f"--si-thickness-um must be a positive thickness, got {t_um!r}")
    src = os.path.join(REPO, "geometry")
    if args.out is None:
        if "%g" % t_um == "%g" % (dg.si_thickness_mm(REPO) * 1e3):
            print(os.path.join(src, "SiD_TestBeam.xml"))
            return
        out = dg.variant_home(t_um, REPO)
    else:
        out = os.path.abspath(args.out)
    dst = os.path.join(out, "geometry")
    if os.path.realpath(dst) == os.path.realpath(src) or \
            os.path.realpath(dst).startswith(os.path.realpath(src) + os.sep):
        _fail(f"--out {args.out!r} would write into the repo's own geometry/")

    contents = _render(t_um)
    if os.path.lexists(dst):
        _require(dst, contents)
        print(os.path.join(dst, "SiD_TestBeam.xml"))
        return

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".geometry-variant-", dir=os.path.dirname(out)) as tmp:
        for rel, data in contents.items():
            p = os.path.join(tmp, "geometry", rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
            shutil.copymode(os.path.join(src, rel), p)
        _validate(tmp, t_um)
        os.makedirs(out, exist_ok=True)
        try:
            os.rename(os.path.join(tmp, "geometry"), dst)
        except OSError:
            if not os.path.isdir(dst):
                raise
            _require(dst, contents)   # another run wrote it first
    print(os.path.join(dst, "SiD_TestBeam.xml"))


if __name__ == "__main__":
    main()
