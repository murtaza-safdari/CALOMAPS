#!/usr/bin/env python3
"""Write a copy of the repo's geometry/ with a different ECal sensor (silicon) thickness, or with a
Geant4 production cut or maximum step length of its own in the sensor silicon, or any of these.

    python sim/make_geometry_variant.py --si-thickness-um 100
    python sim/make_geometry_variant.py --si-thickness-um 100 --out DIR
    python sim/make_geometry_variant.py --si-thickness-um 100 --si-cut-um 1 --si-step-um 5

The variant goes to OUT/geometry/, and OUT/geometry/SiD_TestBeam.xml, its compact, is the only
thing printed on stdout. Without --out, OUT is $CALOMAPS_DATA_BASE/geometry_si<T>um_<hash>, or
geometry_si<T>um[_cut<C>um][_stepmax<S>um]_<hash> with --si-cut-um and --si-step-um
(decal_geometry.variant_home), <hash> being the first 10 hex digits of a content hash of the
repo's geometry/: checkouts with the same geometry share one variant, and a checkout whose
geometry differs gets its own. --si-thickness-um defaults to the repo's own thickness. Without
--out, --si-cut-um and --si-step-um, a thickness equal to the repo's own prints the repo's compact
and writes nothing.

OUT/geometry/ is byte-for-byte the repo's geometry/ except in SiD_TestBeam.xml: the values of
three constants, ECal_Si_thickness and the air pads either side of the silicon,
ECal_Si_air_in/out, and a comment after the XML declaration that names the thickness (and the
cut and the step limit) and the sha256 of the source geometry/. The thickness change is split equally between the
two pads, so every layer keeps its depth and its Si mid-plane; only the sensor thickness changes.
The constants stay plain values (a number times a unit), which the notebooks' own compact parsers
can also evaluate.

--si-cut-um C also puts the sensitive Silicon slices of my_custom_ecal.xml into a DD4hep region,
SiSensorRegion (a region="SiSensorRegion" attribute on each), and defines that region, with a
production cut of C um for every particle type, in a <regions> element just before <limits> in
SiD_TestBeam.xml. Geant4 then creates as tracks the secondaries (delta rays above all) whose range
in silicon exceeds C, instead of depositing their energy along the parent's step; everywhere else
the run's own cut (ddsim's physics.rangecut) applies. The source geometry must have no <regions>
element and no region on its Silicon slices.

--si-step-um S gives the sensitive Silicon slices a limitset of their own,
decal_geometry.SI_STEP_LIMITSET, in place of the calorimeter's (limits="si_step_limits" instead
of limits="cal_limits"), defined first under <limits> in SiD_TestBeam.xml with a step_length_max
of S um for every particle. No Geant4 step in the silicon is then longer than S, so a step is
close to straight: a charged particle's steps, each rebuilt from its midpoint, mean momentum and
length, chain end to start (analysis/pixelav/deposit_jobs.py relies on that). The calorimeter's
own limit (5 mm) still applies everywhere else. The source geometry must not define that
limitset already.

OUT is a config root: pass it as --home to analysis/extract_readouts.py,
extract_trackermom.py and extract_cascade.py. sim/geometry_variant.sh calls this script when
CALOMAPS_SI_THICKNESS_UM, CALOMAPS_SI_CUT_UM or CALOMAPS_SI_STEP_UM is set, so the sim wrappers rarely need it directly.

An existing OUT/geometry is never rewritten, since another run may be simulating it: if it
already holds exactly this variant nothing is written, otherwise the script stops. A new variant
is written to a temporary directory beside OUT, validated, and renamed into place, so no reader
sees a partial tree and a refused variant leaves nothing behind.

Refused: a thickness <= 0, one that leaves either air pad without positive thickness, a cut or
step limit <= 0, a source tree without exactly one definition of each of the three constants (in
SiD_TestBeam.xml), and a variant whose Si mid-planes or layer depths differ from the repo's, or
whose silicon does not read back with the requested cut and step limit.
"""
import argparse, math, os, re, shutil, sys, tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import decal_geometry as dg   # noqa: E402

SI, AIR_IN, AIR_OUT = "ECal_Si_thickness", "ECal_Si_air_in", "ECal_Si_air_out"
SI_REGION = "SiSensorRegion"
# Thinner than this is round-off of a zero pad, which Geant4 cannot build as a solid.
MIN_PAD_MM = 1e-6
_VALUE = re.compile(r'(\bvalue\s*=\s*")([^"]*)(")')
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_DECLARATION = re.compile(r"\A﻿?<\?xml[^>]*\?>[ \t]*(\r?\n)?")
_MARKER = re.compile(r"<!-- ECal silicon-thickness variant written by sim/make_geometry_variant\.py: "
                     r"ECal_Si_thickness (\S+) um(?:, Si production cut (\S+) um)?"
                     r"(?:, Si step limit (\S+) um)?, "
                     r"source geometry/ sha256 ([0-9a-f]+) -->")
_SLICE = re.compile(r"<slice\b[^>]*>")
_LIMITS = re.compile(r"^([ \t]*)<limits\b[^>]*>", re.M)
_REGIONS = re.compile(r"<regions\b")


def _fail(msg):
    sys.exit(f"make_geometry_variant: {msg}")


def _uncommented(text, pattern):
    """The matches of the compiled pattern in text that do not start inside an XML comment."""
    comments = [m.span() for m in _COMMENT.finditer(text)]
    return [m for m in pattern.finditer(text) if not any(a <= m.start() < b for a, b in comments)]


def _definitions(text, name):
    """The <constant name="name" ...> elements of text that are not inside an XML comment."""
    return _uncommented(text, re.compile(r'<constant\b[^>]*?\bname\s*=\s*"' + re.escape(name) + r'"[^>]*>'))


def _set_value(text, name, value):
    """text with the value of its one definition of `name` replaced."""
    m, = _definitions(text, name)
    element, n = _VALUE.subn(lambda v: v.group(1) + value + v.group(3), m.group(0), count=1)
    if n != 1:
        _fail(f"the {name} constant has no value attribute")
    return text[:m.start()] + element + text[m.end():]


def _mark(text, t_um, source_sha, cut_um=None, step_um=None):
    """text with the variant marker comment after its XML declaration (or first)."""
    cut = "" if cut_um is None else f", Si production cut {cut_um:g} um"
    step = "" if step_um is None else f", Si step limit {step_um:g} um"
    note = (f"<!-- ECal silicon-thickness variant written by sim/make_geometry_variant.py: "
            f"ECal_Si_thickness {t_um:g} um{cut}{step}, source geometry/ sha256 {source_sha} -->\n")
    m = _DECLARATION.match(text)
    if not m:
        return note + text
    return text[:m.end()] + ("" if m.group(1) else "\n") + note + text[m.end():]


def _si_slices(ecal):
    """The sensitive Silicon <slice> elements of my_custom_ecal.xml's text, not commented out."""
    slices = [s for s in _uncommented(ecal, _SLICE)
              if re.search(r'\bmaterial\s*=\s*"Silicon"', s.group(0))
              and re.search(r'\bsensitive\s*=\s*"yes"', s.group(0))]
    if not slices:
        _fail("my_custom_ecal.xml has no sensitive Silicon slice")
    return slices


def _one_limits(compact):
    """The one <limits> opening tag of SiD_TestBeam.xml's text."""
    limits = _uncommented(compact, _LIMITS)
    if len(limits) != 1 or limits[0].group(0).endswith("/>"):
        _fail(f"found {len(limits)} <limits> elements with content in SiD_TestBeam.xml; expected "
              f"exactly one")
    return limits[0]


def _add_region(compact, ecal, cut_um):
    """(compact, ecal): the texts of SiD_TestBeam.xml and my_custom_ecal.xml with the
    SiSensorRegion of production cut cut_um defined before <limits> and named by every sensitive
    Silicon slice."""
    if _uncommented(compact, _REGIONS) or _uncommented(ecal, _REGIONS):
        _fail("the source geometry already has a <regions> element; a production-cut variant "
              "defines its own")
    m = _one_limits(compact)
    indent = m.group(1)
    region = (f'<regions>\n{indent}  <region name="{SI_REGION}" eunit="MeV" lunit="mm" '
              f'cut="{cut_um / 1000.0:.12g}" threshold="0.0"/>\n{indent}</regions>\n{indent}')
    compact = compact[:m.start() + len(indent)] + region + compact[m.start() + len(indent):]

    slices = _si_slices(ecal)
    if any(re.search(r"\bregion\s*=", s.group(0)) for s in slices):
        _fail("a Silicon slice of my_custom_ecal.xml already names a region")
    for s in reversed(slices):   # back to front, so the earlier offsets stay valid
        element = re.sub(r"(\s*/?>)$", rf' region="{SI_REGION}"\1', s.group(0), count=1)
        ecal = ecal[:s.start()] + element + ecal[s.end():]
    return compact, ecal


def _add_step_limit(compact, ecal, step_um):
    """(compact, ecal): the texts of SiD_TestBeam.xml and my_custom_ecal.xml with the limitset
    SI_STEP_LIMITSET, a step_length_max of step_um for every particle, defined first under
    <limits> and named by every sensitive Silicon slice in place of its own limits."""
    name = dg.SI_STEP_LIMITSET
    own = re.compile(r'<limitset\b[^>]*\bname\s*=\s*"' + re.escape(name) + '"')
    if _uncommented(compact, own) or _uncommented(ecal, own):
        _fail(f"the source geometry already defines a limitset {name!r}")
    m = _one_limits(compact)
    indent = m.group(1)
    limitset = (f'\n{indent}  <limitset name="{name}">\n{indent}    <limit name="step_length_max" '
                f'particles="*" value="{step_um / 1000.0:.12g}" unit="mm" />\n{indent}  </limitset>')
    compact = compact[:m.end()] + limitset + compact[m.end():]

    for s in reversed(_si_slices(ecal)):   # back to front, so the earlier offsets stay valid
        element, n = re.subn(r'(\blimits\s*=\s*")[^"]*(")', rf"\g<1>{name}\2", s.group(0), count=1)
        if n == 0:
            element = re.sub(r"(\s*/?>)$", rf' limits="{name}"\1', s.group(0), count=1)
        ecal = ecal[:s.start()] + element + ecal[s.end():]
    return compact, ecal


def _render(t_um, cut_um=None, step_um=None):
    """{relative path: bytes} of the t_um variant (with production cut cut_um and maximum step
    length step_um in the silicon, when given) of the repo's geometry/."""
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
    ecal = xml["my_custom_ecal.xml"]
    if cut_um is not None:
        text, ecal = _add_region(text, ecal, cut_um)
    if step_um is not None:
        text, ecal = _add_step_limit(text, ecal, step_um)
    if ecal != xml["my_custom_ecal.xml"]:
        contents["my_custom_ecal.xml"] = ecal.encode("utf-8")
    contents["SiD_TestBeam.xml"] = _mark(text, t_um, dg.geometry_hash(REPO), cut_um,
                                         step_um).encode("utf-8")
    return contents


def _validate(home, t_um, cut_um=None, step_um=None):
    """Fail unless the tree at `home` has the requested Si thickness, production cut and step
    limit (none without cut_um or step_um), positive air pads, and the repo's Si mid-planes and
    layer depths."""
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
    try:
        cut_mm = dg.si_production_cut_mm(home)
    except ValueError as e:
        _fail(f"the variant's silicon region does not read back: {e}")
    want = None if cut_um is None else cut_um / 1000.0
    if (cut_mm is None) != (want is None) or (want is not None and not abs(cut_mm - want) <= 1e-12 * want):
        _fail(f"variant Si production cut reads back as {cut_mm!r} mm, expected {want!r} mm")
    try:
        step_mm = dg.si_step_limit_mm(home)
    except ValueError as e:
        _fail(f"the variant's silicon step limit does not read back: {e}")
    want = None if step_um is None else step_um / 1000.0
    if (step_mm is None) != (want is None) or (want is not None and not abs(step_mm - want) <= 1e-12 * want):
        _fail(f"variant Si step limit reads back as {step_mm!r} mm, expected {want!r} mm")


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
    extra = [f"a {m.group(2)} um Si production cut" if m and m.group(2) else "",
             f"a {m.group(3)} um Si step limit" if m and m.group(3) else ""]
    extra = " with " + " and ".join(x for x in extra if x) if any(extra) else ""
    what = (f"the {m.group(1)} um variant{extra} of a geometry/ with sha256 {m.group(4)[:10]}..." if m
            else "no variant marker")
    _fail(f"{dst} already exists and is not this variant (it has {what}; differing: "
          f"{', '.join(differ[:5])}{' ...' if len(differ) > 5 else ''}). It is left untouched, since "
          f"a run may be using it: remove it or choose another --out.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--si-thickness-um", type=float, metavar="T",
                    help="ECal sensor silicon thickness in microns (default: the repo geometry's, 320)")
    ap.add_argument("--si-cut-um", type=float, metavar="C",
                    help="Geant4 production cut in the sensor silicon, in microns (default: none, "
                         "the silicon takes the run's cut)")
    ap.add_argument("--si-step-um", type=float, metavar="S",
                    help="Geant4 maximum step length in the sensor silicon, in microns (default: "
                         "none, the silicon takes the calorimeter's 5 mm)")
    ap.add_argument("--out", metavar="DIR",
                    help="config root to write; the variant goes to DIR/geometry/ (default: "
                         "$CALOMAPS_DATA_BASE/geometry_si<T>um[_cut<C>um][_stepmax<S>um]_<hash>)")
    args = ap.parse_args()

    base_um = dg.si_thickness_mm(REPO) * 1e3
    t_um = base_um if args.si_thickness_um is None else args.si_thickness_um
    if not (math.isfinite(t_um) and t_um > 0):
        _fail(f"--si-thickness-um must be a positive thickness, got {t_um!r}")
    cut_um = args.si_cut_um
    if cut_um is not None and not (math.isfinite(cut_um) and cut_um > 0):
        _fail(f"--si-cut-um must be a positive length, got {cut_um!r}")
    step_um = args.si_step_um
    if step_um is not None and not (math.isfinite(step_um) and step_um > 0):
        _fail(f"--si-step-um must be a positive length, got {step_um!r}")
    if dg.si_production_cut_mm(REPO) is not None:
        _fail(f"the repo's geometry/ already puts its silicon in a region of its own; this script "
              f"writes variants of a geometry without one")
    if dg.si_step_limit_mm(REPO) is not None:
        _fail(f"the repo's geometry/ already gives its silicon a step limit of its own; this script "
              f"writes variants of a geometry without one")
    src = os.path.join(REPO, "geometry")
    if args.out is None:
        if "%g" % t_um == "%g" % base_um and cut_um is None and step_um is None:
            print(os.path.join(src, "SiD_TestBeam.xml"))
            return
        out = dg.variant_home(t_um, REPO, si_cut_um=cut_um, si_step_um=step_um)
    else:
        out = os.path.abspath(args.out)
    dst = os.path.join(out, "geometry")
    if os.path.realpath(dst) == os.path.realpath(src) or \
            os.path.realpath(dst).startswith(os.path.realpath(src) + os.sep):
        _fail(f"--out {args.out!r} would write into the repo's own geometry/")

    contents = _render(t_um, cut_um, step_um)
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
        _validate(tmp, t_um, cut_um, step_um)
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
