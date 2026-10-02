"""DECAL barrel geometry read from a config root's compact XML (stdlib + numpy only).

A config root is a directory holding geometry/SiD_TestBeam.xml and geometry/my_custom_ecal.xml:
the repo itself, or a copy written by sim/make_geometry_variant.py. extract_readouts.py, the
crossing extractors and the variant generator read the layer stack through this module.
sensor_crossings.si_layer_centers() keeps its own numeric copy of the baseline Si mid-planes;
the crossing extractors compare it with layer_radii() (sensor_crossings.geometry_provenance)
and refuse a geometry where the two disagree.

    layer_radii(home)      per-layer Si mid-plane apothems [mm] (ECalBarrel_rmin + in-layer offset)
    layer_stack(home)      per-layer [(material, thickness_mm), ...], repeats expanded
    si_thickness_mm(home)  the sensor silicon thickness [mm]
    cell_size_mm(home)     the pixel pitch [mm] (ECal_cell_size)
    geometry_hash(home)    content hash of home/geometry/
    variant_home(T, home)  where sim/make_geometry_variant.py puts the T-micron variant of home

The radii are the compact's own numbers. The ECalBarrel_o2_v03 driver places the whole stack a
further env_safety (0.1 mm) out; sensor_crossings.ECAL_STACK_OFFSET_MM carries that offset.
"""
import ast, hashlib, math, operator, os, xml.etree.ElementTree as ET
import numpy as np

UNITS_MM = {'m': 1000.0, 'cm': 10.0, 'mm': 1.0, 'um': 1e-3, 'micron': 1e-3, 'nm': 1e-6}
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _pv(v, c, _resolving=()):
    """Evaluate a compact value in mm (lengths) or as a plain number.

    Allowed: numbers, + - * /, power (^ as in DD4hep's evaluator, or **), unary signs,
    parentheses, unit names and the names of constants in `c`, which are resolved wherever they
    appear. Unit names are whole words ('50*um' parses, '50um' does not, as ddsim also rejects
    it). Anything else -- an unknown name or unit, a constant defined in terms of itself, a
    function call, a non-finite result -- raises ValueError instead of yielding a number."""
    if not v: return 0.0

    def fail(why):
        return ValueError(f"cannot parse geometry value {v!r} ({why}; "
                          f"units understood: {', '.join(sorted(UNITS_MM))})")

    try:
        tree = ast.parse(v.strip().replace("^", "**"), mode="eval")
    except SyntaxError:
        raise fail("not an arithmetic expression") from None
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    unknown = sorted(names - set(UNITS_MM) - set(c))
    if unknown:
        raise fail(f"unknown name(s) {', '.join(map(repr, unknown))}")

    def ev(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARYOPS:
            return _UNARYOPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name):
            if n.id in UNITS_MM:
                return UNITS_MM[n.id]
            if n.id in _resolving:
                raise ValueError(f"constant {n.id!r} is defined in terms of itself")
            return _pv(c[n.id], c, _resolving + (n.id,))
        raise fail(f"{type(n).__name__} is not allowed")

    try:
        x = ev(tree.body)
    except (ArithmeticError, TypeError) as e:
        raise fail(f"{type(e).__name__}: {e}") from None
    if not (isinstance(x, float) and math.isfinite(x)):
        raise fail(f"evaluates to {x!r}")
    return x


def compact_constants(calomaps_home):
    """name -> value string for every <constant> in <home>/geometry/SiD_TestBeam.xml."""
    g = os.path.join(calomaps_home, "geometry")
    return {x.get("name"): x.get("value")
            for x in ET.parse(os.path.join(g, "SiD_TestBeam.xml")).getroot().findall(".//constant")}


def _ecal_barrel(calomaps_home):
    g = os.path.join(calomaps_home, "geometry")
    return ET.parse(os.path.join(g, "my_custom_ecal.xml")).getroot().find(".//detector[@name='ECalBarrel']")


def _layer_blocks(calomaps_home):
    """(rmin_mm, [(repeat, [(material, thickness_mm), ...]), ...]) for the barrel ECal."""
    consts, det = compact_constants(calomaps_home), _ecal_barrel(calomaps_home)
    blocks = [(int(layer.get("repeat", 1)),
               [(s.get("material"), _pv(s.get("thickness"), consts)) for s in layer.findall("slice")])
              for layer in det.findall("layer")]
    return _pv(det.find("dimensions").get("rmin"), consts), blocks


def layer_radii(calomaps_home):
    cur, blocks = _layer_blocks(calomaps_home)
    planes = []
    for rep, sl in blocks:
        thick = sum(t for _, t in sl)
        off = sioff = 0.0
        for material, t in sl:
            if material == "Silicon": sioff = off + t / 2
            off += t
        for _ in range(rep):
            planes.append(cur + sioff); cur += thick
    return np.array(planes)


def layer_stack(calomaps_home):
    _, blocks = _layer_blocks(calomaps_home)
    return [list(sl) for rep, sl in blocks for _ in range(rep)]


def si_thickness_mm(calomaps_home):
    """Sensor silicon thickness [mm]. Raises ValueError unless every layer holds exactly one
    Silicon slice and all of them have the same thickness (the crossing code assumes one)."""
    _, blocks = _layer_blocks(calomaps_home)
    found = set()
    for i, (_, sl) in enumerate(blocks):
        si = [t for material, t in sl if material == "Silicon"]
        if len(si) != 1:
            raise ValueError(f"layer block {i} has {len(si)} Silicon slices (expected 1)")
        found.add(si[0])
    if len(found) != 1:
        raise ValueError(f"layer blocks have different Silicon thicknesses: {sorted(found)} mm")
    return found.pop()


def cell_size_mm(calomaps_home):
    """Pixel pitch [mm] from the compact's ECal_cell_size -- the same constant that
    drives the CartesianGridXY segmentation, so the cluster readout is built on the
    simulated pitch instead of trusting a hand-synced copy."""
    g = os.path.join(calomaps_home, "geometry")
    consts = compact_constants(calomaps_home)
    v = _pv(consts.get("ECal_cell_size"), consts)
    if v <= 0:
        raise SystemExit(f"could not resolve ECal_cell_size from {g}/SiD_TestBeam.xml "
                         f"(got {v!r}) -- does --home point at the geometry you simulated with?")
    return v


def pop_home_arg(argv, default):
    """Split `--home DIR` (or `--home=DIR`) out of a script's argument list.
    Returns (home, remaining arguments in their original order)."""
    home, rest, i = default, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--home":
            if i + 1 >= len(argv):
                raise SystemExit("--home needs a directory (the config root that was simulated)")
            home, i = argv[i + 1], i + 2
            continue
        if a.startswith("--home="):
            home = a.split("=", 1)[1]
        else:
            rest.append(a)
        i += 1
    return home, rest


def geometry_files(calomaps_home):
    """Relative paths of the files under <home>/geometry, sorted, skipping hidden entries
    (.DS_Store, ._*, .nfs*)."""
    src, out = os.path.join(calomaps_home, "geometry"), []
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        out += [os.path.relpath(os.path.join(root, f), src) for f in sorted(files) if not f.startswith(".")]
    return out


def geometry_hash(calomaps_home):
    """sha256 hex digest of <home>/geometry: every file's relative path and bytes."""
    h, src = hashlib.sha256(), os.path.join(calomaps_home, "geometry")
    for rel in geometry_files(calomaps_home):
        with open(os.path.join(src, rel), "rb") as f:
            data = f.read()
        h.update(f"{rel}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def variant_home(si_thickness_um, source_home, data_base=None):
    """The config root sim/make_geometry_variant.py writes, by default, for the si_thickness_um
    variant of source_home's geometry: <data_base>/geometry_si<T>um_<hash>, where <hash> is the
    first 10 hex digits of geometry_hash(source_home) and data_base defaults to
    $CALOMAPS_DATA_BASE, else ~/CALOMAPS-data. A changed source geometry gets its own directory."""
    base = data_base or os.environ.get("CALOMAPS_DATA_BASE") or os.path.expanduser("~/CALOMAPS-data")
    return os.path.join(base, "geometry_si%gum_%s" % (si_thickness_um, geometry_hash(source_home)[:10]))
