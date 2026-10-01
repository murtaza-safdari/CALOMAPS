"""DECAL barrel geometry read from a config root's compact XML (stdlib + numpy only).

A config root is a directory holding geometry/SiD_TestBeam.xml and geometry/my_custom_ecal.xml:
the repo itself, or a copy written by sim/make_geometry_variant.py. Every reader of the layer
stack (extract_readouts.py, the crossing extractors, the variant generator) parses it here.

    layer_radii(home)      per-layer Si mid-plane apothems [mm] (ECalBarrel_rmin + in-layer offset)
    layer_stack(home)      per-layer [(material, thickness_mm), ...], repeats expanded
    si_thickness_mm(home)  the sensor silicon thickness [mm]
    cell_size_mm(home)     the pixel pitch [mm] (ECal_cell_size)

The radii are the compact's own numbers. The ECalBarrel_o2_v03 driver places the whole stack a
further env_safety (0.1 mm) out; sensor_crossings.ECAL_STACK_OFFSET_MM carries that offset.
"""
import os, re, xml.etree.ElementTree as ET
import numpy as np

UNITS_MM = {'m': 1000.0, 'cm': 10.0, 'mm': 1.0, 'um': 1e-3, 'micron': 1e-3, 'nm': 1e-6}
_NAME = re.compile(r'\b[A-Za-z_]\w*')


def _pv(v, c, _resolving=()):
    """Evaluate a compact value in mm (lengths) or as a plain number. Constant names from `c`
    are resolved wherever they appear, including inside expressions; unit names are whole-word
    ('50*um' parses, '50um' does not, as ddsim also rejects it). An unknown name or unit raises
    ValueError instead of yielding 0."""
    if not v: return 0.0
    unknown = []

    def sub(m):
        name = m.group(0)
        if name in UNITS_MM:
            return repr(UNITS_MM[name])
        if name in c:
            if name in _resolving:
                raise ValueError(f"constant {name!r} is defined in terms of itself")
            return "(" + repr(_pv(c[name], c, _resolving + (name,))) + ")"
        unknown.append(name)
        return name

    s = _NAME.sub(sub, v)
    try: return float(eval(s, {"__builtins__": None}, {}))
    except Exception:
        try: return float(v)
        except Exception:
            names = f"unknown name(s) {', '.join(map(repr, unknown))}; " if unknown else ""
            raise ValueError(f"cannot parse geometry value {v!r} "
                             f"({names}units understood: {', '.join(sorted(UNITS_MM))})")


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
