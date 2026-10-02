"""Parts of a ddsim steering file's default output name, read from ddsim's command line
(stdlib + numpy). ddsim exec()s the steering file in its own process, so sys.argv is ddsim's.

events_tag(): run_sim_trackermom.py and run_sim_fullcascade.py name their default output
_<N>evt, N being the --numberOfEvents / -N of the command line (the last one, as ddsim's argparse
takes it). Without one, N is the fallback the steering file passes (its SIM.numberOfEvents, which
ddsim's command line would otherwise set) when that is positive; otherwise the name carries no
event count, with a WARNING. events_tag() does not raise.

seed_tag(): run_sim_trackermom.py and run_sim_fullcascade.py name their default output _seed<R>
after the random seed R of the run when it is not DEFAULT_SEED, their seed without
CALOMAPS_SEED: R is the --random.seed of the command line (the last one), which ddsim applies over
the steering file's, else the fallback the steering file passes (its SIM.random.seed, read from
CALOMAPS_SEED). The default seed adds nothing to the name. Without an integer seed from either,
the name carries no seed tag, with a WARNING. seed_tag() does not raise.

si_output_tag(): run_sim.py, run_sim_trackermom.py and run_sim_fullcascade.py name their default
output after the geometry ddsim simulates. It finds the compact on the command line, reads its
ECal silicon thickness, the production cut of the silicon's own region and the maximum step
length of the silicon's own limitset with analysis/decal_geometry.py, and returns "_si<T>um" for a
thickness other than this checkout's own (320 um) followed by "_cut<C>um" when the silicon has a
production cut of its own and "_stepmax<S>um" when it has a step limit of its own: "" for the
repo geometry, and in every case the tag sim/geometry_variant.sh gives the same geometry.

CALOMAPS_SI_THICKNESS_UM, CALOMAPS_SI_CUT_UM and CALOMAPS_SI_STEP_UM select a variant compact only
in the sim/*.sh wrappers. Here they are compared with the compact, and a WARNING is printed when a
knob and the compact disagree. When the compact cannot be read (no --compactFile on the command
line, an unreadable file, a compact that is not <config root>/geometry/SiD_TestBeam.xml), the name
stays untagged, with a WARNING if the thickness knob names a thickness other than the checkout's
or the cut or step knob is set. si_output_tag() does not raise.
"""
import os, sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(_REPO, "analysis"))
import decal_geometry as dg   # noqa: E402

DEFAULT_SEED = 424242   # the steering files' SIM.random.seed without CALOMAPS_SEED
KNOB = "CALOMAPS_SI_THICKNESS_UM"
CUT_KNOB = "CALOMAPS_SI_CUT_UM"
STEP_KNOB = "CALOMAPS_SI_STEP_UM"
# the silicon's own settings, in tag order: (knob, reader of a compact, tag, what it is)
_SETTINGS = ((CUT_KNOB, dg.compact_si_cut_mm, "cut", "production cut"),
             (STEP_KNOB, dg.compact_si_step_mm, "stepmax", "step limit"))


def compact_files(argv):
    """The files given to the last --compactFile in argv (ddsim's option takes one or more, or
    one as --compactFile=FILE), or [] when there is none."""
    files, i = [], 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--compactFile="):
            files = [a.split("=", 1)[1]]
        elif a == "--compactFile":
            files = []
            while i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 1
                files.append(argv[i])
        i += 1
    return files


def number_of_events(argv):
    """The value of the last --numberOfEvents N, --numberOfEvents=N, -N N or -NN in argv, or
    None when there is none or it is not an integer."""
    n, i = None, 1
    while i < len(argv):
        a, value = argv[i], None
        if a in ("--numberOfEvents", "-N"):
            if i + 1 < len(argv):
                i += 1
                value = argv[i]
        elif a.startswith("--numberOfEvents="):
            value = a.split("=", 1)[1]
        elif a.startswith("-N") and not a.startswith("--"):
            value = a[2:]
        if value is not None:
            try:
                n = int(value)
            except ValueError:
                n = None
        i += 1
    return n


def events_tag(argv=None, fallback=None):
    """"_<N>evt" for the default output name of the ddsim run whose command line is argv
    (default sys.argv); fallback is the N to use when the command line gives none."""
    try:
        n = number_of_events(sys.argv if argv is None else argv)
        if n is None and isinstance(fallback, int) and fallback > 0:
            n = fallback
    except Exception:
        n = None
    if n is None:
        print("WARNING: no --numberOfEvents on the ddsim command line; the default output name "
              "carries no event count. Name the output with --outputFile.")
        return ""
    return "_%devt" % n


def random_seed(argv):
    """The value of the last --random.seed R or --random.seed=R in argv, or None when there is
    none or it is not an integer."""
    seed, i = None, 1
    while i < len(argv):
        a, value = argv[i], None
        if a == "--random.seed":
            if i + 1 < len(argv):
                i += 1
                value = argv[i]
        elif a.startswith("--random.seed="):
            value = a.split("=", 1)[1]
        if value is not None:
            try:
                seed = int(value)
            except ValueError:
                seed = None
        i += 1
    return seed


def seed_tag(argv=None, fallback=None):
    """"_seed<R>" for the default output name of the ddsim run whose command line is argv
    (default sys.argv), or "" when R is DEFAULT_SEED; fallback is the seed to use when the
    command line gives none (the steering file's SIM.random.seed)."""
    try:
        seed = random_seed(sys.argv if argv is None else argv)
        if seed is None and isinstance(fallback, int) and not isinstance(fallback, bool):
            seed = fallback
    except Exception:
        seed = None
    if seed is None:
        print("WARNING: no integer random seed on the ddsim command line or in the steering file; "
              "the default output name carries no seed. Name the output with --outputFile.")
        return ""
    return "" if seed == DEFAULT_SEED else "_seed%d" % seed


def _um(mm):
    return "%g" % (mm * 1e3)


def _knob_um(environ, name):
    """(raw value, "%g" of its number) of the knob `name` in environ; the raw value twice when it
    is unset or not a number."""
    knob = environ.get(name, "").strip()
    try:
        return knob, "%g" % float(knob)
    except ValueError:
        return knob, knob


def si_output_tag(argv=None, environ=None):
    """"[_si<T>um][_cut<C>um][_stepmax<S>um]" for the default output name of the ddsim run whose
    command line is argv (default sys.argv): T the silicon thickness of its compact when it is not
    the checkout's, C the production cut of the silicon's own region and S the maximum step length
    of its own limitset, when it has them. Prints a WARNING when a knob in environ (default
    os.environ) disagrees with the compact, or, when the compact cannot be read, when the
    thickness knob names a thickness other than the checkout's or the cut or step knob is set."""
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    knob, knob_um = _knob_um(environ, KNOB)
    knobs = [_knob_um(environ, name) for name, _, _, _ in _SETTINGS]
    base_um = None
    try:
        base_um = _um(dg.si_thickness_mm(_REPO))
        files = compact_files(argv)
        if len(files) != 1:
            raise ValueError("more than one file given to --compactFile" if files
                             else "no --compactFile on the ddsim command line")
        si_um = _um(dg.compact_si_thickness_mm(files[0]))
        values = [read(files[0]) for _, read, _, _ in _SETTINGS]
    except Exception as e:
        if knob and knob_um != base_um:
            print("WARNING: %s=%s, but the silicon thickness of the simulated geometry could not be "
                  "confirmed from its compact (%s). The default output name is untagged; pass the "
                  "variant compact that sim/make_geometry_variant.py prints as --compactFile, or "
                  "name the output with --outputFile." % (KNOB, knob, e))
        for (name, _, _, what), (raw, _) in zip(_SETTINGS, knobs):
            if raw:
                print("WARNING: %s=%s, but the %s of the simulated silicon could not be "
                      "confirmed from its compact (%s). The default output name is untagged; pass "
                      "the variant compact that sim/make_geometry_variant.py prints as "
                      "--compactFile, or name the output with --outputFile." % (name, raw, what, e))
        return ""
    ums = [None if mm is None else _um(mm) for mm in values]
    tag = ("" if si_um == base_um else "_si%sum" % si_um) + "".join(
        "" if um is None else "_%s%sum" % (prefix, um) for (_, _, prefix, _), um in zip(_SETTINGS, ums))
    if knob and knob_um != si_um:
        print("WARNING: the geometry simulated is the --compactFile's, %s um silicon (%s); %s=%s "
              "does not change it, the sim/*.sh wrappers pass the matching variant compact. The "
              "default output name is tagged by the compact: %s."
              % (si_um, files[0], KNOB, knob, tag or "no tag"))
    for (name, _, _, what), (raw, raw_um), um in zip(_SETTINGS, knobs, ums):
        if raw and raw_um != um:
            print("WARNING: the geometry simulated is the --compactFile's, %s (%s); %s=%s does not "
                  "change it, the sim/*.sh wrappers pass the matching variant compact. The default "
                  "output name is tagged by the compact: %s."
                  % ("no %s of its own in the silicon" % what if um is None
                     else "a %s um %s in the silicon" % (um, what),
                     files[0], name, raw, tag or "no tag"))
    return tag
