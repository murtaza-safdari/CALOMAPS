"""Parts of a ddsim steering file's default output name, read from ddsim's command line
(stdlib + numpy). ddsim exec()s the steering file in its own process, so sys.argv is ddsim's.

events_tag(): run_sim_trackermom.py and run_sim_fullcascade.py name their default output
_<N>evt, N being the --numberOfEvents / -N of the command line (the last one, as ddsim's argparse
takes it). Without one, N is the fallback the steering file passes (its SIM.numberOfEvents, which
ddsim's command line would otherwise set) when that is positive; otherwise the name carries no
event count, with a WARNING. events_tag() does not raise.

si_output_tag(): run_sim.py, run_sim_trackermom.py and run_sim_fullcascade.py name their default
output after the geometry ddsim simulates. It finds the compact on the command line, reads its
ECal silicon thickness with analysis/decal_geometry.py and returns "" for this checkout's own
thickness (320 um), else "_si<T>um", the tag sim/geometry_variant.sh gives the same geometry.

CALOMAPS_SI_THICKNESS_UM selects a variant compact only in the sim/*.sh wrappers. Here it is
compared with the compact, and a WARNING is printed when the two disagree. When the compact's
thickness cannot be read (no --compactFile on the command line, an unreadable file, a compact
that is not <config root>/geometry/SiD_TestBeam.xml), the name stays untagged, with a WARNING if
the knob names a thickness other than the checkout's. si_output_tag() does not raise.
"""
import os, sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(_REPO, "analysis"))
import decal_geometry as dg   # noqa: E402

KNOB = "CALOMAPS_SI_THICKNESS_UM"


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


def _um(mm):
    return "%g" % (mm * 1e3)


def si_output_tag(argv=None, environ=None):
    """"" or "_si<T>um" for the default output name of the ddsim run whose command line is argv
    (default sys.argv), T being the silicon thickness of its compact. Prints a WARNING when the
    knob in environ (default os.environ) disagrees with that thickness, or when it names a
    thickness other than the checkout's and the compact's cannot be read."""
    argv = sys.argv if argv is None else argv
    knob = (os.environ if environ is None else environ).get(KNOB, "").strip()
    try:
        knob_um = "%g" % float(knob)
    except ValueError:
        knob_um = knob                  # unset or not a number: equal to no thickness
    base_um = None
    try:
        base_um = _um(dg.si_thickness_mm(_REPO))
        files = compact_files(argv)
        if len(files) != 1:
            raise ValueError("more than one file given to --compactFile" if files
                             else "no --compactFile on the ddsim command line")
        si_um = _um(dg.compact_si_thickness_mm(files[0]))
    except Exception as e:
        if knob and knob_um != base_um:
            print("WARNING: %s=%s, but the silicon thickness of the simulated geometry could not be "
                  "confirmed from its compact (%s). The default output name is untagged; pass the "
                  "variant compact that sim/make_geometry_variant.py prints as --compactFile, or "
                  "name the output with --outputFile." % (KNOB, knob, e))
        return ""
    tag = "" if si_um == base_um else "_si%sum" % si_um
    if knob and knob_um != si_um:
        print("WARNING: the geometry simulated is the --compactFile's, %s um silicon (%s); %s=%s "
              "does not change it, the sim/*.sh wrappers pass the matching variant compact. The "
              "default output name is tagged by the compact: %s."
              % (si_um, files[0], KNOB, knob, tag or "no tag"))
    return tag
