# geometry_variant.sh -- sourced by the sim wrappers (make_pixelav_inputs.sh, generate_batched.sh,
# generate_dataset.sh) to resolve the ECal silicon knobs:
#
#   CALOMAPS_SI_THICKNESS_UM   sensor thickness in um; unset, or the repo geometry's own
#                              thickness (320), simulates the repo geometry unchanged
#   CALOMAPS_SI_CUT_UM         Geant4 production cut in the sensor silicon, in um; unset keeps the
#                              run's own cut there (no region of its own)
#   CALOMAPS_SI_STEP_UM        Geant4 maximum step length in the sensor silicon, in um; unset keeps
#                              the calorimeter's (cal_limits, 5 mm)
#
# Any other value simulates the copy sim/make_geometry_variant.py writes, or finds already
# written, at ${CALOMAPS_DATA_BASE:-~/CALOMAPS-data}/geometry_si<T>um[_cut<C>um][_stepmax<S>um]_<hash>/
# (<hash>: the repo's geometry/ content; the generator prints the path), in which only the
# ECal_Si_thickness and air-pad constants differ (layer depths and Si mid-planes are the
# baseline's) and, with a cut, the silicon sits in a region of its own, and with a step limit it
# has a limitset of its own. Every output name of the run is tagged _si<T>um (a thickness other
# than the repo's), _cut<C>um (a cut) and _stepmax<S>um (a step limit), in that order, so it
# cannot overwrite or pass for baseline output: the tag sim/steering_output.py gives the same
# compact.
#
# Needs REPO_DIR. Sets:
#   COMPACT    the compact file for ddsim
#   GEO_HOME   its config root (pass to the extractors as --home, to extract_readouts.py as --home)
#   SI_TAG     "" for the baseline, else [_si<T>um][_cut<C>um][_stepmax<S>um]
# and defines si_tagged NAME, which prints NAME with SI_TAG appended unless it already ends in it.
# Returns non-zero (callers: `source .../geometry_variant.sh || exit 1`) on a value that is not a
# plain number of microns, or one whose tag would carry an exponent (below 0.0001 or from 1e6),
# before anything is written, or when the variant cannot be generated.

COMPACT="$REPO_DIR/geometry/SiD_TestBeam.xml"
GEO_HOME="$REPO_DIR"
SI_TAG=""

si_tagged() {
    case "$1" in
        *"$SI_TAG") printf '%s\n' "$1" ;;
        *)          printf '%s\n' "$1$SI_TAG" ;;
    esac
}

# The %g spelling of the knob $1's value $2, which the output-name tags use; refused when it has an
# exponent (1e-05, 2e+06), since the extractors read only plain numbers back from a name.
_si_knob_g() {
    local g
    g="$(LC_ALL=C printf '%g' "$2")"
    if ! [[ "$g" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "ERROR: $1='$2' would be spelled $g in the output names, which the extractors cannot read back; give a value of at least 0.0001 and below 1000000" >&2
        return 1
    fi
    printf '%s\n' "$g"
}

if [ -n "${CALOMAPS_SI_THICKNESS_UM:-}" ] || [ -n "${CALOMAPS_SI_CUT_UM:-}" ] || \
   [ -n "${CALOMAPS_SI_STEP_UM:-}" ]; then
    _variant_args=()
    if [ -n "${CALOMAPS_SI_THICKNESS_UM:-}" ]; then
        if ! [[ "$CALOMAPS_SI_THICKNESS_UM" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "ERROR: CALOMAPS_SI_THICKNESS_UM='$CALOMAPS_SI_THICKNESS_UM' is not a thickness in microns" >&2
            return 1
        fi
        _g="$(_si_knob_g CALOMAPS_SI_THICKNESS_UM "$CALOMAPS_SI_THICKNESS_UM")" || return 1
        _variant_args+=(--si-thickness-um "$_g")
    fi
    if [ -n "${CALOMAPS_SI_CUT_UM:-}" ]; then
        if ! [[ "$CALOMAPS_SI_CUT_UM" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "ERROR: CALOMAPS_SI_CUT_UM='$CALOMAPS_SI_CUT_UM' is not a production cut in microns" >&2
            return 1
        fi
        _g="$(_si_knob_g CALOMAPS_SI_CUT_UM "$CALOMAPS_SI_CUT_UM")" || return 1
        _variant_args+=(--si-cut-um "$_g")
    fi
    if [ -n "${CALOMAPS_SI_STEP_UM:-}" ]; then
        if ! [[ "$CALOMAPS_SI_STEP_UM" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "ERROR: CALOMAPS_SI_STEP_UM='$CALOMAPS_SI_STEP_UM' is not a step length in microns" >&2
            return 1
        fi
        _g="$(_si_knob_g CALOMAPS_SI_STEP_UM "$CALOMAPS_SI_STEP_UM")" || return 1
        _variant_args+=(--si-step-um "$_g")
    fi
    # The generator prints the repo's own compact for the repo thickness without a cut or step limit.
    COMPACT="$("${CALOMAPS_PY:-python3}" "$REPO_DIR/sim/make_geometry_variant.py" \
                 "${_variant_args[@]}")" || return 1
    if [ "$COMPACT" != "$REPO_DIR/geometry/SiD_TestBeam.xml" ]; then
        GEO_HOME="$(dirname "$(dirname "$COMPACT")")"
        SI_TAG="$("${CALOMAPS_PY:-python3}" - "$REPO_DIR/sim" "$COMPACT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from steering_output import si_output_tag
print(si_output_tag(["ddsim", "--compactFile", sys.argv[2]], {}))
PY
)" || return 1
        if [ -z "$SI_TAG" ]; then
            echo "ERROR: the variant $COMPACT reads back as the repo geometry (no _si<T>um/_cut<C>um/_stepmax<S>um tag)" >&2
            return 1
        fi
        echo "Si variant: geometry $COMPACT (outputs tagged $SI_TAG)"
    fi
    unset _variant_args _g
fi
unset -f _si_knob_g
