# geometry_variant.sh -- sourced by the sim wrappers (make_pixelav_inputs.sh, generate_batched.sh,
# generate_dataset.sh) to resolve the ECal silicon-thickness knob:
#
#   CALOMAPS_SI_THICKNESS_UM   sensor thickness in um; unset, or the repo geometry's own
#                              thickness (320), simulates the repo geometry unchanged
#
# Any other value simulates the copy sim/make_geometry_variant.py writes, or finds already
# written, at ${CALOMAPS_DATA_BASE:-~/CALOMAPS-data}/geometry_si<T>um_<hash>/ (<hash>: the repo's
# geometry/ content; the generator prints the path), in which only the ECal_Si_thickness and
# air-pad constants differ (layer depths and Si mid-planes are the baseline's), and tags every
# output name of the run with _si<T>um so it cannot overwrite or pass for baseline output.
#
# Needs REPO_DIR. Sets:
#   COMPACT    the compact file for ddsim
#   GEO_HOME   its config root (pass to the extractors as --home, to extract_readouts.py as --home)
#   SI_TAG     "" for the baseline, else "_si<T>um"
# and defines si_tagged NAME, which prints NAME with SI_TAG appended unless it already ends in it.
# Returns non-zero (callers: `source .../geometry_variant.sh || exit 1`) on a value that is not a
# plain number of microns or when the variant cannot be generated.

COMPACT="$REPO_DIR/geometry/SiD_TestBeam.xml"
GEO_HOME="$REPO_DIR"
SI_TAG=""

si_tagged() {
    case "$1" in
        *"$SI_TAG") printf '%s\n' "$1" ;;
        *)          printf '%s\n' "$1$SI_TAG" ;;
    esac
}

if [ -n "${CALOMAPS_SI_THICKNESS_UM:-}" ]; then
    if ! [[ "$CALOMAPS_SI_THICKNESS_UM" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "ERROR: CALOMAPS_SI_THICKNESS_UM='$CALOMAPS_SI_THICKNESS_UM' is not a thickness in microns" >&2
        return 1
    fi
    _si_um="$(LC_ALL=C printf '%g' "$CALOMAPS_SI_THICKNESS_UM")"
    # The generator prints the repo's own compact when _si_um is the repo thickness.
    COMPACT="$("${CALOMAPS_PY:-python3}" "$REPO_DIR/sim/make_geometry_variant.py" \
                 --si-thickness-um "$_si_um")" || return 1
    if [ "$COMPACT" != "$REPO_DIR/geometry/SiD_TestBeam.xml" ]; then
        SI_TAG="_si${_si_um}um"
        GEO_HOME="$(dirname "$(dirname "$COMPACT")")"
        echo "Si thickness ${_si_um} um: geometry variant $COMPACT (outputs tagged $SI_TAG)"
    fi
    unset _si_um
fi
