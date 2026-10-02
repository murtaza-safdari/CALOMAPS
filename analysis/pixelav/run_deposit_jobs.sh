#!/usr/bin/env bash
# ============================================================================
# CALOMAPS: run ppixelav2_inject_deposits over a set of deposit jobs, several parts at a time.
#
#   bash analysis/pixelav/run_deposit_jobs.sh <jobs prefix> <PIXELAV run dir> [nproc] [seed]
#
#   <jobs prefix>       the prefix given to analysis/pixelav/deposit_jobs.py: its parts
#                       <prefix>.part000.txt ... (made with --parts N), or <prefix>.jobs.txt
#   <PIXELAV run dir>   a sensor directory built by setup/setup_pixelav.sh (ppixel2.init,
#                       wgt_pot.init and the driver); it must be the one the jobs were built for
#   nproc               parts run at the same time (default: the number of parts, at most 8)
#   seed                the driver's seed (default 1); a job's cluster depends on (seed, job id)
#                       only, so the result does not depend on how the jobs were split into parts
#
# Each part <prefix>.partNNN.txt gives <prefix>.partNNN.out (clusters, last time slice) and
# <prefix>.partNNN.log. Existing outputs are overwritten. Exit 1 when any part fails.
# <prefix>.run.json records the run, also when a part fails (analysis/pixelav/charge_maps.py
# stores it): driver seed, nproc, parts and the failed ones, the sensor title (ppixel2.init line
# 1) and "xsize ysize thick" (line 2 of the first finished part's output), sha256 of
# ppixel2.init, wgt_pot.init and the driver, the PIXELAV source commit (the checkout above the
# run directory) and this checkout's commit ("unknown" where git cannot tell), host, UTC start,
# wall seconds, the jobs, pairs transported and pairs outside their window of the parts that
# finished, and the exit status.
# ============================================================================
set -euo pipefail
ulimit -s unlimited 2>/dev/null || ulimit -s 65520 2>/dev/null || true

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
  sed -n '2,23p' "$0"; exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$1"; RUN="$(cd "$2" && pwd)"; SEED="${4:-1}"
DRIVER="$RUN/ppixelav2_inject_deposits"
[ -x "$DRIVER" ] || { echo "[deposit-jobs] ERROR: no ppixelav2_inject_deposits in $RUN (run setup/setup_pixelav.sh)"; exit 1; }

case "$PREFIX" in /*) ;; *) PREFIX="$PWD/$PREFIX" ;; esac
shopt -s nullglob
PARTS=("$PREFIX".part[0-9][0-9][0-9].txt)
shopt -u nullglob
[ ${#PARTS[@]} -gt 0 ] || PARTS=("$PREFIX.jobs.txt")
[ -f "${PARTS[0]}" ] || { echo "[deposit-jobs] ERROR: no $PREFIX.partNNN.txt and no $PREFIX.jobs.txt"; exit 1; }
NPROC="${3:-$(( ${#PARTS[@]} < 8 ? ${#PARTS[@]} : 8 ))}"

TITLE="$(head -1 "$RUN/ppixel2.init" || true)"
echo "[deposit-jobs] ${#PARTS[@]} part(s), $NPROC at a time, seed $SEED, sensor $TITLE"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$RUN"
START=$SECONDS
printf '%s\n' "${PARTS[@]}" | xargs -P "$NPROC" -I{} bash -c \
  'p="$1"; o="${p%.txt}"; o="${o%.jobs}"; "$2" "$p" "$o.out" "$3" last > "$o.log" 2>&1 || echo "[deposit-jobs] FAILED: $p (see $o.log)"' _ {} "$DRIVER" "$SEED"

FAIL=0; JOBS=0; PAIRS=0; OUTSIDE=0; FAILED=(); FIRST_OUT=""
for p in "${PARTS[@]}"; do
  o="${p%.txt}"; o="${o%.jobs}"
  LINE="$(grep '^DONE:' "$o.log" || true)"
  if [ -z "$LINE" ]; then
    echo "[deposit-jobs] ERROR: $p did not finish: $(grep -m1 '^ERROR' "$o.log" || tail -1 "$o.log")"; FAIL=1
    FAILED+=("$p"); continue
  fi
  read -r J P O <<<"$(echo "$LINE" | awk '{print $2, $4, $7}')"
  JOBS=$((JOBS + J)); PAIRS=$((PAIRS + P)); OUTSIDE=$((OUTSIDE + O))
  [ -n "$FIRST_OUT" ] || FIRST_OUT="$o.out"
done
WALL=$((SECONDS - START))
echo "[deposit-jobs] $JOBS jobs, $PAIRS pairs transported, $OUTSIDE pairs outside their window, $WALL s"

# ---- run record: <prefix>.run.json ----
json_str() {   # $1 as a JSON string, control characters dropped
  local s="${1//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf '"%s"' "$(printf '%s' "$s" | tr -d '\000-\037')"
}
json_num() {   # $1 as a JSON number when it is one, else as a string
  if [[ "$1" =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then printf '%s' "$1"; else json_str "$1"; fi
}
json_list() {  # the arguments as a JSON list of strings
  local sep="" a
  printf '['
  for a in "$@"; do printf '%s' "$sep"; json_str "$a"; sep=", "; done
  printf ']'
}
sha256() {     # sha256 of the file $1, or "unknown"
  local h
  h="$( { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1" 2>/dev/null; } | awk '{print $1}' || true)"
  printf '%s' "${h:-unknown}"
}
git_head() {   # HEAD of the git checkout at $1, or "unknown"
  local h
  h="$(git -C "$1" rev-parse --verify -q HEAD 2>/dev/null || true)"
  printf '%s' "${h:-unknown}"
}
SRC="$(dirname "$RUN")"
PIXELAV_COMMIT="unknown"
[ -e "$SRC/.git" ] && PIXELAV_COMMIT="$(git_head "$SRC")"
SIZES=()
[ -n "$FIRST_OUT" ] && [ -f "$FIRST_OUT" ] && read -r -a SIZES <<<"$(sed -n 2p "$FIRST_OUT" || true)"
size() { if [ ${#SIZES[@]} -ge 3 ]; then json_num "${SIZES[$1]}"; else printf 'null'; fi; }
RECORD="$PREFIX.run.json"
{
  printf '{\n'
  printf '  "prefix": %s,\n' "$(json_str "$PREFIX")"
  printf '  "run_dir": %s,\n' "$(json_str "$RUN")"
  printf '  "driver": %s,\n' "$(json_str "$DRIVER")"
  printf '  "driver_seed": %s,\n' "$(json_num "$SEED")"
  printf '  "slices": "last",\n'
  printf '  "nproc": %s,\n' "$(json_num "$NPROC")"
  printf '  "n_parts": %s,\n' "${#PARTS[@]}"
  printf '  "failed_parts": %s,\n' "$(json_list ${FAILED[@]+"${FAILED[@]}"})"
  printf '  "sensor_title": %s,\n' "$(json_str "$TITLE")"
  printf '  "xsize_um": %s,\n' "$(size 0)"
  printf '  "ysize_um": %s,\n' "$(size 1)"
  printf '  "thick_um": %s,\n' "$(size 2)"
  printf '  "sha256_ppixel2_init": %s,\n' "$(json_str "$(sha256 "$RUN/ppixel2.init")")"
  printf '  "sha256_wgt_pot_init": %s,\n' "$(json_str "$(sha256 "$RUN/wgt_pot.init")")"
  printf '  "sha256_driver": %s,\n' "$(json_str "$(sha256 "$DRIVER")")"
  printf '  "pixelav_commit": %s,\n' "$(json_str "$PIXELAV_COMMIT")"
  printf '  "calomaps_commit": %s,\n' "$(json_str "$(git_head "$SCRIPT_DIR")")"
  printf '  "host": %s,\n' "$(json_str "$(hostname 2>/dev/null || uname -n)")"
  printf '  "start_utc": %s,\n' "$(json_str "$START_UTC")"
  printf '  "wall_s": %s,\n' "$WALL"
  printf '  "jobs": %s,\n' "$JOBS"
  printf '  "pairs": %s,\n' "$PAIRS"
  printf '  "pairs_outside": %s,\n' "$OUTSIDE"
  printf '  "exit_status": %s\n' "$FAIL"
  printf '}\n'
} > "$RECORD"
echo "[deposit-jobs] run record: $RECORD"
exit $FAIL
