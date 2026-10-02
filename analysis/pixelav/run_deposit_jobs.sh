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
# ============================================================================
set -euo pipefail
ulimit -s unlimited 2>/dev/null || ulimit -s 65520 2>/dev/null || true

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
  sed -n '2,16p' "$0"; exit 1
fi
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

echo "[deposit-jobs] ${#PARTS[@]} part(s), $NPROC at a time, seed $SEED, sensor $(head -1 "$RUN/ppixel2.init")"
cd "$RUN"
START=$SECONDS
printf '%s\n' "${PARTS[@]}" | xargs -P "$NPROC" -I{} bash -c \
  'p="$1"; o="${p%.txt}"; o="${o%.jobs}"; "$2" "$p" "$o.out" "$3" last > "$o.log" 2>&1 || echo "[deposit-jobs] FAILED: $p (see $o.log)"' _ {} "$DRIVER" "$SEED"

FAIL=0; JOBS=0; PAIRS=0; OUTSIDE=0
for p in "${PARTS[@]}"; do
  o="${p%.txt}"; o="${o%.jobs}"
  LINE="$(grep '^DONE:' "$o.log" || true)"
  if [ -z "$LINE" ]; then
    echo "[deposit-jobs] ERROR: $p did not finish: $(grep -m1 '^ERROR' "$o.log" || tail -1 "$o.log")"; FAIL=1; continue
  fi
  read -r J P O <<<"$(echo "$LINE" | awk '{print $2, $4, $7}')"
  JOBS=$((JOBS + J)); PAIRS=$((PAIRS + P)); OUTSIDE=$((OUTSIDE + O))
done
echo "[deposit-jobs] $JOBS jobs, $PAIRS pairs transported, $OUTSIDE pairs outside their window, $((SECONDS - START)) s"
exit $FAIL
