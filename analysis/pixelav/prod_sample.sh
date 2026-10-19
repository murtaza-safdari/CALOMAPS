#!/usr/bin/env bash
# ============================================================================
# CALOMAPS: produce a sample batch by batch through the deposits route: simulation, PIXELAV
# deposit jobs, the injection driver and the charge maps, then copy the compact results.
#
#   bash analysis/pixelav/prod_sample.sh <particle> <energy GeV> <first batch> <last batch> \
#        <driver seed base> <work dir> <dest dir>
#
#   batch b simulates PROD_EVENTS events of <particle> at <energy GeV> in 100 um silicon with the
#   sim seed PROD_SIM_SEED_BASE + 1000 b, and runs the driver with the seed <driver seed base> + b.
#   No batch may get the driver seed 1 (the seed of the legacy sample), so that the batches of all
#   samples can be combined. The work dir holds the full outputs; it must not be the dest dir.
# Knobs (environment):
#   PIXELAV_RUN_DIR      the PIXELAV run directory of the sensor (required; setup/setup_pixelav.sh)
#   PROD_EVENTS          events per batch (default 20)
#   PROD_NPROC           driver processes at a time (default 30)
#   PROD_PARTS           job parts per batch (default 30)
#   PROD_SIM_SEED_BASE   sim seed of batch 0 (default 424242)
#   CALOMAPS_HOME        the checkout whose scripts run (default: the one holding this script)
#   PROD_KEEP_WORK       set: keep <work>/b<b> after the copy; unset: remove it once verified
#   PROD_DRY_RUN         set: print every command that would run, run nothing
# Writes <dest>/b<b>/: the jobs.npz, summary.txt, run.json and charges.npz of the batch, its
# inputs.log and run.log, the part outputs gzipped (.out.gz), then DONE once the copy is verified
# (file count, npz byte sizes, gzip -t). A batch with DONE is skipped, so a rerun resumes. A batch
# that fails gets FAILED (the stage and the log tail) and the next batch runs. <dest>/ALL_DONE is
# written when no batch failed. The Key4hep stack is sourced when ddsim is not on PATH.
# ============================================================================
set -u

if [ $# -ne 7 ]; then sed -n '2,27p' "$0"; exit 1; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_HOME="${CALOMAPS_HOME:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
P_PART="$1"; P_GEV="$2"; P_FIRST="$3"; P_LAST="$4"; P_DBASE="$5"
P_EVENTS="${PROD_EVENTS:-20}"; P_NPROC="${PROD_NPROC:-30}"; P_NPARTS="${PROD_PARTS:-30}"
P_SBASE="${PROD_SIM_SEED_BASE:-424242}"; PY="${CALOMAPS_PY:-python3}"; DRY="${PROD_DRY_RUN:-}"

die() { echo "[prod] ERROR: $*" >&2; exit 1; }
isint() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; }
for v in "first batch=$P_FIRST" "last batch=$P_LAST" "driver seed base=$P_DBASE" "PROD_EVENTS=$P_EVENTS" \
         "PROD_NPROC=$P_NPROC" "PROD_PARTS=$P_NPARTS" "PROD_SIM_SEED_BASE=$P_SBASE"; do
  isint "${v#*=}" || die "${v%%=*} '${v#*=}' is not a non-negative integer"
done
case "$P_GEV" in ''|*[!0-9.]*|*.*.*|.*|*.) die "energy '$P_GEV' is not a number in GeV" ;; esac
[ "$P_FIRST" -le "$P_LAST" ] || die "first batch $P_FIRST is after last batch $P_LAST"
if [ $((P_DBASE + P_FIRST)) -le 1 ] && [ $((P_DBASE + P_LAST)) -ge 1 ]; then
  die "driver seed base $P_DBASE gives batch $((1 - P_DBASE)) the driver seed 1, the seed of the legacy sample"
fi
[ -n "${PIXELAV_RUN_DIR:-}" ] || die "PIXELAV_RUN_DIR is not set (the PIXELAV run directory of the sensor)"
[ -d "$PIXELAV_RUN_DIR" ] || die "PIXELAV_RUN_DIR=$PIXELAV_RUN_DIR is not a directory"
P_RUN="$(cd "$PIXELAV_RUN_DIR" && pwd -P)"
absdir() { if [ -d "$1" ]; then (cd "$1" && pwd -P); else case "$1" in /*) echo "${1%/}" ;; *) echo "$PWD/${1%/}" ;; esac; fi; }
PROD_WORK="$(absdir "$6")"; PROD_DEST="$(absdir "$7")"
case "$PROD_DEST/" in "$PROD_WORK/"*) die "dest dir $PROD_DEST is the work dir $PROD_WORK or inside it" ;; esac
case "$PROD_WORK/" in "$PROD_DEST/"*) die "work dir $PROD_WORK is inside the dest dir $PROD_DEST" ;; esac
# readonly: the environment script sourced below cannot change the paths this script removes under
readonly PROD_HOME P_PART P_GEV P_FIRST P_LAST P_DBASE P_EVENTS P_NPROC P_NPARTS P_SBASE P_RUN PROD_WORK PROD_DEST DRY

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[prod] $(utc) $*"; }
# run <log|-> <command...>: the command, its output appended to the log; printed in a dry run
run() { local log="$1"; shift
  if [ -n "$DRY" ]; then printf '+'; printf ' %q' "$@"; [ "$log" = - ] || printf ' >> %q 2>&1' "$log"; echo; return 0; fi
  if [ "$log" = - ]; then "$@"; else "$@" >> "$log" 2>&1; fi; }
rmwork() {  # removes one batch work dir, never anything outside <work>
  case "$1" in "$PROD_WORK"/b[0-9]*) run - rm -rf -- "$1" ;; *) die "refusing to remove $1" ;; esac; }
if ! command -v ddsim > /dev/null 2>&1; then
  if [ -n "$DRY" ]; then echo "+ unset KEY4HEP_STACK; source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-02-01"
  else unset KEY4HEP_STACK; source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-02-01 > /dev/null 2>&1 \
    || die "cannot source the Key4hep stack"; fi
fi
export CALOMAPS_HOME="$PROD_HOME"
fail() {  # fail <stage> <log>: FAILED holds the stage and the log tail
  say "b$b $1 FAILED"
  if [ -z "$DRY" ]; then { echo "stage: $1"; tail -20 "$2" 2>/dev/null; } > "$D/FAILED"; fi
  NFAIL=$((NFAIL + 1)); }
size() { wc -c < "$1" | tr -d ' '; }

NDONE=0; NSKIP=0; NFAIL=0
for b in $(seq "$P_FIRST" "$P_LAST"); do
  B="$PROD_WORK/b$b"; D="$PROD_DEST/b$b"
  if [ -f "$D/DONE" ]; then say "b$b skipped ($D/DONE exists)"; NSKIP=$((NSKIP + 1)); continue; fi
  SSEED=$((P_SBASE + 1000 * b)); DSEED=$((P_DBASE + b))
  say "b$b inputs start, sim seed $SSEED, driver seed $DSEED"
  rmwork "$B"; run - rm -f "$D/FAILED"; run - mkdir -p "$B/out" "$B/data" "$D"
  run "$B/inputs.log" env CALOMAPS_SI_THICKNESS_UM=100 CALOMAPS_PIXELAV_ROUTE=deposits \
    CALOMAPS_GUN_PARTICLE="$P_PART" CALOMAPS_GUN_ENERGY_GEV="$P_GEV" CALOMAPS_N_EVENTS="$P_EVENTS" \
    CALOMAPS_SEED="$SSEED" CALOMAPS_JOB_PARTS="$P_NPARTS" PIXELAV_RUN_DIR="$P_RUN" \
    CALOMAPS_OUTPUT_DIR="$B/out" CALOMAPS_DATA_BASE="$B/data" \
    bash "$PROD_HOME/sim/make_pixelav_inputs.sh" || { fail inputs "$B/inputs.log"; continue; }
  if [ -n "$DRY" ]; then P="$B/out/pixelav_jobs_<tag>"
  else
    set -- "$B"/out/pixelav_jobs_*.jobs.npz
    [ $# -eq 1 ] && [ -f "$1" ] || { echo "not one jobs.npz in $B/out: $*" >> "$B/inputs.log"; fail inputs "$B/inputs.log"; continue; }
    P="${1%.jobs.npz}"
  fi
  say "b$b driver start"
  run "$B/run.log" bash "$PROD_HOME/analysis/pixelav/run_deposit_jobs.sh" "$P" "$P_RUN" "$P_NPROC" "$DSEED" \
    || { fail driver "$B/run.log"; continue; }
  say "b$b charge maps start"
  run "$B/run.log" "$PY" "$PROD_HOME/analysis/pixelav/charge_maps.py" "$P" || { fail charges "$B/run.log"; continue; }
  say "b$b copy start"
  ok=1
  run - cp "$P.jobs.npz" "$P.summary.txt" "$P.run.json" "$P.charges.npz" "$B/inputs.log" "$B/run.log" "$D/" || ok=0
  if [ -n "$DRY" ]; then OUTS="$P.part000.out ..."; else OUTS="$(ls "$P".part[0-9][0-9][0-9].out "$P".out 2>/dev/null)"; fi
  NOUT=0
  for f in $OUTS; do
    NOUT=$((NOUT + 1))
    if [ -n "$DRY" ]; then echo "+ gzip -c $f > $D/${f##*/}.gz"; else gzip -c "$f" > "$D/${f##*/}.gz" || ok=0; fi
  done
  if [ -z "$DRY" ]; then
    [ "$NOUT" -ge 1 ] && [ "$(ls "$D" | wc -l)" -eq $((6 + NOUT)) ] || ok=0
    for f in "$P.jobs.npz" "$P.charges.npz"; do [ "$(size "$f")" = "$(size "$D/${f##*/}")" ] || ok=0; done
    for f in "$D"/*.out.gz; do gzip -t "$f" 2> /dev/null || ok=0; done
  fi
  [ "$ok" = 1 ] || { echo "copy to $D not verified" >> "$B/run.log"; fail copy "$B/run.log"; continue; }
  run - touch "$D/DONE"
  [ -n "${PROD_KEEP_WORK:-}" ] || rmwork "$B"
  NDONE=$((NDONE + 1)); say "b$b done"
done
say "batches done $NDONE, skipped $NSKIP, failed $NFAIL"
[ "$NFAIL" -eq 0 ] || exit 1
run - touch "$PROD_DEST/ALL_DONE"
