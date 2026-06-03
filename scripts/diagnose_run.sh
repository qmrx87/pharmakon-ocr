#!/usr/bin/env bash
# =============================================================================
# scripts/diagnose_run.sh  —  surface the actual cause of a failed VignOCR job.
#
# WHY THIS EXISTS
#   The .err files emitted by SLURM only say "FATAL: training failed (rc=1) —
#   see <scratch>/train_*.log". The actual Python traceback lives in that file
#   on scratch. This script finds the failure, prints WHERE the run dir is,
#   and tails the last 200 lines of the Python log so the cause is visible.
#
# USAGE
#   bash scripts/diagnose_run.sh                 # auto-pick the most recent failed run
#   bash scripts/diagnose_run.sh <stage>         # auto-pick the latest run of a stage
#   bash scripts/diagnose_run.sh <stage> <jobid> # explicit
#
# Examples:
#   bash scripts/diagnose_run.sh                          # latest failure across all stages
#   bash scripts/diagnose_run.sh 02a_train_vignette       # latest Stage A
#   bash scripts/diagnose_run.sh 02_train_detection 62301870
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
# shellcheck source=../slurm/lib.sh
source "$REPO_ROOT/slurm/lib.sh"

vignocr_require_env >/dev/null 2>&1 || true   # only needed for paths; don't hard-fail
vignocr_paths

STAGE="${1:-}"
JOBID="${2:-}"

# --------------------------------------------------------------------------- #
# Resolve which run dir to inspect.
# --------------------------------------------------------------------------- #
RUN_DIR=""
LOG_DIR=""

if [[ -n "$STAGE" && -n "$JOBID" ]]; then
  LOG_DIR="$VIGNOCR_LOGS_DIR/$STAGE/$JOBID"
  [[ -f "$LOG_DIR/BACKPOINTERS.txt" ]] && RUN_DIR="$(awk '/^run_dir:/{print $2}' "$LOG_DIR/BACKPOINTERS.txt")"
elif [[ -n "$STAGE" ]]; then
  # latest run of this stage
  RUN_DIR="$(vignocr_latest_run "$STAGE")"
  LOG_DIR="$(ls -1dt "$VIGNOCR_LOGS_DIR/$STAGE"/*/ 2>/dev/null | head -n1 || true)"
  LOG_DIR="${LOG_DIR%/}"
else
  # Globally latest failed run across all stages: look for FAILED markers under logs/slurm.
  FAILED_DIR="$(ls -1dt "$VIGNOCR_LOGS_DIR"/*/*/FAILED 2>/dev/null | head -n1 || true)"
  if [[ -n "$FAILED_DIR" ]]; then
    LOG_DIR="$(dirname "$FAILED_DIR")"
    [[ -f "$LOG_DIR/BACKPOINTERS.txt" ]] && RUN_DIR="$(awk '/^run_dir:/{print $2}' "$LOG_DIR/BACKPOINTERS.txt")"
  fi
  if [[ -z "$LOG_DIR" ]]; then
    # No FAILED marker yet (legacy pre-trap runs). Find the most recent .err in the repo root.
    ERR="$(ls -1t "$VIGNOCR_REPO_ROOT"/vignocr-*-*.err 2>/dev/null | head -n1 || true)"
    [[ -n "$ERR" ]] && vlog "no centralized FAILED found; tailing legacy .err: $ERR"
    if [[ -n "$ERR" ]]; then
      RUN_DIR="$(grep -oE '/[^ ]+/runs/[^ ]+' "$ERR" 2>/dev/null | tail -n1 || true)"
    fi
  fi
fi

# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
echo "================================================================"
echo " VignOCR run diagnosis"
echo "================================================================"
echo "  stage    : ${STAGE:-<auto>}"
echo "  job id   : ${JOBID:-<auto>}"
echo "  log dir  : ${LOG_DIR:-<none found>}"
echo "  run dir  : ${RUN_DIR:-<none found>}"
echo "  logs root: $VIGNOCR_LOGS_DIR"
echo "  runs root: $VIGNOCR_RUNS_DIR"
echo "================================================================"

if [[ -n "$LOG_DIR" && -f "$LOG_DIR/BACKPOINTERS.txt" ]]; then
  echo
  echo "--- BACKPOINTERS.txt ---"
  cat "$LOG_DIR/BACKPOINTERS.txt"
fi

# Locate the most useful Python log to tail.
PRIMARY=""
if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
  for f in train_detection.log train_vignette.log train_detection_ddp.log \
           train_ocr.log autolabel.log eval_export_detection.log eval_ocr.log \
           pipeline_benchmark.log finetune.log validate.log; do
    if [[ -f "$RUN_DIR/$f" ]]; then PRIMARY="$RUN_DIR/$f"; break; fi
  done
  if [[ -z "$PRIMARY" ]]; then
    PRIMARY="$(ls -1t "$RUN_DIR"/*.log 2>/dev/null | head -n1 || true)"
  fi
fi
# Otherwise look inside the central log dir (post-mortem trap will have copied it).
if [[ -z "$PRIMARY" && -n "$LOG_DIR" && -d "$LOG_DIR" ]]; then
  PRIMARY="$(ls -1t "$LOG_DIR"/*.log 2>/dev/null | head -n1 || true)"
fi

if [[ -z "$PRIMARY" || ! -f "$PRIMARY" ]]; then
  echo
  echo "No Python log found."
  echo "Inspect the SLURM .err / .out yourself with:"
  echo "    ls -1t logs/slurm/<stage>/*/ ; cat logs/slurm/<stage>/<jobid>/*.err"
  exit 1
fi

echo
echo "================================================================"
echo " Last 200 lines of: $PRIMARY"
echo "================================================================"
tail -n 200 "$PRIMARY"

echo
echo "================================================================"
echo " To resume this run after fixing the cause:"
echo "   export VIGNOCR_RESUME_RUN_DIR=$RUN_DIR"
echo "   sbatch --account=\$VIGNOCR_ACCOUNT --export=ALL slurm/<the sbatch you ran>"
echo "================================================================"
