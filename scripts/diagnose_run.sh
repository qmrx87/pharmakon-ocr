#!/usr/bin/env bash
# =============================================================================
# scripts/diagnose_run.sh  —  surface the actual cause of a failed VignOCR job.
#
# WHY THIS EXISTS
#   When a job in the afterok DAG fails, the .err of the FAILING stage holds the
#   real cause — but downstream stages are then CANCELLED by SLURM (and never
#   produce any log at all). Operators tend to look at the LAST stage they tried
#   to run and conclude "no log = mystery". This script:
#
#     • Lists every stage's log dir and the most-recent .out/.err in it.
#     • Auto-finds the actual failure (FAILED marker, then non-empty .err, then
#       sacct's State=FAILED) and dumps it.
#     • Detects dep-cancelled jobs and tells you to look at the upstream stage.
#     • Tails Python logs from scratch AND the central log dir.
#
# USAGE
#   bash scripts/diagnose_run.sh                 # show ALL recent activity, auto-pick the failure
#   bash scripts/diagnose_run.sh <stage>         # latest run of this stage
#   bash scripts/diagnose_run.sh <stage> <jobid> # explicit (jobid required when stage repeats)
#   bash scripts/diagnose_run.sh --jobid <id>    # by job id alone (queries sacct)
#   bash scripts/diagnose_run.sh --all           # dump everything (verbose)
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
# shellcheck source=../slurm/lib.sh
source "$REPO_ROOT/slurm/lib.sh"

vignocr_require_env >/dev/null 2>&1 || true
vignocr_paths

STAGE=""
JOBID=""
SHOW_ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)   SHOW_ALL=1 ;;
    --jobid) JOBID="${2:?--jobid needs a value}"; shift ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)       if [[ -z "$STAGE" ]]; then STAGE="$1"
             elif [[ -z "$JOBID" ]]; then JOBID="$1"
             else vdie "unexpected arg: $1"
             fi ;;
  esac
  shift
done

# --------------------------------------------------------------------------- #
# 0. Cheap overview — what's in every stage's central log dir.
# --------------------------------------------------------------------------- #
echo "================================================================"
echo " VignOCR log inventory: $VIGNOCR_LOGS_DIR"
echo "================================================================"
if [[ -d "$VIGNOCR_LOGS_DIR" ]]; then
  for sd in "$VIGNOCR_LOGS_DIR"/*/; do
    [[ -d "$sd" ]] || continue
    sd_name="$(basename "$sd")"
    # Count files (.out/.err written by SLURM directly + per-jobid subdirs from
    # the postmortem trap).
    n_slurm="$(ls -1 "$sd"*.out "$sd"*.err 2>/dev/null | wc -l | tr -d ' ')"
    n_jobs="$(ls -1d "$sd"*/ 2>/dev/null | wc -l | tr -d ' ')"
    printf '  %-30s  slurm files=%-3s  per-job dirs=%s\n' "$sd_name" "$n_slurm" "$n_jobs"
    # Show the most recent .out + .err names (truncated, no contents)
    for f in $(ls -1t "$sd"*.out "$sd"*.err 2>/dev/null | head -n 4); do
      printf '       %s  (%s bytes)\n' "$(basename "$f")" "$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo ?)"
    done
  done
else
  echo "  (logs root does not exist yet — submit a job first)"
fi
echo

# --------------------------------------------------------------------------- #
# 1. sacct: get the recent VignOCR job history so we can talk about state.
# --------------------------------------------------------------------------- #
have_sacct=0
if command -v sacct >/dev/null 2>&1; then have_sacct=1; fi
if [[ "$have_sacct" -eq 1 ]]; then
  echo "================================================================"
  echo " Recent VignOCR jobs (sacct, last 24h)"
  echo "================================================================"
  sacct --user="$USER" --starttime=now-24hours \
        --name=vignocr-01-validate,vignocr-02-train-det,vignocr-02a-train-vignette,vignocr-02b-train-ddp,vignocr-03-eval-export,vignocr-04-autolabel,vignocr-04-train-ocr,vignocr-05-eval-ocr,vignocr-05-finetune-ocr,vignocr-06-bench \
        --format=JobID,JobName%30,State,ExitCode,Start,Elapsed,Reason%40 \
        --noheader 2>/dev/null | sed 's/^/  /' || echo "  (sacct returned no rows)"
  echo
fi

# --------------------------------------------------------------------------- #
# 2. Pick a target.
#    Priority:
#      a) explicit (STAGE, JOBID) → that exact log dir
#      b) explicit STAGE only     → its latest job
#      c) explicit JOBID only     → find which stage via sacct
#      d) auto-pick: most recent FAILED marker > non-empty .err > newest .out
# --------------------------------------------------------------------------- #
LOG_DIR=""    # central dir for the chosen job
RUN_DIR=""    # scratch run dir for the chosen job
SLURM_OUT=""  # the SLURM .out path
SLURM_ERR=""  # the SLURM .err path
JOB_STATE=""  # from sacct, if available
JOB_REASON="" # from sacct, if available

if [[ -n "$JOBID" && -z "$STAGE" && "$have_sacct" -eq 1 ]]; then
  # Look up the job name and infer the stage from it.
  jn="$(sacct -j "$JOBID" -o JobName --noheader 2>/dev/null | head -n1 | xargs || true)"
  case "$jn" in
    vignocr-01-validate)         STAGE=01_validate ;;
    vignocr-02-train-det)        STAGE=02_train_detection ;;
    vignocr-02a-train-vignette)  STAGE=02a_train_vignette ;;
    vignocr-02b-train-ddp)       STAGE=02_train_detection ;;
    vignocr-03-eval-export)      STAGE=03_eval_export_detection ;;
    vignocr-04-autolabel)        STAGE=04_autolabel_ocr ;;
    vignocr-04-train-ocr)        STAGE=04_train_ocr ;;
    vignocr-05-eval-ocr)         STAGE=05_eval_ocr ;;
    vignocr-05-finetune-ocr)     STAGE=05_finetune_ocr ;;
    vignocr-06-bench)            STAGE=06_pipeline_benchmark ;;
  esac
fi

# Pick the SLURM .out/.err for the (STAGE, JOBID) pair, falling back as needed.
_match_slurm_files() {
  local stage="$1"; local jid="$2"; local sd="$VIGNOCR_LOGS_DIR/$stage"
  if [[ -n "$jid" ]]; then
    SLURM_OUT="$(ls -1 "$sd"/*-"$jid".out 2>/dev/null | head -n1 || true)"
    SLURM_ERR="$(ls -1 "$sd"/*-"$jid".err 2>/dev/null | head -n1 || true)"
  else
    SLURM_OUT="$(ls -1t "$sd"/*.out 2>/dev/null | head -n1 || true)"
    SLURM_ERR="$(ls -1t "$sd"/*.err 2>/dev/null | head -n1 || true)"
  fi
  LOG_DIR="$sd${jid:+/$jid}"
  [[ -f "$LOG_DIR/BACKPOINTERS.txt" ]] && RUN_DIR="$(awk '/^run_dir:/{print $2}' "$LOG_DIR/BACKPOINTERS.txt")" || true
}

if [[ -n "$STAGE" ]]; then
  _match_slurm_files "$STAGE" "$JOBID"
else
  # No stage and no jobid — auto-pick. Priority: FAILED marker, then non-empty .err.
  FAILED_DIR="$(ls -1dt "$VIGNOCR_LOGS_DIR"/*/*/FAILED 2>/dev/null | head -n1 || true)"
  if [[ -n "$FAILED_DIR" ]]; then
    LOG_DIR="$(dirname "$FAILED_DIR")"
    STAGE="$(basename "$(dirname "$LOG_DIR")")"
    JOBID="$(basename "$LOG_DIR")"
    _match_slurm_files "$STAGE" "$JOBID"
  fi
  if [[ -z "$STAGE" ]]; then
    # Find the newest non-empty .err across all stages.
    NEW_ERR="$(find "$VIGNOCR_LOGS_DIR" -maxdepth 2 -name '*.err' -type f -not -empty -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | awk '{print $2}' || true)"
    if [[ -n "$NEW_ERR" ]]; then
      SLURM_ERR="$NEW_ERR"
      STAGE="$(basename "$(dirname "$NEW_ERR")")"
      # Job id is the trailing digits before .err
      JOBID="$(echo "$NEW_ERR" | sed -E 's/.*-([0-9]+)\.err$/\1/')"
      _match_slurm_files "$STAGE" "$JOBID"
    fi
  fi
  if [[ -z "$STAGE" ]]; then
    # Last resort: newest .out anywhere (likely still running).
    NEW_OUT="$(find "$VIGNOCR_LOGS_DIR" -maxdepth 2 -name '*.out' -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | awk '{print $2}' || true)"
    if [[ -n "$NEW_OUT" ]]; then
      SLURM_OUT="$NEW_OUT"
      STAGE="$(basename "$(dirname "$NEW_OUT")")"
      JOBID="$(echo "$NEW_OUT" | sed -E 's/.*-([0-9]+)\.out$/\1/')"
      _match_slurm_files "$STAGE" "$JOBID"
    fi
  fi
fi

# Pull state + reason from sacct if we know the job id.
if [[ -n "$JOBID" && "$have_sacct" -eq 1 ]]; then
  JOB_STATE="$(sacct -j "$JOBID" --noheader -o State 2>/dev/null | head -n1 | xargs || true)"
  JOB_REASON="$(sacct -j "$JOBID" --noheader -o Reason%80 2>/dev/null | head -n1 | xargs || true)"
fi

# --------------------------------------------------------------------------- #
# 3. Report.
# --------------------------------------------------------------------------- #
echo "================================================================"
echo " Diagnosis target"
echo "================================================================"
printf '  stage     : %s\n' "${STAGE:-<not found>}"
printf '  job id    : %s\n' "${JOBID:-<not found>}"
printf '  sacct st. : %s\n' "${JOB_STATE:-<unknown>}"
printf '  reason    : %s\n' "${JOB_REASON:-<none>}"
printf '  log dir   : %s\n' "${LOG_DIR:-<not found>}"
printf '  run dir   : %s\n' "${RUN_DIR:-<not found>}"
printf '  slurm .out: %s\n' "${SLURM_OUT:-<not found>}"
printf '  slurm .err: %s\n' "${SLURM_ERR:-<not found>}"
echo "================================================================"

# DEP-CANCEL EARLY EXIT: jobs cancelled by afterok never produce a log.
case "$JOB_STATE" in
  *CANCELLED*|*DependencyNeverSatisfied*)
    echo
    echo "  ⚠  This job was CANCELLED — most likely because an upstream stage"
    echo "      in the afterok DAG failed. Look at the UPSTREAM stage instead."
    echo "      Quick scan of recent failures:"
    if [[ "$have_sacct" -eq 1 ]]; then
      sacct --user="$USER" --starttime=now-24hours \
            --state=FAILED,TIMEOUT,NODE_FAIL,OUT_OF_MEMORY \
            -o JobID,JobName%30,State,ExitCode,Start --noheader 2>/dev/null \
            | sed 's/^/        /' || true
    fi
    ;;
esac

# Show BACKPOINTERS if present (post-mortem trap dropped it).
if [[ -n "$LOG_DIR" && -f "$LOG_DIR/BACKPOINTERS.txt" ]]; then
  echo
  echo "--- $LOG_DIR/BACKPOINTERS.txt ---"
  cat "$LOG_DIR/BACKPOINTERS.txt"
fi

# Show SLURM .err (always, when present — that's where setup + module failures land).
if [[ -n "$SLURM_ERR" && -f "$SLURM_ERR" ]]; then
  echo
  echo "================================================================"
  echo " SLURM .err: $SLURM_ERR ($(stat -c%s "$SLURM_ERR" 2>/dev/null || stat -f%z "$SLURM_ERR" 2>/dev/null || echo ?) bytes)"
  echo "================================================================"
  if [[ -s "$SLURM_ERR" ]]; then
    tail -n 300 "$SLURM_ERR"
  else
    echo "  (empty — job may have been cancelled or never started)"
  fi
fi

# Show SLURM .out tail (useful when stderr is empty but stdout shows the trace).
if [[ -n "$SLURM_OUT" && -f "$SLURM_OUT" ]]; then
  echo
  echo "================================================================"
  echo " SLURM .out tail: $SLURM_OUT ($(stat -c%s "$SLURM_OUT" 2>/dev/null || stat -f%z "$SLURM_OUT" 2>/dev/null || echo ?) bytes)"
  echo "================================================================"
  if [[ -s "$SLURM_OUT" ]]; then
    tail -n 100 "$SLURM_OUT"
  else
    echo "  (empty)"
  fi
fi

# Show the most useful Python log from scratch OR central log dir.
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
if [[ -z "$PRIMARY" && -n "$LOG_DIR" && -d "$LOG_DIR" ]]; then
  PRIMARY="$(ls -1t "$LOG_DIR"/*.log 2>/dev/null | head -n1 || true)"
fi

if [[ -n "$PRIMARY" && -f "$PRIMARY" ]]; then
  echo
  echo "================================================================"
  echo " Python log: $PRIMARY"
  echo "================================================================"
  tail -n 200 "$PRIMARY"
fi

# Resume hint (only meaningful if we actually have a run dir).
if [[ -n "$RUN_DIR" ]]; then
  echo
  echo "================================================================"
  echo " To resume this run after fixing the cause:"
  echo "   export VIGNOCR_RESUME_RUN_DIR=$RUN_DIR"
  echo "   sbatch --account=\$VIGNOCR_ACCOUNT --chdir=$VIGNOCR_REPO_ROOT --export=ALL slurm/<the sbatch>"
  echo "================================================================"
fi

# Verbose / --all mode: dump every recent .err in every stage.
if [[ "$SHOW_ALL" -eq 1 ]]; then
  echo
  echo "================================================================"
  echo " --all: dumping every recent .err across all stages"
  echo "================================================================"
  for e in $(find "$VIGNOCR_LOGS_DIR" -maxdepth 2 -name '*.err' -type f 2>/dev/null | sort); do
    echo
    echo "--- $e ($(stat -c%s "$e" 2>/dev/null || stat -f%z "$e" 2>/dev/null || echo ?) bytes) ---"
    [[ -s "$e" ]] && tail -n 80 "$e" || echo "  (empty)"
  done
fi
