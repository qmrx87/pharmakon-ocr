#!/usr/bin/env bash
# =============================================================================
# slurm/submit_all.sh  —  submit the full VignOCR training/eval DAG on Narval,
# chained with --dependency=afterok so each stage runs only if the prior one
# SUCCEEDED. Echoes every job id and the resulting dependency graph.
#
#   DAG (3-stage):
#         01 validate
#           ├─afterok─> 02a Stage A — train vignette detector (data2/)   [parallel]
#           └─afterok─> 02  Stage B — train field detector (data/)        [parallel]
#                         └─afterok─> 03 eval + ONNX export (Stage B)
#                                       └─afterok─> 04 Stage C bootstrap — PaddleOCR auto-label
#                                                     ⏹  HUMAN REVIEW in Roboflow (DAG stops here)
#                                                     └─manual─> 05 Stage C fine-tune (sbatch)
#                                                                   └─manual─> 06 pipeline benchmark
#
# The account is injected from $VIGNOCR_ACCOUNT on EVERY submission (overriding
# the `--account=def-<PI>` placeholder baked into each .sbatch file). The PI is
# used by lib.sh to resolve ~/projects/def-$VIGNOCR_PI.
#
# ---------------------------------------------------------------------------
# USAGE
#   export VIGNOCR_ACCOUNT=def-<PI>      # e.g. def-smith   (required)
#   export VIGNOCR_PI=<PI>               # e.g. smith       (required)
#   bash slurm/submit_all.sh             # submit the whole DAG
#   bash slurm/submit_all.sh --ddp       # use the multi-GPU (02b) training stage
#   bash slurm/submit_all.sh --test-only # validate every job's directives, submit nothing
#   bash slurm/submit_all.sh --from 03   # start the DAG at stage 03 (skip 01-02)
#
# RUNNING A SINGLE STAGE STANDALONE (no DAG): each .sbatch is self-contained.
# Submit one directly and it discovers its inputs from the latest run dirs:
#   sbatch --account=$VIGNOCR_ACCOUNT slurm/03_eval_export_detection.sbatch
#   # pin an explicit input instead of "latest":
#   sbatch --account=$VIGNOCR_ACCOUNT \
#          --export=ALL,VIGNOCR_DET_CKPT=/scratch/.../best.ckpt \
#          slurm/03_eval_export_detection.sbatch
#
# RESUMING an interrupted training stage in place:
#   sbatch --account=$VIGNOCR_ACCOUNT \
#          --export=ALL,VIGNOCR_RESUME_RUN_DIR=$HOME/scratch/vignocr/runs/02_train_detection/latest \
#          slurm/02_train_detection.sbatch
# =============================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"   # gives us vlog/vwarn/vdie + env validation helpers

# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
TEST_ONLY=0
USE_DDP=0
START_FROM="01"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-only) TEST_ONLY=1 ;;
    --ddp)       USE_DDP=1 ;;
    --from)      START_FROM="${2:?--from needs a stage number, e.g. --from 03}"; shift ;;
    --from=*)    START_FROM="${1#*=}" ;;
    -h|--help)   sed -n '2,52p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) vdie "unknown arg: $1 (try --test-only, --ddp, --from <NN>, or --help)" ;;
  esac
  shift
done

vignocr_require_env
# Resolve the storage tiers so the summary can show $VIGNOCR_RUNS_DIR. This is
# cheap and login-node-safe (it only needs the repo root + makes scratch dirs).
vignocr_paths
vlog "account=$VIGNOCR_ACCOUNT  PI=$VIGNOCR_PI  ddp=$USE_DDP  test_only=$TEST_ONLY  from=$START_FROM"
vlog "logs root: $VIGNOCR_LOGS_DIR  (per-stage/per-jobid)"

# Pre-flight gate: detection training will fail offline if the COCO pretrained
# weights aren't cached. Warn loudly here and offer the one-liner to fix it.
PRETRAIN_CACHED="$(vignocr_pretrained_weights detection/rfdetr_medium)"
if [[ -z "$PRETRAIN_CACHED" && -d "$SCRIPT_DIR/.." ]]; then
  vwarn "No cached RF-DETR pretrained weights at \$VIGNOCR_PRETRAINED_DIR ($VIGNOCR_PRETRAINED_DIR)."
  vwarn "Compute nodes have no internet — training will refuse to start. Run this NOW on the login node:"
  vwarn "    bash $(realpath --relative-to="$PWD" "$SCRIPT_DIR/..")/scripts/fetch_pretrained.sh"
  vwarn "Then re-run submit_all.sh. (To bypass for a login-node smoke test, export VIGNOCR_ALLOW_ONLINE_PRETRAIN=1.)"
fi

# Pre-create the centralized log directories so the #SBATCH --output paths
# resolve at submit time. We use `mkdir -p` for each STAGE name; SLURM creates
# the per-jobid subdir lazily when it opens the .out/.err.
for entry in \
    "01_validate" \
    "02_train_detection" \
    "02a_train_vignette" \
    "03_eval_export_detection" \
    "04_autolabel_ocr" \
    "04_train_ocr" \
    "05_finetune_ocr" \
    "05_eval_ocr" \
    "06_pipeline_benchmark"; do
  mkdir -p "$VIGNOCR_LOGS_DIR/$entry"
done

# The detection training stage is swappable: single-A100 (02) or DDP (02b).
DET_TRAIN_SCRIPT="02_train_detection.sbatch"
[[ "$USE_DDP" -eq 1 ]] && DET_TRAIN_SCRIPT="02b_train_detection_ddp.sbatch"

# Main chain (afterok). Stage 02a Vignette runs in PARALLEL alongside 02 — it
# is submitted as a sibling branch off 01 below (not in this serial list).
# After 04 the DAG STOPS so a human can review the auto-labelled crops in
# Roboflow. Stage 05 (finetune) and 06 (benchmark) are submitted manually after
# review by re-invoking submit_all.sh with `--from 05`.
STAGES=(
  "01:01_validate_data.sbatch"
  "02:$DET_TRAIN_SCRIPT"
  "03:03_eval_export_detection.sbatch"
  "04:04_autolabel_ocr.sbatch"
  "05:05_finetune_ocr.sbatch"
  "06:06_pipeline_benchmark.sbatch"
)
# The stage that gates on a HUMAN review step. The main loop stops here unless
# --from explicitly resumes after the gate.
HUMAN_GATE_STAGE="05"
# Parallel side-branch (runs concurrently with 02): vignette detector training.
PARALLEL_BRANCHES=("02a:02a_train_vignette.sbatch")

# --------------------------------------------------------------------------- #
# --test-only: validate each job's #SBATCH directives WITHOUT submitting. This
# is the Phase-7 exit gate. `sbatch --test-only` parses directives + checks the
# partition/account against the live scheduler but never queues the job.
# --------------------------------------------------------------------------- #
if [[ "$TEST_ONLY" -eq 1 ]]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    vwarn "sbatch not found (not on Narval?). Falling back to a static directive lint."
    rc=0
    for entry in "${STAGES[@]}" "${PARALLEL_BRANCHES[@]}"; do
      script="$SCRIPT_DIR/${entry#*:}"
      [[ -f "$script" ]] || { vwarn "missing: $script"; rc=1; continue; }
      # Lint: must start with a shebang and carry the required directives.
      head -n1 "$script" | grep -q '^#!' || { vwarn "$script: missing shebang"; rc=1; }
      for d in '#SBATCH --job-name=' '#SBATCH --account=' '#SBATCH --time=' '#SBATCH --mem='; do
        grep -q -- "$d" "$script" || { vwarn "$script: missing directive '$d'"; rc=1; }
      done
      bash -n "$script" || { vwarn "$script: bash syntax error"; rc=1; }
      vlog "linted $script"
    done
    [[ "$rc" -eq 0 ]] && vlog "static lint PASSED for all stages" || vdie "static lint FAILED"
    exit "$rc"
  fi

  vlog "validating every stage with 'sbatch --test-only' (account=$VIGNOCR_ACCOUNT)"
  rc=0
  for entry in "${STAGES[@]}" "${PARALLEL_BRANCHES[@]}"; do
    script="$SCRIPT_DIR/${entry#*:}"
    if sbatch --test-only --account="$VIGNOCR_ACCOUNT" "$script"; then
      vlog "OK  ${entry#*:}"
    else
      vwarn "FAILED --test-only: ${entry#*:}"; rc=1
    fi
  done
  [[ "$rc" -eq 0 ]] && vlog "ALL stages accepted by the scheduler (--test-only)" \
                    || vdie "one or more stages rejected by --test-only"
  exit "$rc"
fi

# --------------------------------------------------------------------------- #
# Real submission. We need sbatch here.
# --------------------------------------------------------------------------- #
command -v sbatch >/dev/null 2>&1 || vdie "sbatch not found — submit_all.sh must run on a Narval login node"

# Parse the numeric job id from `sbatch`'s "Submitted batch job 12345" line.
parse_jobid() { grep -oE '[0-9]+' | tail -n1; }

# submit_stage <script> [dep_jobid] -> echoes the new job id.
# When dep_jobid is set, the job is queued with --dependency=afterok:<dep>.
# --chdir pins the job's CWD to the REPO ROOT — this is what makes the relative
# `#SBATCH --output=logs/slurm/...` paths resolve consistently regardless of
# where the user invoked submit_all.sh from.
submit_stage() {
  local script="$SCRIPT_DIR/$1"; local dep="${2:-}"
  local dep_args=()
  [[ -n "$dep" ]] && dep_args=(--dependency="afterok:$dep")
  local out jobid
  # --account on the CLI overrides the placeholder in the file. --export=ALL
  # forwards our VIGNOCR_* env (account/PI/module/path overrides) into the job.
  # --chdir anchors %x/%j --output paths to the repo root.
  out="$(sbatch --parsable \
        --account="$VIGNOCR_ACCOUNT" \
        --chdir="$VIGNOCR_REPO_ROOT" \
        --export=ALL \
        "${dep_args[@]}" \
        "$script")"
  # --parsable prints just "<jobid>" (or "<jobid>;<cluster>"); be defensive.
  jobid="$(printf '%s' "$out" | parse_jobid)"
  [[ -n "$jobid" ]] || vdie "failed to parse job id from sbatch output: '$out' (script=$script)"
  echo "$jobid"
}

declare -A JOBID         # stage_number -> job id
PREV_JOBID=""            # the upstream dependency for the next stage
SUBMITTED=()

vlog "submitting DAG (afterok chain), starting at stage $START_FROM"
GATE_HIT=0
for entry in "${STAGES[@]}"; do
  num="${entry%%:*}"; script="${entry#*:}"
  # Skip stages before --from. The first submitted stage carries NO dependency
  # (so a partial DAG can start cleanly); later stages chain on the previous.
  if [[ "$num" < "$START_FROM" ]]; then
    vlog "skip stage $num ($script) — before --from $START_FROM"
    continue
  fi
  # Stop at the human-review gate unless --from explicitly resumes past it.
  if [[ "$num" == "$HUMAN_GATE_STAGE" && "$START_FROM" < "$HUMAN_GATE_STAGE" ]]; then
    vlog "STOP at stage $HUMAN_GATE_STAGE — human review gate. Resume with: bash submit_all.sh --from $HUMAN_GATE_STAGE"
    GATE_HIT=1
    break
  fi
  jid="$(submit_stage "$script" "$PREV_JOBID")"
  JOBID["$num"]="$jid"
  SUBMITTED+=("$num:$script:$jid:${PREV_JOBID:-<none>}")
  vlog "submitted stage $num: $script  job=$jid  afterok=${PREV_JOBID:-<none>}"
  PREV_JOBID="$jid"

  # After the gating stage 01, fan out the parallel side-branches (e.g. 02a
  # vignette training) so they run alongside the serial chain.
  if [[ "$num" == "01" ]]; then
    for sib in "${PARALLEL_BRANCHES[@]}"; do
      sib_num="${sib%%:*}"; sib_script="${sib#*:}"
      sib_jid="$(submit_stage "$sib_script" "$jid")"
      JOBID["$sib_num"]="$sib_jid"
      SUBMITTED+=("$sib_num:$sib_script:$sib_jid:$jid")
      vlog "submitted parallel  $sib_num: $sib_script  job=$sib_jid  afterok=$jid"
    done
  fi
done

# --------------------------------------------------------------------------- #
# Summary: print the DAG + a couple of handy monitoring commands.
# --------------------------------------------------------------------------- #
echo "" >&2
vlog "================= VignOCR DAG submitted ================="
ALL_IDS=()
for line in "${SUBMITTED[@]}"; do
  IFS=':' read -r num script jid dep <<< "$line"
  printf '   stage %s  job %-10s  <- afterok %s   (%s)\n' "$num" "$jid" "$dep" "$script" >&2
  ALL_IDS+=("$jid")
done
echo "" >&2
vlog "monitor:    squeue -u \$USER --states=all"
vlog "DAG view:   squeue -u \$USER -o '%.10i %.30j %.10T %.20E'   (E shows the afterok dependency)"
vlog "cancel all: scancel ${ALL_IDS[*]}"
vlog "run dirs:   $VIGNOCR_RUNS_DIR/<stage>/latest"
vlog "========================================================="

# Emit the final benchmark job id on stdout (handy for scripting/CI capture).
echo "${PREV_JOBID}"
