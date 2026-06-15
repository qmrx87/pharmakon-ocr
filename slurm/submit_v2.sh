#!/usr/bin/env bash
# =============================================================================
# slurm/submit_v2.sh — submit the V2 challenger DAG on Narval (parallel jobs).
#
#   DAG (v2, divided + parallel):
#     ┌─ 02a' nano vignette cropper (data2)        [parallel — shared front-end]
#     └─ 12  build VLM dataset (annotations -> JSON, docTR values)
#              └─afterok─> 13 fine-tune Donut (v2a)
#                            └─afterok─> 14 compare v1 / vlm / fullpage
#   v2b (full-page docTR + PARSeq) needs NO training — stage 14 evaluates it
#   from pretrained weights directly. Stage 14 also picks up the v1 Stage B
#   checkpoint automatically when one exists, so the report covers everything
#   trained so far.
#
# PREREQS (login node, once):
#   bash scripts/fetch_pretrained_v2.sh     # Donut + docTR + rfdetr-nano caches
#   pip install --no-index -e .[ml,v2]      # (setup_narval.sh covers [ml])
#
# USAGE
#   export VIGNOCR_ACCOUNT=def-khenni VIGNOCR_PI=khenni
#   bash slurm/submit_v2.sh                 # full v2 DAG
#   bash slurm/submit_v2.sh --skip-cropper  # don't (re)train the nano cropper
#   bash slurm/submit_v2.sh --test-only     # validate directives, submit nothing
# =============================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

TEST_ONLY=0
SKIP_CROPPER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-only)    TEST_ONLY=1 ;;
    --skip-cropper) SKIP_CROPPER=1 ;;
    -h|--help)      sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) vdie "unknown arg: $1" ;;
  esac
  shift
done

vignocr_require_env
vignocr_paths
export VIGNOCR_DATA_ACTIVE="${VIGNOCR_DATA_ACTIVE:-real}"
vlog "V2 DAG | account=$VIGNOCR_ACCOUNT skip_cropper=$SKIP_CROPPER test_only=$TEST_ONLY"

for d in 02a_train_vignette 12_build_vlm_dataset 13_train_vlm_donut 14_compare_variants; do
  mkdir -p "$VIGNOCR_LOGS_DIR/$d"
done

STAGES=(12_build_vlm_dataset.sbatch 13_train_vlm_donut.sbatch 14_compare_variants.sbatch)

if [[ "$TEST_ONLY" -eq 1 ]]; then
  command -v sbatch >/dev/null 2>&1 || vdie "sbatch not found (not on Narval?)"
  rc=0
  for s in "${STAGES[@]}" 02a_train_vignette.sbatch; do
    sbatch --test-only --account="$VIGNOCR_ACCOUNT" "$SCRIPT_DIR/$s" \
      && vlog "OK  $s" || { vwarn "FAILED --test-only: $s"; rc=1; }
  done
  exit "$rc"
fi
command -v sbatch >/dev/null 2>&1 || vdie "sbatch not found — run on a Narval login node"

submit() {  # submit <script> [dep] [extra --export pairs]
  local script="$SCRIPT_DIR/$1" dep="${2:-}" extra="${3:-}"
  local dep_args=()
  [[ -n "$dep" ]] && dep_args=(--dependency="afterok:$dep")
  local export_list="ALL${extra:+,$extra}"
  for v in VIGNOCR_ACCOUNT VIGNOCR_PI VIGNOCR_REPO_ROOT VIGNOCR_SCRATCH_DIR \
           VIGNOCR_LOGS_DIR VIGNOCR_PRETRAINED_DIR VIGNOCR_VENV_DIR \
           VIGNOCR_RUNS_DIR VIGNOCR_SCRATCH VIGNOCR_DATA_ROOT VIGNOCR_DATA_ACTIVE \
           VIGNOCR_VLM_DATASET_DIR VIGNOCR_VLM_MODEL_DIR; do
    [[ -n "${!v:-}" ]] && export_list="$export_list,$v=${!v}"
  done
  sbatch --parsable --account="$VIGNOCR_ACCOUNT" --chdir="$VIGNOCR_REPO_ROOT" \
         --export="$export_list" "${dep_args[@]}" "$script" | grep -oE '^[0-9]+'
}

# Parallel branch 1: the nano vignette cropper (reuses the Stage A job with the
# nano config — model.name + variant-aware pretrained cache do the rest).
if [[ "$SKIP_CROPPER" -eq 0 ]]; then
  JID_CROP="$(submit 02a_train_vignette.sbatch "" "VIGNOCR_DETECTION_CONFIG=detection/rfdetr_nano_vignette")"
  vlog "submitted 02a' nano cropper  job=$JID_CROP  (parallel)"
fi

# Parallel branch 2: dataset -> Donut -> compare.
JID_12="$(submit 12_build_vlm_dataset.sbatch)"
vlog "submitted 12 build-vlm-dataset job=$JID_12"
JID_13="$(submit 13_train_vlm_donut.sbatch "$JID_12")"
vlog "submitted 13 train-vlm-donut   job=$JID_13  afterok=$JID_12"
JID_14="$(submit 14_compare_variants.sbatch "$JID_13")"
vlog "submitted 14 compare-variants  job=$JID_14  afterok=$JID_13"

vlog "=================== V2 DAG submitted ==================="
vlog "monitor:  squeue -u \$USER -o '%.10i %.28j %.10T %.20E'"
vlog "report:   logs/slurm/14_compare_variants/<jobid>/compare_report.md"
vlog "========================================================="
echo "$JID_14"
