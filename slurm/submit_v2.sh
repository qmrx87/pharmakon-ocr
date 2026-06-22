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
vignocr_warn_if_stale   # refuse to submit a stale checkout (the recurring root cause)
export VIGNOCR_DATA_ACTIVE="${VIGNOCR_DATA_ACTIVE:-real}"
vlog "V2 DAG | account=$VIGNOCR_ACCOUNT skip_cropper=$SKIP_CROPPER test_only=$TEST_ONLY"

# ── Preflight: the scratch venv MUST already carry the v2 deps (compute nodes
#    are OFFLINE and cannot install). Catch a missing stack HERE, on the login
#    node, instead of after 4 jobs queue and die at import. Skip with
#    VIGNOCR_SKIP_PREFLIGHT=1. transformers+sentencepiece gate v2a (Donut);
#    doctr gates v2b + the dataset value backend — missing doctr only WARNS
#    (you can still run with VIGNOCR_VLM_VALUE_BACKEND=stub).
if [[ "${VIGNOCR_SKIP_PREFLIGHT:-0}" != "1" && -d "$VIGNOCR_VENV_DIR" ]]; then
  vlog "preflight: checking v2 deps in $VIGNOCR_VENV_DIR"
  vignocr_load_modules   # the venv's python needs its base `module load python` on PATH
  # shellcheck disable=SC1091
  source "$VIGNOCR_VENV_DIR/bin/activate"
  # REAL imports, not importlib.find_spec: a module can be *locatable* yet fail
  # to import (e.g. doctr's h5py 'undefined symbol' ABI break — find_spec passed
  # it, then stage 12/14 died). Importing here gives the login node the truth.
  # transformers+sentencepiece gate v2a (Donut) -> HARD. doctr gates only v2b +
  # the *optional* doctr value backend (the default is trocr) -> SOFT (warn).
  python - <<'PY' || vdie "v2a deps (transformers/sentencepiece) are missing or broken in the venv — run 'bash scripts/setup_narval.sh' on the LOGIN node first (compute nodes have no internet). Override with VIGNOCR_SKIP_PREFLIGHT=1."
import importlib, sys
def _try(mod):
    try:
        importlib.import_module(mod); return None
    except Exception as e:  # ImportError OR a native-lib ABI error at import
        return f"{type(e).__name__}: {e}"
hard_fail = []
for m in ("transformers", "sentencepiece"):   # v2a Donut — required
    err = _try(m)
    print(f"    {'OK     ' if err is None else 'BROKEN '} {m}" + (f"  ({err})" if err else ""))
    if err: hard_fail.append(m)
err = _try("doctr")                            # v2b + optional doctr value backend
print(f"    {'OK     ' if err is None else 'BROKEN '} doctr" + (f"  ({err})" if err else ""))
if err:
    print("  WARN: doctr unimportable -> v2b (full-page) will be SKIPPED by the")
    print("        comparison, and the VLM dataset MUST use the trocr backend")
    print("        (the default). v2a (Donut) + v1 are UNAFFECTED.")
sys.exit(1 if hard_fail else 0)
PY
fi

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
