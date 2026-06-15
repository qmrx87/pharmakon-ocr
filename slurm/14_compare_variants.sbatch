#!/usr/bin/env bash
# =============================================================================
# slurm/14_compare_variants.sbatch — the DECIDER: v1 vs v2a (Donut) vs
# v2b (full-page docTR/PARSeq) on the SAME held-out test split.
#
# Produces experiments/v2_compare/compare_report.{json,md}: per-field exact
# match + CER + coverage + latency p50/p95 per variant. Variants whose weights
# are absent are skipped gracefully, so this stage can run after ANY subset of
# the training jobs has landed and be re-run as the rest finish.
# =============================================================================
#SBATCH --job-name=vignocr-14-compare
#SBATCH --account=def-<PI>
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/14_compare_variants/%x-%j.out
#SBATCH --error=logs/slurm/14_compare_variants/%x-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"
vignocr_preamble "14_compare_variants"

# Dataset (reference values + test images) from stage 12.
if [[ -z "${VIGNOCR_VLM_DATASET_DIR:-}" ]]; then
  BUILD_RUN="$(vignocr_latest_run 12_build_vlm_dataset)"
  [[ -n "$BUILD_RUN" && -f "$BUILD_RUN/VLM_DATASET_DIR" ]] \
    || vdie "VIGNOCR_VLM_DATASET_DIR unset and no stage-12 run found"
  VIGNOCR_VLM_DATASET_DIR="$(cat "$BUILD_RUN/VLM_DATASET_DIR")"
fi

# v2a checkpoint from stage 13 (skip the variant if not trained yet).
VLM_DIR="${VIGNOCR_VLM_MODEL_DIR:-}"
if [[ -z "$VLM_DIR" ]]; then
  TRAIN_RUN="$(vignocr_latest_run 13_train_vlm_donut)"
  [[ -n "$TRAIN_RUN" && -f "$TRAIN_RUN/BEST_CKPT" ]] && VLM_DIR="$(cat "$TRAIN_RUN/BEST_CKPT")"
fi

# v1 Stage B checkpoint (optional — include when trained).
V1_CKPT="${VIGNOCR_DET_CKPT:-}"
if [[ -z "$V1_CKPT" ]]; then
  V1_RUN="$(vignocr_latest_run 02_train_detection)"
  [[ -n "$V1_RUN" && -f "$V1_RUN/BEST_CKPT" ]] && V1_CKPT="$(cat "$V1_RUN/BEST_CKPT")"
fi

VARIANTS="fullpage"
[[ -n "$VLM_DIR" ]] && VARIANTS="vlm,$VARIANTS"
[[ -n "$V1_CKPT" ]] && VARIANTS="v1,$VARIANTS"
OUT_DIR="$VIGNOCR_RUN_DIR/v2_compare"

vlog "comparing variants [$VARIANTS] on $VIGNOCR_VLM_DATASET_DIR (test split)"
set +e
python -m vignocr.v2.compare \
    --dataset      "$VIGNOCR_VLM_DATASET_DIR" \
    --split        test \
    --variants     "$VARIANTS" \
    ${VLM_DIR:+--vlm-dir "$VLM_DIR"} \
    ${V1_CKPT:+--v1-detector "$V1_CKPT"} \
    --out          "$OUT_DIR" \
  2>&1 | tee "$VIGNOCR_RUN_DIR/compare_variants.log"
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "comparison failed (rc=$rc) — see $VIGNOCR_RUN_DIR/compare_variants.log"

# Copy the human-readable report into the repo-side logs for easy retrieval.
cp -f "$OUT_DIR/compare_report.md" "$VIGNOCR_JOB_LOG_DIR/" 2>/dev/null || true
touch "$VIGNOCR_RUN_DIR/COMPARE_OK"
vlog "stage 14 complete — report: $OUT_DIR/compare_report.md"
