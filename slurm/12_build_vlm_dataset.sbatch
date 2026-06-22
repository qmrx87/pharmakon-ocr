#!/usr/bin/env bash
# =============================================================================
# slurm/12_build_vlm_dataset.sbatch — V2 stage: annotations -> VLM JSON dataset.
#
# Runs scripts/build_vlm_dataset.py over the Stage B `real` export: crops every
# annotated field box, reads its value (reviewed autolabel CSV first, else
# pretrained docTR/PARSeq), and writes the (image, gt_parse JSON) pairs Donut
# fine-tunes on, plus a low-confidence review queue.
#
# GPU: docTR recognition over ~5.8k crops — minutes on an A100; the small
# allocation keeps queue time short. OFFLINE-safe: docTR loads from
# $DOCTR_CACHE_DIR (prefetched by scripts/fetch_pretrained_v2.sh).
# =============================================================================
#SBATCH --job-name=vignocr-12-build-vlm
#SBATCH --account=def-<PI>
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/12_build_vlm_dataset/%x-%j.out
#SBATCH --error=logs/slurm/12_build_vlm_dataset/%x-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"
vignocr_preamble "12_build_vlm_dataset"

# Dataset lands on SCRATCH (images are copied next to metadata.jsonl).
VLM_DATASET_DIR="${VIGNOCR_VLM_DATASET_DIR:-$VIGNOCR_SCRATCH_DIR/datasets/vlm}"
export VIGNOCR_VLM_DATASET_DIR="$VLM_DATASET_DIR"
BACKEND="${VIGNOCR_VLM_VALUE_BACKEND:-trocr}"

vlog "building VLM dataset -> $VLM_DATASET_DIR (value backend: $BACKEND)"
set +e
python scripts/build_vlm_dataset.py \
    --config  v2/vlm_donut \
    --output  "$VLM_DATASET_DIR" \
    --backend "$BACKEND" \
  2>&1 | tee "$VIGNOCR_RUN_DIR/build_vlm_dataset.log"
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "VLM dataset build failed (rc=$rc) — see $VIGNOCR_RUN_DIR/build_vlm_dataset.log"

echo "$VLM_DATASET_DIR" > "$VIGNOCR_RUN_DIR/VLM_DATASET_DIR"
touch "$VIGNOCR_RUN_DIR/BUILD_VLM_OK"
vlog "stage 12 complete — dataset=$VLM_DATASET_DIR (review queue: $VLM_DATASET_DIR/review_values.csv)"
