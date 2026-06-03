#!/usr/bin/env bash
# =============================================================================
# slurm/04_autolabel_ocr.sbatch  —  STAGE C BOOTSTRAP: auto-label every field
# crop with PaddleOCR so the user can correct in Roboflow and feed the result
# back into 05_finetune_ocr.sbatch.
#
# Runs on a single GPU node (PaddleOCR uses CUDA when available, falls back to
# CPU otherwise). One A100 is overkill but keeps the DAG single-tier — adjust to
# a smaller GPU partition if your cluster offers one.
# =============================================================================
#SBATCH --job-name=vignocr-04-autolabel
#SBATCH --account=def-<PI>
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Centralized logs. SLURM writes its .out/.err here (flat, per stage); lib.sh's
# vignocr_install_postmortem then creates logs/slurm/<stage>/<jobid>/ for copied
# train logs + FAILED/DONE markers. The per-stage dir is pre-created by
# submit_all.sh; --chdir anchors CWD to the repo root.
#SBATCH --output=logs/slurm/04_autolabel_ocr/%x-%j.out
#SBATCH --error=logs/slurm/04_autolabel_ocr/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"

vignocr_preamble "04_autolabel_ocr"

# Which data.yaml block to crop from; default = Stage B fields (real).
DATASET="${VIGNOCR_AUTOLABEL_DATASET:-real}"
# Output dir lives on scratch (lots of small JPEGs) and is symlinked into the run dir
# so reviewers/cleanups have one canonical path.
OUT_DIR="${VIGNOCR_OCR_DATASET_DIR:-$VIGNOCR_SCRATCH_DIR/ocr_dataset}"
BACKEND="${VIGNOCR_OCR_AUTOLABEL_BACKEND:-paddleocr}"

mkdir -p "$OUT_DIR"
ln -sfn "$OUT_DIR" "$VIGNOCR_RUN_DIR/ocr_dataset"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
vlog "GPU check before autolabel:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

vlog "auto-labelling: dataset=$DATASET backend=$BACKEND output=$OUT_DIR"
set +e
python -m vignocr.ocr.autolabel \
    --dataset "$DATASET" \
    --output  "$OUT_DIR" \
    --backend "$BACKEND" \
    --seed    "$VIGNOCR_SEED" \
    2>&1 | tee "$VIGNOCR_RUN_DIR/autolabel.log"
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "autolabel failed (rc=$rc) — see $VIGNOCR_RUN_DIR/autolabel.log"

# Surface the manifest summary right where the operator will look.
cp -f "$OUT_DIR/manifest.json" "$VIGNOCR_RUN_DIR/autolabel_manifest.json" || true

touch "$VIGNOCR_RUN_DIR/AUTOLABEL_OK"
vlog "stage 04 complete — output=$OUT_DIR (review_low_conf.csv + manifest.json inside)"
vlog "  NEXT: zip $OUT_DIR, import to Roboflow (or your tool), correct labels, and"
vlog "        re-upload to $OUT_DIR (overwriting paddle.txt / labels.csv) for stage 05."
