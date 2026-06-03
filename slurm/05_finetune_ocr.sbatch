#!/usr/bin/env bash
# =============================================================================
# slurm/05_finetune_ocr.sbatch  —  STAGE C: PaddleOCR recognition fine-tune
# on the human-corrected dataset that came out of stage 04.
#
# Run this AFTER you have uploaded the corrected paddle.txt / labels.csv files
# back into $VIGNOCR_OCR_DATASET_DIR (typically ~/scratch/vignocr/ocr_dataset).
# This is NOT chained via afterok by default — the human-review step gates it.
# =============================================================================
#SBATCH --job-name=vignocr-05-finetune-ocr
#SBATCH --account=def-<PI>
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -Eeuo pipefail
SBATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=lib.sh
source "$SBATCH_DIR/lib.sh"

vignocr_preamble "05_finetune_ocr"

CFG="${VIGNOCR_OCR_FINETUNE_CONFIG:-ocr/finetune}"
DATASET_DIR="${VIGNOCR_OCR_DATASET_DIR:-$VIGNOCR_SCRATCH_DIR/ocr_dataset}"
BASE_CKPT="${VIGNOCR_OCR_BASE_CKPT:-}"   # optional pretrained PaddleOCR rec weights

[[ -d "$DATASET_DIR" ]] || vdie "corrected dataset not found at $DATASET_DIR — upload stage 04 output here first"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
vlog "GPU check:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

vlog "fine-tune: cfg=$CFG dataset=$DATASET_DIR base_ckpt=${BASE_CKPT:-<from-config>}"
set +e
python - "$CFG" "$VIGNOCR_RUN_DIR" "$DATASET_DIR" "${BASE_CKPT:-}" 2>&1 <<'PY' | tee "$VIGNOCR_RUN_DIR/finetune.log"
import sys
from pathlib import Path
cfg, run_dir, ds_dir, base = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), (sys.argv[4] or None)
from vignocr.ocr.finetune import run
best = run(cfg, run_dir, dataset_dir=ds_dir, base_ckpt=base)
print(f"VIGNOCR_BEST_OCR_CKPT={best}")
PY
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "fine-tune failed (rc=$rc) — see $VIGNOCR_RUN_DIR/finetune.log"

BEST="$(grep -oE 'VIGNOCR_BEST_OCR_CKPT=.*' "$VIGNOCR_RUN_DIR/finetune.log" | tail -n1 | cut -d= -f2- || true)"
if [[ -n "$BEST" && -f "$BEST" ]]; then
  echo "$BEST" > "$VIGNOCR_RUN_DIR/BEST_OCR_CKPT"
fi

touch "$VIGNOCR_RUN_DIR/FINETUNE_OCR_OK"
vlog "stage 05 complete — best=$(cat "$VIGNOCR_RUN_DIR/BEST_OCR_CKPT" 2>/dev/null || echo '?')"
vlog "  NEXT: set configs/ocr/recognition.yaml: backend_checkpoint to the path above,"
vlog "        re-deploy the inference service, and run the regression suite."
