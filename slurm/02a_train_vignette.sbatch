#!/usr/bin/env bash
# =============================================================================
# slurm/02a_train_vignette.sbatch  —  STAGE A: vignette-region detector.
#
# Trains a small RF-DETR head on data2/ (3 classes: entete, vin, date_info) so
# the production pipeline can find the vignette inside a wider drug-box photo
# before handing the crop to the Stage B field detector.
#
# Smaller resource budget than Stage B: fewer classes (3 vs 17), smaller frames
# (416×416 vs 640×640), easier task. Same structure / safety policy / resume
# semantics as 02_train_detection.sbatch (Stage B) — only the bound config and
# the run-dir stage name differ. Resumable on requeue.
# =============================================================================
#SBATCH --job-name=vignocr-02a-train-vignette
#SBATCH --account=def-<PI>
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"
vignocr_preamble "02a_train_vignette"

# Bind to Stage A; train.py resolves dataset (vignette -> data2/) + head width (3)
# from this config.
DET_CONFIG="${VIGNOCR_DETECTION_CONFIG:-detection/rfdetr_vignette}"

# Resume from a prior interrupted run of THIS run dir, if a checkpoint exists in
# the Stage A scratch tree. We probe both the canonical scratch (when the env
# was overridden) and the default scratch root.
_resume_probe() {
  local rd_name; rd_name="$(basename "$VIGNOCR_RUN_DIR")"
  for d in \
      "${VIGNOCR_SCRATCH:-}/detection/rfdetr_vignette/$rd_name" \
      "$VIGNOCR_SCRATCH_DIR/detection/rfdetr_vignette/$rd_name" \
      "$VIGNOCR_RUN_DIR"; do
    [[ -z "$d" ]] && continue
    for ckpt in "$d/checkpoint_last.pth" "$d/last.ckpt" "$d/checkpoint.pth"; do
      [[ -f "$ckpt" ]] && { echo "$ckpt"; return; }
    done
  done
  echo ""
}
RESUME_CKPT="$(_resume_probe)"
[[ -n "$RESUME_CKPT" ]] && vlog "resuming vignette training from $RESUME_CKPT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
vlog "GPU check before training:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

vlog "launching RF-DETR vignette training: config=$DET_CONFIG seed=$VIGNOCR_SEED run_dir=$VIGNOCR_RUN_DIR"
set +e
python - "$DET_CONFIG" "$VIGNOCR_RUN_DIR" "${RESUME_CKPT:-}" 2>&1 <<'PY' | tee "$VIGNOCR_RUN_DIR/train_vignette.log"
import sys
from pathlib import Path
cfg_path, run_dir, resume = sys.argv[1], Path(sys.argv[2]), (sys.argv[3] or None)
from vignocr.detection.train import run
best = run(cfg_path, run_dir, resume=resume)
print(f"VIGNOCR_BEST_CKPT={best}")
PY
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "vignette training failed (rc=$rc) — see $VIGNOCR_RUN_DIR/train_vignette.log"

BEST="$(grep -oE 'VIGNOCR_BEST_CKPT=.*' "$VIGNOCR_RUN_DIR/train_vignette.log" | tail -n1 | cut -d= -f2- || true)"
if [[ -n "$BEST" && -f "$BEST" ]]; then
  echo "$BEST" > "$VIGNOCR_RUN_DIR/BEST_CKPT"
else
  vwarn "no checkpoint reported after training (search the rfdetr_vignette scratch tree manually)"
fi

touch "$VIGNOCR_RUN_DIR/TRAIN_VIGNETTE_OK"
vlog "stage 02a complete — best=$(cat "$VIGNOCR_RUN_DIR/BEST_CKPT" 2>/dev/null || echo '?') run_dir=$VIGNOCR_RUN_DIR"
