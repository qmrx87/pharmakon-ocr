#!/usr/bin/env bash
# =============================================================================
# slurm/13_train_vlm_donut.sbatch — V2a: fine-tune Donut on the VLM dataset.
#
# Entrypoint: vignocr.v2.donut_train.run(cfg, run_dir, resume) -> Path(best dir)
# Same conventions as the v1 trainers: run-dir snapshot, scratch checkpoints,
# offline HF cache (prefetched), resume via VIGNOCR_RESUME_RUN_DIR + the
# best/ checkpoint dir. ~500 train images x 30 epochs on one A100 ≈ 2-4 h.
# =============================================================================
#SBATCH --job-name=vignocr-13-train-vlm
#SBATCH --account=def-<PI>
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/13_train_vlm_donut/%x-%j.out
#SBATCH --error=logs/slurm/13_train_vlm_donut/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"
vignocr_preamble "13_train_vlm_donut"

# Dataset: explicit env > the latest stage-12 run's pointer file.
if [[ -z "${VIGNOCR_VLM_DATASET_DIR:-}" ]]; then
  BUILD_RUN="$(vignocr_latest_run 12_build_vlm_dataset)"
  [[ -n "$BUILD_RUN" && -f "$BUILD_RUN/VLM_DATASET_DIR" ]] \
    || vdie "VIGNOCR_VLM_DATASET_DIR unset and no 12_build_vlm_dataset run found — run stage 12 first"
  VIGNOCR_VLM_DATASET_DIR="$(cat "$BUILD_RUN/VLM_DATASET_DIR")"
fi
export VIGNOCR_VLM_DATASET_DIR
vlog "VLM dataset: $VIGNOCR_VLM_DATASET_DIR"

# Resume: a prior run dir whose checkpoint tree has a best/ to continue from.
RESUME_ARG=""
if [[ -n "${VIGNOCR_RESUME_RUN_DIR:-}" ]]; then
  CAND="$VIGNOCR_SCRATCH/scratch/v2/donut/$(basename "$VIGNOCR_RESUME_RUN_DIR")/best"
  [[ -d "$CAND" ]] && RESUME_ARG="$CAND" && vlog "resuming Donut from $CAND"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
vlog "GPU check:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

vlog "launching Donut fine-tune: config=v2/vlm_donut run_dir=$VIGNOCR_RUN_DIR"
set +e
python - "v2/vlm_donut" "$VIGNOCR_RUN_DIR" "${RESUME_ARG:-}" 2>&1 <<'PY' | tee "$VIGNOCR_RUN_DIR/train_vlm_donut.log"
import sys
from pathlib import Path
cfg, run_dir, resume = sys.argv[1], Path(sys.argv[2]), (sys.argv[3] or None)
from vignocr.v2.donut_train import run
best = run(cfg, run_dir, resume=resume)
print(f"VIGNOCR_BEST_CKPT={best}")
PY
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "Donut fine-tune failed (rc=$rc) — see $VIGNOCR_RUN_DIR/train_vlm_donut.log"

BEST="$(grep -oE 'VIGNOCR_BEST_CKPT=.*' "$VIGNOCR_RUN_DIR/train_vlm_donut.log" | tail -n1 | cut -d= -f2- || true)"
[[ -n "$BEST" && -d "$BEST" ]] && echo "$BEST" > "$VIGNOCR_RUN_DIR/BEST_CKPT" \
  || vwarn "no checkpoint dir reported after training"

touch "$VIGNOCR_RUN_DIR/TRAIN_VLM_OK"
vlog "stage 13 complete — best=$(cat "$VIGNOCR_RUN_DIR/BEST_CKPT" 2>/dev/null || echo '?')"
