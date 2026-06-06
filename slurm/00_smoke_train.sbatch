#!/usr/bin/env bash
# =============================================================================
# slurm/00_smoke_train.sbatch — FAST end-to-end smoke for the detection trainer.
#
# Proves, in ONE clean BATCH job, that the fixed environment actually trains:
#   • pytorch_lightning / rfdetr.training import (RC-5),
#   • RF-DETR's train() accepts our kwargs (lr_encoder, etc.),
#   • a 1-epoch Stage A run produces a checkpoint + the class_names.json map.
#
# WHY A BATCH JOB (not interactive salloc)
#   The interactive smoke failed for two NON-code reasons:
#     1. `salloc` + the commands were pasted as one block, so the commands ran
#        while salloc was still QUEUING (mangled, then killed at the wall).
#     2. The manual `python ...` did NOT source lib.sh, so HF_HUB_OFFLINE /
#        TRANSFORMERS_OFFLINE were unset — and RF-DETR building its DINOv2
#        backbone reached the HF Hub, which HANGS on an offline compute node.
#   This script sources lib.sh first, so the OFFLINE env + venv + module stack
#   are exactly what the real training jobs use. Submit it and read one log.
#
# USAGE (login node):
#   export VIGNOCR_ACCOUNT=def-khenni VIGNOCR_PI=khenni
#   cd $HOME/projects/def-khenni/ydait/pharmakon-ocr
#   mkdir -p logs/slurm/00_smoke
#   sbatch --account=$VIGNOCR_ACCOUNT --chdir=$PWD slurm/00_smoke_train.sbatch
#   # watch:  tail -f logs/slurm/00_smoke/vignocr-00-smoke-*.out
# =============================================================================
#SBATCH --job-name=vignocr-00-smoke
#SBATCH --account=def-khenni
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/00_smoke/%x-%j.out
#SBATCH --error=logs/slurm/00_smoke/%x-%j.err

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"
source "$SLURM_DIR/lib.sh"

# Env contract + storage tiers + OFFLINE env (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
# / TORCH_HOME / HF_HOME) + venv + module stack — same as every real training job.
vignocr_require_env
vignocr_paths
vignocr_load_modules
vignocr_activate_venv

export VIGNOCR_MAX_EPOCHS="${VIGNOCR_MAX_EPOCHS:-1}"   # cap epochs for the smoke
DET_CONFIG="detection/rfdetr_vignette"                 # Stage A = smallest/fastest

vlog "offline env: HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-unset}"
vlog "TORCH_HOME=${TORCH_HOME:-unset}  HF_HOME=${HF_HOME:-unset}"

PRETRAIN="$(vignocr_pretrained_weights "$DET_CONFIG")"
if [[ -n "$PRETRAIN" ]]; then
  vlog "pretrained (cached, offline-safe): $PRETRAIN"
else
  vwarn "no cached pretrained weights — run 'bash scripts/fetch_pretrained.sh' on the LOGIN node first"
fi

vlog "GPU:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

# 1) Time the imports so a SLOW cold-Lustre import is visible (and not mistaken
#    for a hang). First import of torch+rfdetr+transformers+lightning off Lustre
#    legitimately takes 1-3 min; that is NOT a failure.
vlog "importing ML stack (first import from Lustre can take 1-3 min)..."
python - <<'PY'
import time; t = time.time()
import torch
import rfdetr, rfdetr.training, pytorch_lightning, torchmetrics  # noqa: F401
print(f"[smoke] imports OK in {time.time()-t:.1f}s | torch {torch.__version__} "
      f"cuda_avail={torch.cuda.is_available()} pl={pytorch_lightning.__version__}")
PY

# 2) Run ONE epoch of Stage A through the REAL trainer (exercises the kwargs fix
#    + label-map persistence + checkpoint discovery — the full code path).
RUN_DIR="$VIGNOCR_SCRATCH_DIR/runs/00_smoke/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
vlog "running ${VIGNOCR_MAX_EPOCHS}-epoch Stage A via vignocr.detection.train.run -> $RUN_DIR"
set +e
python - "$DET_CONFIG" "$RUN_DIR" 2>&1 <<'PY' | tee "$RUN_DIR/smoke_train.log"
import sys
from pathlib import Path
from vignocr.detection.train import run
best = run(sys.argv[1], Path(sys.argv[2]))
print(f"[smoke] DONE best_checkpoint={best}")
PY
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -eq 0 ]] && grep -q "best_checkpoint=" "$RUN_DIR/smoke_train.log"; then
  vlog "============================================================"
  vlog " SMOKE PASSED — imports + RF-DETR train() kwargs + 1-epoch run OK"
  vlog " checkpoint: $(grep -oE 'best_checkpoint=.*' "$RUN_DIR/smoke_train.log" | tail -n1)"
  vlog "============================================================"
else
  vdie "SMOKE FAILED (rc=$rc) — see $RUN_DIR/smoke_train.log and the .err above"
fi
