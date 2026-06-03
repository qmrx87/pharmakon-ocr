#!/usr/bin/env bash
# =============================================================================
# slurm/02_train_detection.sbatch  —  STAGE 2: RF-DETR detection training on a
# SINGLE A100 (Narval). Checkpoints to ~/scratch run dir; resumable.
#
# Narval GPU node = 4×A100-40GB / 48 cores / ~510G. One GPU's fair share is
# ~12 cores + ~120G. We request exactly one A100 here (use 02b for multi-GPU).
#
# Entrypoint (config-driven library API, see docs/INTERFACES.md `detection/`):
#   vignocr.detection.train.run(cfg_path, run_dir, resume=None) -> Path(best ckpt)
# RF-DETR places checkpoints under $VIGNOCR_SCRATCH/detection/rfdetr_medium/<run
# name> (lib.sh exports VIGNOCR_SCRATCH) and writes a run_meta.json snapshot into
# run_dir. We capture the returned best-checkpoint path into BEST_CKPT.
#
# RESUME: resubmitting with VIGNOCR_RESUME_RUN_DIR=<dir> reuses that run dir and
# auto-detects the latest checkpoint.pth to continue from (idempotent requeue,
# e.g. after a node failure or hitting the time limit). The DAG passes a fresh
# dir; --requeue + this resume make a preempted job pick up where it left off.
#
# Augmentation note: training MUST use band-color-preserving augmentation (no
# hue/channel flips that invert green↔red). That policy lives in the detection
# config, NOT here — this script only schedules the run.
# =============================================================================
#SBATCH --job-name=vignocr-02-train-det
#SBATCH --account=def-<PI>
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
# Centralized logs. SLURM writes its .out/.err here (flat, per stage); lib.sh's
# vignocr_install_postmortem then creates logs/slurm/<stage>/<jobid>/ for copied
# train logs + FAILED/DONE markers. The per-stage dir is pre-created by
# submit_all.sh; --chdir anchors CWD to the repo root.
#SBATCH --output=logs/slurm/02_train_detection/%x-%j.out
#SBATCH --error=logs/slurm/02_train_detection/%x-%j.err
#SBATCH --open-mode=append
# Requeue automatically if the node is preempted/fails (resume picks up last.ckpt).
#SBATCH --requeue

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"

vignocr_preamble "02_train_detection"

# Detection config NAME (load_config resolves it under configs/); overridable.
DET_CONFIG="${VIGNOCR_DETECTION_CONFIG:-detection/rfdetr_medium}"
export VIGNOCR_DETECTION_CONFIG="$DET_CONFIG"   # propagate to lib helpers

# Resume from a prior interrupted run of THIS run dir, if present. The helper is
# now CONFIG-AWARE — it reads cfg.train.checkpoint.dir from $DET_CONFIG so a new
# detection config doesn't need a code change here.
RESUME_CKPT="$(vignocr_find_resume "$VIGNOCR_RUN_DIR" "$DET_CONFIG")"
[[ -n "$RESUME_CKPT" ]] && vlog "resuming detection training from $RESUME_CKPT"

# Offline-init: surface the cached pretrained weights up front so the operator
# sees whether the trainer will use them. If empty, the trainer will fall back
# to VIGNOCR_ALLOW_ONLINE_PRETRAIN (compute-node jobs WITHOUT this env will fail
# fast with a clear "run scripts/fetch_pretrained.sh" message — no 6-min hang).
PRETRAIN_CACHE="$(vignocr_pretrained_weights "$DET_CONFIG")"
if [[ -n "$PRETRAIN_CACHE" ]]; then
  vlog "pretrained weights (cached, offline-safe): $PRETRAIN_CACHE"
else
  vwarn "no cached pretrained weights at \$VIGNOCR_PRETRAINED_DIR — if compute node, training will refuse to start; run 'bash scripts/fetch_pretrained.sh' on the LOGIN node first"
fi

# Pin determinism: thread count = allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
vlog "GPU check before training:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

vlog "launching RF-DETR training (single A100): config=$DET_CONFIG seed=$VIGNOCR_SEED run_dir=$VIGNOCR_RUN_DIR"
# Inline driver calls the documented run(cfg_path, run_dir, resume) and prints the
# returned best-checkpoint path on a sentinel line we parse out below.
# `set +e` around the tee'd pipeline so we capture PIPESTATUS and emit our own
# actionable error instead of the generic ERR trap (pipefail would abort first).
set +e
python - "$DET_CONFIG" "$VIGNOCR_RUN_DIR" "${RESUME_CKPT:-}" 2>&1 <<'PY' | tee "$VIGNOCR_RUN_DIR/train_detection.log"
import sys
from pathlib import Path
cfg_path, run_dir, resume = sys.argv[1], Path(sys.argv[2]), (sys.argv[3] or None)
from vignocr.detection.train import run
best = run(cfg_path, run_dir, resume=resume)
print(f"VIGNOCR_BEST_CKPT={best}")
PY
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "detection training failed (rc=$rc) — see $VIGNOCR_RUN_DIR/train_detection.log"

# Prefer the path the trainer returned; fall back to discovery by artifact name.
BEST="$(grep -oE 'VIGNOCR_BEST_CKPT=.*' "$VIGNOCR_RUN_DIR/train_detection.log" | tail -n1 | cut -d= -f2- || true)"
[[ -n "$BEST" && -f "$BEST" ]] || BEST="$(vignocr_find_best "$VIGNOCR_RUN_DIR")"
[[ -n "$BEST" ]] && echo "$BEST" > "$VIGNOCR_RUN_DIR/BEST_CKPT" \
  || vwarn "no checkpoint found after training (searched run dir + scratch)"

touch "$VIGNOCR_RUN_DIR/TRAIN_DETECTION_OK"
vlog "stage 02 complete — best=$(cat "$VIGNOCR_RUN_DIR/BEST_CKPT" 2>/dev/null || echo '?') run_dir=$VIGNOCR_RUN_DIR"
