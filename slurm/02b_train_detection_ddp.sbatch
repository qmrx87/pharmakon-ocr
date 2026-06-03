#!/usr/bin/env bash
# =============================================================================
# slurm/02b_train_detection_ddp.sbatch  —  STAGE 2 (multi-GPU): RF-DETR
# detection training across ALL GPUs on a node via torchrun DDP.
#
# Narval GPU node = 4×A100-40GB / 48 cores / ~510G. We grab the whole node:
#   --gres=gpu:a100:4  --cpus-per-task=48  --mem=498G   (one task; torchrun
#   spawns one process per GPU underneath). This is the throughput path; use
#   02_train_detection.sbatch for a quick single-A100 run.
#
# DDP launch pattern (Alliance-recommended): a single Slurm task per node runs
# `torchrun --nproc_per_node=<gpus> ...`; torchrun owns the per-GPU process
# spawn + rendezvous. For a SINGLE node this needs no network rendezvous addr;
# the multi-node generalization (set --nodes>1) wires MASTER_ADDR from the
# Slurm nodelist (left in, commented, for when you scale out).
#
# Entrypoint: the SAME documented trainer as stage 02 —
#   vignocr.detection.train.run(cfg_path, run_dir, resume=None).
# RF-DETR's underlying .train() picks up the distributed context torchrun sets
# (RANK / WORLD_SIZE / LOCAL_RANK), so every torchrun-spawned process simply
# calls run() and RF-DETR coordinates the data-parallel training. Because that
# module exposes a library function (no __main__), torchrun cannot `-m` it; we
# materialize a tiny launcher into the run dir at job time (a generated artifact,
# captured for provenance — NOT a committed source file) that imports and calls
# run(). This keeps the committed orchestration bash-only.
#
# RESUME: identical contract to stage 02 (VIGNOCR_RESUME_RUN_DIR + latest ckpt).
# =============================================================================
#SBATCH --job-name=vignocr-02b-train-ddp
#SBATCH --account=def-<PI>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"
vignocr_preamble "02_train_detection"   # share the detection stage namespace

DET_CONFIG="${VIGNOCR_DETECTION_CONFIG:-detection/rfdetr_medium}"

# Number of GPUs visible to this job (one process per GPU under torchrun).
NPROC="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC="${NPROC:-4}"
NNODES="${SLURM_NNODES:-1}"
export OMP_NUM_THREADS="$(( ${SLURM_CPUS_PER_TASK:-48} / NPROC ))"
[[ "$OMP_NUM_THREADS" -lt 1 ]] && export OMP_NUM_THREADS=1
vlog "DDP topology: nodes=$NNODES gpus_per_node=$NPROC omp_threads=$OMP_NUM_THREADS"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

RESUME_CKPT="$(vignocr_find_resume "$VIGNOCR_RUN_DIR")"
[[ -n "$RESUME_CKPT" ]] && vlog "resuming from $RESUME_CKPT"

# Materialize the DDP launcher (generated artifact, kept in the run dir). It reads
# config/run-dir/resume from argv and calls the documented run(). Every torchrun
# rank executes it; RF-DETR uses the torchrun-provided RANK/WORLD_SIZE/LOCAL_RANK.
LAUNCHER="$VIGNOCR_RUN_DIR/_ddp_launch_detection.py"
cat > "$LAUNCHER" <<'PY'
import os, sys
from pathlib import Path
cfg_path, run_dir, resume = sys.argv[1], Path(sys.argv[2]), (sys.argv[3] or None)
from vignocr.detection.train import run
best = run(cfg_path, run_dir, resume=resume)
# Only the global-rank-0 process records the canonical best-checkpoint sentinel.
if os.environ.get("RANK", "0") in ("0", ""):
    print(f"VIGNOCR_BEST_CKPT={best}")
PY

# Rendezvous: single-node uses --standalone. For multi-node (--nodes>1) uncomment
# below to derive MASTER_ADDR from the first node in the Slurm nodelist.
RDZV_ARGS=(--standalone)
# if [[ "$NNODES" -gt 1 ]]; then
#   MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
#   MASTER_PORT="${MASTER_PORT:-29500}"
#   RDZV_ARGS=(--rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" --nnodes="$NNODES" --rdzv-id="${SLURM_JOB_ID:-vignocr}")
#   vlog "multi-node rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
# fi

vlog "launching RF-DETR DDP training: config=$DET_CONFIG nproc_per_node=$NPROC seed=$VIGNOCR_SEED"
# `srun` launches one task per node; `torchrun` fans out to one rank per GPU.
# `set +e` around the tee'd pipeline to capture PIPESTATUS for our own error.
set +e
srun --kill-on-bad-exit=1 torchrun \
  "${RDZV_ARGS[@]}" \
  --nproc_per_node="$NPROC" \
  "$LAUNCHER" "$DET_CONFIG" "$VIGNOCR_RUN_DIR" "${RESUME_CKPT:-}" \
  2>&1 | tee "$VIGNOCR_RUN_DIR/train_detection_ddp.log"
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "DDP detection training failed (rc=$rc) — see $VIGNOCR_RUN_DIR/train_detection_ddp.log"

BEST="$(grep -oE 'VIGNOCR_BEST_CKPT=.*' "$VIGNOCR_RUN_DIR/train_detection_ddp.log" | tail -n1 | cut -d= -f2- || true)"
[[ -n "$BEST" && -f "$BEST" ]] || BEST="$(vignocr_find_best "$VIGNOCR_RUN_DIR")"
[[ -n "$BEST" ]] && echo "$BEST" > "$VIGNOCR_RUN_DIR/BEST_CKPT" || vwarn "no checkpoint found after DDP training"

touch "$VIGNOCR_RUN_DIR/TRAIN_DETECTION_OK"
vlog "stage 02b complete — best=$(cat "$VIGNOCR_RUN_DIR/BEST_CKPT" 2>/dev/null || echo '?') run_dir=$VIGNOCR_RUN_DIR"
