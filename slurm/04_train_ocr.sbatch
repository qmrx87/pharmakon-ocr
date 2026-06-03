#!/usr/bin/env bash
# =============================================================================
# slurm/04_train_ocr.sbatch  —  STAGE 4: OCR recognition training (single A100).
#
# Trains the per-field recognizer (PaddleOCR/CRNN baseline; transformer variant
# behind a config switch). Field-type-aware preprocessing + per-field
# orientation correction (rotated `vin` fields) are config-driven in the OCR
# module — this script only schedules + checkpoints the run to ~/scratch.
#
# Entrypoint (config-driven library API, see docs/INTERFACES.md `ocr/`):
#   vignocr.ocr.train.run(cfg_path, run_dir) -> Path(run_dir)
#   The backend's best-model path is recorded in run_dir/manifest.json["checkpoint"].
#
# Crops source: the OCR trainer builds its training set directly from the active
# dataset's ANNOTATED field crops (data/coco.crops_for_image) + gold labels from
# the fixture ground_truth.json. It reads the dataset root from configs/data.yaml
# (lib.sh symlinks that to the project-space data) — it does NOT take a detector
# or root argument. (A detector is only needed for end-to-end inference, exercised
# in the stage-06 benchmark, not for recognizer fine-tuning on gold boxes.)
#
# NOTE: ocr.train.run has no resume parameter; an interrupted OCR run is re-run
# fresh. VIGNOCR_RESUME_RUN_DIR still reuses the run dir (so logs/snapshot land in
# the same place) but training restarts from the configured base model.
# =============================================================================
#SBATCH --job-name=vignocr-04-train-ocr
#SBATCH --account=def-<PI>
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
# Centralized logs. SLURM writes its .out/.err here (flat, per stage); lib.sh's
# vignocr_install_postmortem then creates logs/slurm/<stage>/<jobid>/ for copied
# train logs + FAILED/DONE markers. The per-stage dir is pre-created by
# submit_all.sh; --chdir anchors CWD to the repo root.
#SBATCH --output=logs/slurm/04_train_ocr/%x-%j.out
#SBATCH --error=logs/slurm/04_train_ocr/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"

vignocr_preamble "04_train_ocr"

# OCR config NAME (load_config resolves under configs/); overridable.
OCR_CONFIG="${VIGNOCR_OCR_CONFIG:-ocr/recognition}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

vlog "launching OCR training (single A100): config=$OCR_CONFIG seed=$VIGNOCR_SEED run_dir=$VIGNOCR_RUN_DIR"
# run(cfg_path, run_dir) returns run_dir; the trainer writes manifest.json with the
# backend's best-model path under ["checkpoint"]. We surface that path on a sentinel.
# NOTE: ocr.train loads the cfg via load_config(name) when given a name, or load_yaml
# when given a path; we pass the config NAME (its __init__ branch handles both).
# `set +e` around the tee'd pipeline to capture PIPESTATUS for our own error.
set +e
python - "$OCR_CONFIG" "$VIGNOCR_RUN_DIR" 2>&1 <<'PY' | tee "$VIGNOCR_RUN_DIR/train_ocr.log"
import json, sys
from pathlib import Path
cfg_path, run_dir = sys.argv[1], Path(sys.argv[2])
from vignocr.ocr.train import run
out_dir = run(cfg_path, run_dir)
# The backend best-model path lives in the manifest the trainer just wrote.
ckpt = None
manifest = Path(out_dir) / "manifest.json"
if manifest.exists():
    ckpt = json.loads(manifest.read_text(encoding="utf-8")).get("checkpoint")
print(f"VIGNOCR_OCR_CKPT={ckpt or out_dir}")
PY
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "OCR training failed (rc=$rc) — see $VIGNOCR_RUN_DIR/train_ocr.log"

# Record the recognizer checkpoint dir (what ocr.eval/benchmark load as rec_model_dir).
CKPT="$(grep -oE 'VIGNOCR_OCR_CKPT=.*' "$VIGNOCR_RUN_DIR/train_ocr.log" | tail -n1 | cut -d= -f2- || true)"
[[ -n "$CKPT" ]] && echo "$CKPT" > "$VIGNOCR_RUN_DIR/BEST_CKPT" || vwarn "no OCR checkpoint recorded in manifest"

touch "$VIGNOCR_RUN_DIR/TRAIN_OCR_OK"
vlog "stage 04 complete — recognizer=$(cat "$VIGNOCR_RUN_DIR/BEST_CKPT" 2>/dev/null || echo '?') run_dir=$VIGNOCR_RUN_DIR"
