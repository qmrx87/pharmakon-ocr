#!/usr/bin/env bash
# =============================================================================
# slurm/03_eval_export_detection.sbatch  —  STAGE 3: evaluate the trained
# detector and export it to ONNX (with a torch-vs-ONNX parity check).
#
# Single A100 (eval is light but the model + parity check want a GPU). Reads the
# best checkpoint produced by stage 02 / 02b, writes metrics + the .onnx into a
# fresh eval run dir under ~/scratch.
#
# Entrypoints (config-driven library API, see docs/INTERFACES.md `detection/`):
#   vignocr.detection.eval.run(ckpt, root, split="valid")
#        -> {map, map_50, per_class_ap, localization_recall{recall,per_class,missing}}
#   vignocr.detection.export.to_onnx(ckpt, out)
#        -> ONNX + torch-vs-onnxruntime parity check (RAISES on mismatch)
#
# GATE (real data): mAP@.5 ≥ 0.85 AND localization_recall == 1.0 on the
# business-critical fields (configs/classes.yaml: business_critical_fields).
# The metric thresholds live in config / the eval module — this script reports
# and persists them; it does not hardcode the numbers.
#
# CKPT discovery order: explicit $VIGNOCR_DET_CKPT  ->  BEST_CKPT of the latest
# 02_train_detection run dir. Override the eval split via $VIGNOCR_EVAL_SPLIT.
# =============================================================================
#SBATCH --job-name=vignocr-03-eval-export
#SBATCH --account=def-<PI>
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
# Centralized logs. SLURM writes its .out/.err here (flat, per stage); lib.sh's
# vignocr_install_postmortem then creates logs/slurm/<stage>/<jobid>/ for copied
# train logs + FAILED/DONE markers. The per-stage dir is pre-created by
# submit_all.sh; --chdir anchors CWD to the repo root.
#SBATCH --output=logs/slurm/03_eval_export_detection/%x-%j.out
#SBATCH --error=logs/slurm/03_eval_export_detection/%x-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"

vignocr_preamble "03_eval_export_detection"

EVAL_SPLIT="${VIGNOCR_EVAL_SPLIT:-valid}"
# Which detection config produced the checkpoint (so find_best searches the
# right scratch tree). Default to Stage B; switch to Stage A by exporting
# VIGNOCR_DETECTION_CONFIG=detection/rfdetr_vignette before sbatch.
DET_CONFIG="${VIGNOCR_DETECTION_CONFIG:-detection/rfdetr_medium}"
export VIGNOCR_DETECTION_CONFIG="$DET_CONFIG"

# --- locate the checkpoint to evaluate ---
CKPT="${VIGNOCR_DET_CKPT:-}"
if [[ -z "$CKPT" ]]; then
  # When evaluating Stage A, look under 02a_train_vignette; else default to 02.
  TRAIN_STAGE="${VIGNOCR_DETECTION_TRAIN_STAGE:-02_train_detection}"
  case "$DET_CONFIG" in
    *vignette*) TRAIN_STAGE="${VIGNOCR_DETECTION_TRAIN_STAGE:-02a_train_vignette}" ;;
  esac
  TRAIN_RUN="$(vignocr_latest_run "$TRAIN_STAGE")"
  [[ -n "$TRAIN_RUN" ]] || vdie "no $TRAIN_STAGE run found and VIGNOCR_DET_CKPT unset — run the training stage first or pass VIGNOCR_DET_CKPT"
  if [[ -f "$TRAIN_RUN/BEST_CKPT" ]]; then
    CKPT="$(cat "$TRAIN_RUN/BEST_CKPT")"
  else
    CKPT="$(vignocr_find_best "$TRAIN_RUN" "$DET_CONFIG")"
  fi
fi
[[ -n "$CKPT" && -f "$CKPT" ]] || vdie "detection checkpoint not found: '${CKPT:-<empty>}'"
vlog "evaluating checkpoint: $CKPT (config=$DET_CONFIG split=$EVAL_SPLIT)"
echo "$CKPT" > "$VIGNOCR_RUN_DIR/EVAL_CKPT"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

# --- eval + export via the documented library API (one driver, both steps) ---
METRICS_JSON="$VIGNOCR_RUN_DIR/detection_metrics.json"
ONNX_OUT="$VIGNOCR_RUN_DIR/detector.onnx"
vlog "running detection eval -> $METRICS_JSON ; ONNX export -> $ONNX_OUT (parity-checked)"
# Use the PACKAGE-resolved root so eval reads the same data training did.
# `set +e` around the tee'd pipeline to capture PIPESTATUS for our own error.
set +e
python - "$CKPT" "$VIGNOCR_RESOLVED_DATA_ROOT" "$EVAL_SPLIT" "$METRICS_JSON" "$ONNX_OUT" 2>&1 \
  <<'PY' | tee "$VIGNOCR_RUN_DIR/eval_export_detection.log"
import json, sys
from pathlib import Path
ckpt, root, split, metrics_out, onnx_out = sys.argv[1:6]
from vignocr.detection.eval import run as eval_run
from vignocr.detection.export import to_onnx

metrics = eval_run(ckpt, root, split=split)
Path(metrics_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
loc = metrics.get("localization_recall", {})
print(f"[eval] map={metrics.get('map')} map_50={metrics.get('map_50')} "
      f"loc_recall={loc.get('recall')} missing={loc.get('missing')}")

# Export + parity check (to_onnx RAISES on parity mismatch -> non-zero exit).
out = to_onnx(ckpt, onnx_out)
print(f"[export] ONNX written + parity OK: {out}")
PY
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "detection eval/export failed (rc=$rc) — see $VIGNOCR_RUN_DIR/eval_export_detection.log"

[[ -f "$ONNX_OUT" ]] && echo "$ONNX_OUT" > "$VIGNOCR_RUN_DIR/DETECTOR_ONNX" || vwarn "ONNX file missing after export"
touch "$VIGNOCR_RUN_DIR/EVAL_EXPORT_OK"
vlog "stage 03 complete — metrics=$METRICS_JSON onnx=$ONNX_OUT run_dir=$VIGNOCR_RUN_DIR"
