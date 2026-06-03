#!/usr/bin/env bash
# =============================================================================
# slurm/05_eval_ocr.sbatch  —  STAGE 5: evaluate the OCR recognizer.
#
# Single A100 (recognition inference). Reports per-field CER and abstention
# precision. Abstention thresholds come from configs/parsing/fields.yaml
# (abstention[flow]; selling stricter than receiving) — NOT hardcoded here.
#
# Entrypoint (config-driven library API, see docs/INTERFACES.md `ocr/`):
#   vignocr.ocr.eval.run(cfg_path=None, ckpt=None, split=None, *, flow=None)
#   -> {overall, per_type, per_field, business_critical, gate{cer_gate,...}, n}
#   `ckpt` points the baseline backend at the fine-tuned recognizer (rec_model_dir).
#   It reads the dataset root from config (lib.sh symlink) — no root argument.
#
# GATE (real data): business_critical CER ≤ cer_gate (configs/ocr/recognition.yaml:
# eval.cer_gate, default 0.05); abstention precision documented. The eval module
# computes gate.passed; this script reports it (no hardcoded thresholds).
#
# CKPT discovery: explicit $VIGNOCR_OCR_CKPT -> BEST_CKPT of the latest
# 04_train_ocr run. Flow profile via $VIGNOCR_FLOW (selling|receiving).
# =============================================================================
#SBATCH --job-name=vignocr-05-eval-ocr
#SBATCH --account=def-<PI>
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
# Centralized logs. SLURM writes its .out/.err here (flat, per stage); lib.sh's
# vignocr_install_postmortem then creates logs/slurm/<stage>/<jobid>/ for copied
# train logs + FAILED/DONE markers. The per-stage dir is pre-created by
# submit_all.sh; --chdir anchors CWD to the repo root.
#SBATCH --output=logs/slurm/05_eval_ocr/%x-%j.out
#SBATCH --error=logs/slurm/05_eval_ocr/%x-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"

vignocr_preamble "05_eval_ocr"

EVAL_SPLIT="${VIGNOCR_EVAL_SPLIT:-valid}"
FLOW="${VIGNOCR_FLOW:-selling}"

CKPT="${VIGNOCR_OCR_CKPT:-}"
if [[ -z "$CKPT" ]]; then
  TRAIN_RUN="$(vignocr_latest_run 04_train_ocr)"
  [[ -n "$TRAIN_RUN" ]] || vdie "no OCR training run found and VIGNOCR_OCR_CKPT unset — run stage 04 first or pass VIGNOCR_OCR_CKPT"
  [[ -f "$TRAIN_RUN/BEST_CKPT" ]] && CKPT="$(cat "$TRAIN_RUN/BEST_CKPT")"
fi
# The recognizer "checkpoint" is a directory (rec_model_dir); accept dir OR file.
[[ -n "$CKPT" && -e "$CKPT" ]] || vdie "OCR checkpoint not found: '${CKPT:-<empty>}'"
vlog "evaluating OCR checkpoint: $CKPT (split=$EVAL_SPLIT flow=$FLOW)"
echo "$CKPT" > "$VIGNOCR_RUN_DIR/EVAL_CKPT"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

METRICS_JSON="$VIGNOCR_RUN_DIR/ocr_metrics.json"
vlog "running OCR eval -> $METRICS_JSON"
# run(cfg_path, ckpt, split, *, flow). cfg_path=None -> configs/ocr/recognition.
# `set +e` around the tee'd pipeline to capture PIPESTATUS for our own error.
set +e
python - "$CKPT" "$EVAL_SPLIT" "$FLOW" "$METRICS_JSON" 2>&1 <<'PY' | tee "$VIGNOCR_RUN_DIR/eval_ocr.log"
import json, sys
from pathlib import Path
ckpt, split, flow, out = sys.argv[1:5]
from vignocr.ocr.eval import run
metrics = run(None, ckpt=Path(ckpt), split=split, flow=flow)
Path(out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
g = metrics.get("gate", {})
bc = metrics.get("business_critical", {})
print(f"[ocr-eval] n={metrics.get('n')} overall_cer={metrics.get('overall', {}).get('cer')} "
      f"bc_cer={bc.get('cer')} abst_prec={bc.get('abstention_precision')} "
      f"gate_passed={g.get('passed')} (cer_gate={g.get('cer_gate')})")
PY
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "OCR eval failed (rc=$rc) — see $VIGNOCR_RUN_DIR/eval_ocr.log"

touch "$VIGNOCR_RUN_DIR/EVAL_OCR_OK"
vlog "stage 05 complete — metrics=$METRICS_JSON run_dir=$VIGNOCR_RUN_DIR"
