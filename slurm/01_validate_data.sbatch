#!/usr/bin/env bash
# =============================================================================
# slurm/01_validate_data.sbatch  —  STAGE 1: dataset integrity + stats.
#
# CPU-ONLY (no GPU): validation reads COCO + asserts the data.yaml integrity
# invariants (split leakage, bbox validity, class-names ⊆ schema, business-
# critical coverage) and prints per-class stats. This MUST be green before any
# training job consumes the dataset — it is the first node of the DAG.
#
# Entrypoints (config-driven library API, see docs/INTERFACES.md `data/`):
#   vignocr.data.validate.check_integrity(root) -> IntegrityReport (raises on fail)
#   vignocr.data.stats.summarize(root)          -> dict (per-class counts, box stats)
# We call these via a small inline driver (the modules expose library functions;
# their __main__ shims take no args and read the config root — we instead pass the
# resolved root and persist JSON into the run dir for provenance).
#
# Resource sizing: validation is light + IO-bound; a modest CPU box is plenty.
# Override the account at submit time:  sbatch --account=$VIGNOCR_ACCOUNT ...
# =============================================================================
#SBATCH --job-name=vignocr-01-validate
#SBATCH --account=def-<PI>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Centralized logs. SLURM writes its .out/.err here (flat, per stage); lib.sh's
# vignocr_install_postmortem then creates logs/slurm/<stage>/<jobid>/ for copied
# train logs + FAILED/DONE markers. The per-stage dir is pre-created by
# submit_all.sh; --chdir anchors CWD to the repo root.
#SBATCH --output=logs/slurm/01_validate/%x-%j.out
#SBATCH --error=logs/slurm/01_validate/%x-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
PROJECT_DIR="$HOME/projects/def-khenni/ydait/pharmakon-ocr"
SLURM_DIR="$PROJECT_DIR/slurm"

source "$SLURM_DIR/lib.sh"
vignocr_preamble "01_validate"

# Logs live inside the run dir for full provenance. Use the PACKAGE-resolved root
# (synthetic -> fixtures/synthetic; real -> <repo>/data symlinked to project space)
# so this agrees with the config-driven training stages.
LOG="$VIGNOCR_RUN_DIR/validate.log"
DATA_ROOT="$VIGNOCR_RESOLVED_DATA_ROOT"
vlog "stage 01 validate_data (root=$DATA_ROOT) -> $LOG"

# Inline driver: integrity (fatal on failure) + stats (informational). The body
# is a LITERAL heredoc ('PY' quoted) so the shell never touches it; the run dir
# + data root arrive via argv. The package selects synthetic/real from
# configs/data.yaml (override with VIGNOCR_DATA_ACTIVE — no code change).
set +e
python - "$DATA_ROOT" "$VIGNOCR_RUN_DIR" 2>&1 <<'PY' | tee "$LOG"
import json, sys
from pathlib import Path
root, run_dir = sys.argv[1], Path(sys.argv[2])
from vignocr.data.validate import check_integrity, IntegrityError
from vignocr.data.stats import summarize

# Stats first (never fatal) so we always emit a dataset profile.
try:
    summary = summarize(root)
    (run_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[stats] per-split:", summary.get("splits"))
    print("[stats] per-class:", summary.get("per_class"))
except Exception as exc:  # noqa: BLE001
    print(f"[stats] WARN: {exc}")

# Integrity gate (fatal): raises IntegrityError on any violated invariant.
report = check_integrity(root)
print("[validate] report:", json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
(run_dir / "integrity.json").write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
print("[validate] integrity OK")
PY
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -ne 0 ]]; then
  vdie "data integrity check FAILED (rc=$rc) — see $LOG. Fix the dataset before training."
fi

# A marker the downstream DAG can assert on (afterok already gates this, but the
# marker makes a standalone resume explicit).
touch "$VIGNOCR_RUN_DIR/VALIDATE_OK"
vlog "stage 01 complete — dataset integrity OK. run_dir=$VIGNOCR_RUN_DIR"
