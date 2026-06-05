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

# Logs live inside the run dir for full provenance.
LOG="$VIGNOCR_RUN_DIR/validate.log"

# Validate the EXACT datasets the training stages bind to — Stage B `real`
# (data/) and Stage A `vignette` (data2/) — not just the global `active` selector.
# Previously stage 01 validated whatever VIGNOCR_DATA_ACTIVE pointed at (default
# `synthetic`), so it could green-light synthetic while 02/02a trained on the real
# exports. Override the list for a synthetic-only smoke test:
#   sbatch --export=ALL,VIGNOCR_VALIDATE_DATASETS="synthetic" slurm/01_validate_data.sbatch
VALIDATE_DATASETS="${VIGNOCR_VALIDATE_DATASETS:-real vignette}"
vlog "stage 01 validate_data (datasets='$VALIDATE_DATASETS') -> $LOG"

# Inline driver: per-dataset stats (informational) + integrity (fatal). The body
# is a LITERAL heredoc ('PY' quoted); run dir + the dataset NAME LIST arrive via
# argv. Each name is resolved through the package's own get_dataset() so root,
# splits, class_names, aliases and severity all match what training will use.
set +e
python - "$VIGNOCR_RUN_DIR" $VALIDATE_DATASETS 2>&1 <<'PY' | tee "$LOG"
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
names = sys.argv[2:] or ["synthetic"]
from vignocr.common.config import get_dataset
from vignocr.data.validate import check_integrity, IntegrityError
from vignocr.data.stats import summarize

failures = []
for name in names:
    print(f"\n========== dataset: {name} ==========")
    try:
        ds = get_dataset(name)
    except KeyError as exc:
        print(f"[{name}] WARN: not declared in data.yaml: {exc}")
        continue
    root = ds["root"]
    print(f"[{name}] root={root}")
    # Stats first (never fatal) so we always emit a profile.
    try:
        summary = summarize(root)
        (run_dir / f"stats.{name}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{name}][stats] per-split:", summary.get("splits"))
        print(f"[{name}][stats] per-class:", summary.get("per_class"))
    except Exception as exc:  # noqa: BLE001
        print(f"[{name}][stats] WARN: {exc}")
    # Integrity gate (fatal): validate AGAINST this dataset's own block.
    try:
        report = check_integrity(root, dataset=ds)
        (run_dir / f"integrity.{name}.json").write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{name}][validate] integrity OK "
              f"(warnings={len(report.warnings)})")
    except IntegrityError as exc:
        print(f"[{name}][validate] INTEGRITY FAILED:\n{exc}")
        failures.append(name)

if failures:
    print(f"\n[validate] FAILED datasets: {failures}")
    sys.exit(1)
print("\n[validate] all datasets OK")
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
