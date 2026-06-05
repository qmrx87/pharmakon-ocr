#!/usr/bin/env bash
# =============================================================================
# scripts/fix_install.sh  —  one-shot recovery: install missing torch lazy deps
# + fetch pretrained weights + verify the import chain.
#
# Run this on the LOGIN node when training jobs crash with
# `ModuleNotFoundError` for a torch transitive dep (sympy / networkx / etc.).
#
# What it does (idempotent — safe to run any time):
#   1. Activate the venv at $VIGNOCR_VENV_DIR.
#   2. `pip install --no-index` torch's lazy deps (sympy/networkx/jinja2/
#      filelock/fsspec/typing_extensions). The torch wheel doesn't always
#      declare these as hard requirements, so they get skipped by --no-index
#      installs and only crash at TRAINING time inside torch._dynamo.
#   3. Verify the full ML import chain on CPU.
#   4. Cache the RF-DETR COCO pretrained weights via scripts/fetch_pretrained.sh.
#
# After this script completes successfully, `bash slurm/submit_all.sh` will
# resume training where it stopped (with VIGNOCR_RESUME_RUN_DIR if you set it).
# =============================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
# shellcheck source=../slurm/lib.sh
source "$REPO_ROOT/slurm/lib.sh"

vlog "=== VignOCR install fix-up ==="
vignocr_require_env
vignocr_paths
vignocr_load_modules

# 1. Activate the venv
if [[ ! -d "$VIGNOCR_VENV_DIR" ]]; then
  vdie "venv not found at $VIGNOCR_VENV_DIR — run scripts/setup_narval.sh first"
fi
# shellcheck disable=SC1091
source "$VIGNOCR_VENV_DIR/bin/activate"
vlog "venv: $(command -v python) ($(python --version 2>&1))"

# 2. Install torch's lazy transitive deps from the wheelhouse.
vlog "installing torch lazy deps (sympy + friends) from the Compute Canada wheelhouse"
python -m pip install --no-index --upgrade \
    sympy networkx jinja2 filelock fsspec typing_extensions \
  || vdie "could not install torch lazy deps from the wheelhouse. Run 'avail_wheels sympy' to confirm availability."

# 2b. Install the RF-DETR TRAINING stack (lazy-imported inside model.train()).
#     This is the module that crashed the latest runs: "No module named
#     'pytorch_lightning'". `import rfdetr` succeeds without it, so it must be
#     installed + verified explicitly.
vlog "installing RF-DETR training stack (pytorch_lightning/torchmetrics/lightning-utilities)"
python -m pip install --no-index --upgrade \
    pytorch_lightning torchmetrics lightning_utilities \
  || vdie "could not install the Lightning training stack from the wheelhouse. Run 'avail_wheels pytorch_lightning torchmetrics lightning_utilities'."

# 3. Verify the full ML import chain — the exact path that crashed at training,
#    INCLUDING the lazily-imported training submodule rfdetr.training.
vlog "verifying ML import chain (incl. rfdetr.training → pytorch_lightning)"
python - <<'PY' || vdie "ML import chain still broken — see traceback above. Install the missing module(s) with: pip install --no-index <module>"
import importlib, sys
chain = ["sympy", "networkx", "jinja2", "filelock", "fsspec",
         "torch", "torch._dynamo", "torchvision", "torchvision.ops",
         "rfdetr", "pytorch_lightning", "torchmetrics", "rfdetr.training"]
fail = []
for m in chain:
    try:
        importlib.import_module(m)
        print(f"    OK  {m}")
    except Exception as e:
        print(f"    FAIL {m}: {e.__class__.__name__}: {e}")
        fail.append(m)
if fail:
    print(f"\n  !! {len(fail)} module(s) failed: {fail}")
    sys.exit(1)
print("\n  ✓ import chain healthy — training is unblocked")
PY
vlog "ML import chain healthy"

# 4. Cache the RF-DETR COCO pretrained weights (idempotent — skips if present).
if [[ -f "$VIGNOCR_PRETRAINED_DIR/rfdetr_medium_coco.pth" && -s "$VIGNOCR_PRETRAINED_DIR/rfdetr_medium_coco.pth" ]]; then
  vlog "pretrained weights already cached at $VIGNOCR_PRETRAINED_DIR/rfdetr_medium_coco.pth"
else
  vlog "fetching RF-DETR pretrained weights"
  bash "$SCRIPT_DIR/fetch_pretrained.sh" \
    || vwarn "fetch_pretrained.sh failed — training jobs will refuse to start until the weights are cached"
fi

# Final status report.
echo ""
vlog "=== fix-up complete ==="
vlog "  venv          : $VIGNOCR_VENV_DIR"
vlog "  pretrained    : $VIGNOCR_PRETRAINED_DIR/rfdetr_medium_coco.pth ($(du -h "$VIGNOCR_PRETRAINED_DIR/rfdetr_medium_coco.pth" 2>/dev/null | cut -f1 || echo 'MISSING'))"
vlog ""
vlog "  Next: resubmit the DAG (the trap fix + this install fix unblock everything):"
vlog "    bash slurm/submit_all.sh"
vlog ""
vlog "  Or resume the previously-failed Stage A run in place:"
vlog "    export VIGNOCR_RESUME_RUN_DIR=$VIGNOCR_RUNS_DIR/02a_train_vignette/latest"
vlog "    sbatch --account=$VIGNOCR_ACCOUNT --chdir=$VIGNOCR_REPO_ROOT --export=ALL slurm/02a_train_vignette.sbatch"
