#!/usr/bin/env bash
# =============================================================================
# scripts/setup_narval.sh  —  ONE-TIME (login-node) bootstrap for VignOCR on
# the Narval cluster (Digital Research Alliance of Canada / Calcul Québec).
#
# WHAT IT DOES
#   1. Validates the env contract (VIGNOCR_ACCOUNT / VIGNOCR_PI).
#   2. Loads the Narval module stack (StdEnv + python + cuda).
#   3. Creates a Python venv in ~/scratch (large, ephemeral tier).
#   4. Installs the pinned deps OFFLINE from the Alliance wheelhouse
#      (--no-index), reading the pins from pyproject.toml ([ml] + core).
#   5. Places / verifies the dataset under ~/projects/def-$VIGNOCR_PI/.
#   6. Sanity-checks GPU visibility (nvidia-smi) and dataset visibility.
#   7. Prints the next steps (how to submit the DAG).
#
# WHY THE LOGIN NODE
#   Narval COMPUTE nodes have NO internet. pip therefore cannot reach PyPI from
#   inside a job. The Alliance ships a curated **wheelhouse** mounted via CVMFS;
#   `pip install --no-index ...` resolves wheels from there. Building the venv on
#   the LOGIN node (which can read the wheelhouse and has a little connectivity
#   for the index metadata) means compute jobs only ever *activate* it.
#
#   THE OFFLINE PATTERN (memorize this):
#       module load StdEnv/2023 python/3.11
#       avail_wheels "torch"            # discover what the wheelhouse carries
#       virtualenv --no-download $VENV  # build venv from the local python, no PyPI
#       source $VENV/bin/activate
#       pip install --no-index --upgrade pip
#       pip install --no-index -e .[ml] # all wheels resolved from the wheelhouse
#   If a pinned version is NOT in the wheelhouse, `avail_wheels <pkg>` shows the
#   nearest available version; adjust the pin in pyproject.toml (coordinate with
#   the team) or `module load` a provided build (e.g. opencv, arrow).
#
# USAGE
#   export VIGNOCR_ACCOUNT=def-<PI>      # e.g. def-smith
#   export VIGNOCR_PI=<PI>               # e.g. smith
#   bash scripts/setup_narval.sh         # run on a Narval LOGIN node
#
# This script is idempotent: re-running it reuses the venv (pass --recreate to
# rebuild from scratch) and re-verifies the dataset.
# =============================================================================

set -Eeuo pipefail

# Resolve repo root from this script's location (scripts/ -> repo root is parent).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# Reuse the shared library for env validation, paths, modules, and logging.
# shellcheck source=../slurm/lib.sh
source "$REPO_ROOT/slurm/lib.sh"

RECREATE=0
for arg in "$@"; do
  case "$arg" in
    --recreate) RECREATE=1 ;;
    -h|--help)
      sed -n '2,46p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) vdie "unknown arg: $arg (try --recreate or --help)" ;;
  esac
done

# --------------------------------------------------------------------------- #
vlog "=== VignOCR Narval setup (login node) ==="
vignocr_require_env
vignocr_paths
vlog "account=$VIGNOCR_ACCOUNT  PI=$VIGNOCR_PI"
vlog "repo_root=$VIGNOCR_REPO_ROOT"
vlog "project_dir=$VIGNOCR_PROJECT_DIR  (shared, backed-up — dataset lives here)"
vlog "scratch_dir=$VIGNOCR_SCRATCH_DIR  (large, ephemeral — venv/checkpoints/logs/runs)"

# --------------------------------------------------------------------------- #
# Guard: this is meant for the LOGIN node. Compute nodes have no internet and
# cannot see the wheelhouse index. We don't hard-fail (some sites run setup in
# an salloc shell), but we warn loudly.
# --------------------------------------------------------------------------- #
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  vwarn "running inside a Slurm allocation (SLURM_JOB_ID=$SLURM_JOB_ID)."
  vwarn "If this is a COMPUTE node, pip --no-index may still work via CVMFS, but"
  vwarn "the documented pattern is to run setup on the LOGIN node. Continuing."
fi

# --------------------------------------------------------------------------- #
# 1. Module stack.
# --------------------------------------------------------------------------- #
vignocr_load_modules

# --------------------------------------------------------------------------- #
# 2. Discover what the Alliance wheelhouse carries for our heavy deps. This is
#    informational — it helps you reconcile pyproject pins with availability.
#    `avail_wheels` exists only on Alliance clusters; skip elsewhere.
# --------------------------------------------------------------------------- #
if command -v avail_wheels >/dev/null 2>&1; then
  vlog "wheelhouse availability (reconcile with pyproject [ml] pins if a pin is missing):"
  for pkg in torch torchvision onnxruntime opencv_python_headless pycocotools albumentations; do
    avail_wheels "$pkg" 2>/dev/null | sed 's/^/    /' || true
  done
else
  vwarn "avail_wheels not found (not on an Alliance cluster?) — skipping wheelhouse audit"
fi

# --------------------------------------------------------------------------- #
# 3. Create the venv in scratch (or reuse it). --no-download => build from the
#    module's python, never reach PyPI for the venv bootstrap itself.
# --------------------------------------------------------------------------- #
if [[ "$RECREATE" -eq 1 && -d "$VIGNOCR_VENV_DIR" ]]; then
  vlog "--recreate: removing existing venv at $VIGNOCR_VENV_DIR"
  rm -rf "$VIGNOCR_VENV_DIR"
fi

if [[ -d "$VIGNOCR_VENV_DIR" ]]; then
  vlog "reusing existing venv at $VIGNOCR_VENV_DIR (pass --recreate to rebuild)"
else
  vlog "creating venv at $VIGNOCR_VENV_DIR"
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv --no-download "$VIGNOCR_VENV_DIR"
  else
    # Fallback to stdlib venv if Alliance's virtualenv shim is absent.
    python -m venv "$VIGNOCR_VENV_DIR"
  fi
fi

# shellcheck disable=SC1091
source "$VIGNOCR_VENV_DIR/bin/activate"
vlog "venv active: $(command -v python) ($(python --version 2>&1))"

# --------------------------------------------------------------------------- #
# 4. Install pinned deps OFFLINE. pip reads the pins from pyproject.toml:
#       core deps      -> always
#       [ml] extra     -> torch / rfdetr / onnx / paddleocr / opencv / ...
#    --no-index forces resolution from the Alliance wheelhouse (CVMFS); no PyPI.
# --------------------------------------------------------------------------- #
vlog "upgrading pip/setuptools/wheel from the wheelhouse (--no-index)"
python -m pip install --no-index --upgrade pip setuptools wheel \
  || vwarn "could not upgrade pip from wheelhouse; continuing with the bundled pip"

vlog "installing vignocr core + [ml] extra from the wheelhouse (--no-index, editable)"
if python -m pip install --no-index -e "$VIGNOCR_REPO_ROOT"[ml]; then
  vlog "ml install OK"
else
  vwarn "editable [ml] install via --no-index failed."
  vwarn "Most common cause: a pyproject pin is not in the wheelhouse for this StdEnv."
  vwarn "Diagnose with:  avail_wheels <package>   then adjust the pin in pyproject.toml"
  vwarn "Falling back to a CORE-ONLY install so the CPU stack is at least usable."
  python -m pip install --no-index -e "$VIGNOCR_REPO_ROOT" \
    || vdie "core --no-index install also failed — inspect the wheelhouse with avail_wheels"
fi

# Record the exact resolved environment next to the venv for provenance.
python -m pip freeze > "$VIGNOCR_SCRATCH_DIR/pip_freeze.setup.txt" 2>/dev/null || true
vlog "recorded resolved versions -> $VIGNOCR_SCRATCH_DIR/pip_freeze.setup.txt"

# --------------------------------------------------------------------------- #
# 5. Dataset placement.
#    The vignocr package resolves the dataset root from configs/data.yaml
#    RELATIVE TO THE REPO (on /home). Two cases:
#      • synthetic (default): the fixture is tiny (a few MB) and CPU-generable,
#        so we just generate it into its repo-relative home (fixtures/synthetic);
#        no project-space staging or symlink needed.
#      • real: the Roboflow export is large -> it must live in project space
#        ($VIGNOCR_DATA_ROOT) and we SYMLINK the in-repo path (configs/data.yaml:
#        datasets.real.root, i.e. data/) to it so the config resolver finds it
#        with NO code edit. The user uploads the export first (rsync/scp/Globus).
# --------------------------------------------------------------------------- #
DATA_ACTIVE="${VIGNOCR_DATA_ACTIVE:-synthetic}"
mkdir -p "$VIGNOCR_DATA_ROOT"

if [[ "$DATA_ACTIVE" == "synthetic" ]]; then
  SYN_ROOT="$(vignocr_dataset_link_target)"   # repo-relative fixtures/synthetic
  if [[ -n "$SYN_ROOT" && -n "$(ls -A "$SYN_ROOT" 2>/dev/null || true)" ]]; then
    vlog "synthetic fixture already present at $SYN_ROOT"
  else
    vlog "generating the synthetic fixture (config-driven, no args) -> $SYN_ROOT"
    VIGNOCR_DATA_ACTIVE=synthetic python -m vignocr.data.synthetic \
      || vwarn "synthetic generation failed (run it later: python -m vignocr.data.synthetic)"
  fi
elif [[ -n "$(ls -A "$VIGNOCR_DATA_ROOT" 2>/dev/null || true)" ]]; then
  vlog "real dataset present under $VIGNOCR_DATA_ROOT:"
  ls -1 "$VIGNOCR_DATA_ROOT" | sed 's/^/    /'
  vignocr_link_dataset   # wire <repo>/data -> project-space export
else
  vwarn "active dataset is 'real' but $VIGNOCR_DATA_ROOT is empty."
  vwarn "Stage the real COCO export here (one-time, from your laptop):"
  vwarn "    rsync -avP ./data/  $USER@narval.alliancecan.ca:$VIGNOCR_DATA_ROOT/"
  vwarn "Then re-run this script (it will symlink <repo>/data -> that export)."
fi

# Also stage the nomenclature CSV/xlsx into project space if it ships in-repo.
if [[ -f "$VIGNOCR_REPO_ROOT/fixtures/nomenclature.csv" && ! -f "$VIGNOCR_PROJECT_DIR/nomenclature.csv" ]]; then
  cp "$VIGNOCR_REPO_ROOT/fixtures/nomenclature.csv" "$VIGNOCR_PROJECT_DIR/nomenclature.csv"
  vlog "copied fixture nomenclature.csv -> $VIGNOCR_PROJECT_DIR/"
fi

# --------------------------------------------------------------------------- #
# 6. Sanity checks.
#    (a) GPU: nvidia-smi works on COMPUTE nodes, NOT the login node. If it's
#        missing here that's EXPECTED on login; we tell the user how to verify
#        on a GPU node via salloc.
#    (b) Package import on CPU (must succeed even without a GPU).
#    (c) Dataset visibility from the package's config resolver.
# --------------------------------------------------------------------------- #
vlog "--- sanity checks ---"

if command -v nvidia-smi >/dev/null 2>&1; then
  vlog "nvidia-smi (GPU visible here):"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /' || true
else
  vwarn "nvidia-smi not found — EXPECTED on the login node (no GPU)."
  vwarn "Verify GPU access on a compute node with:"
  vwarn "    salloc --account=$VIGNOCR_ACCOUNT --gres=gpu:a100:1 --cpus-per-task=12 --mem=64G --time=0:20:00"
  vwarn "    nvidia-smi"
fi

vlog "importing vignocr.common on CPU (must work without [ml]):"
if VIGNOCR_REPO_ROOT="$VIGNOCR_REPO_ROOT" python -c "import vignocr; from vignocr.common import get_classes; c=get_classes(); print('    vignocr', vignocr.__version__, '-', c.num_classes, 'classes:', ','.join(c.names))"; then
  vlog "core import OK"
else
  vdie "core import failed — the venv is broken; re-run with --recreate"
fi

vlog "checking torch + CUDA wiring (best-effort; real CUDA check belongs on a GPU node):"
python - <<'PY' 2>/dev/null || vwarn "torch not importable here (fine if [ml] fell back to core-only)"
try:
    import torch
    print(f"    torch {torch.__version__}  cuda_built={torch.version.cuda}  is_available={torch.cuda.is_available()}")
except Exception as e:
    print(f"    torch import skipped: {e}")
PY

# --------------------------------------------------------------------------- #
# 6b. Pre-fetch RF-DETR COCO pretrained weights into VIGNOCR_PRETRAINED_DIR.
#     Compute nodes have NO outbound network — without this cache, the trainer
#     would hang trying to download the backbone from the Hub. Login node has
#     limited HTTPS, so we run the fetch here, once.
# --------------------------------------------------------------------------- #
if python -c "import rfdetr" 2>/dev/null; then
  vlog "pre-fetching RF-DETR COCO pretrained weights -> $VIGNOCR_PRETRAINED_DIR"
  bash "$SCRIPT_DIR/fetch_pretrained.sh" \
    || vwarn "fetch_pretrained.sh failed — Stage A/B training will refuse to start on compute nodes. Re-run it manually before submitting jobs."
else
  vwarn "rfdetr not importable here (the [ml] install probably fell back to core-only)."
  vwarn "Without rfdetr we cannot pre-fetch its pretrained weights. Train jobs will fail."
  vwarn "To force rfdetr from PyPI (login node has limited HTTPS):"
  vwarn "    pip install rfdetr"
  vwarn "Then re-run this script."
fi

# --------------------------------------------------------------------------- #
# 6c. Pre-create the centralized SLURM log tree so the #SBATCH --output paths
#     resolve at submit time (SLURM does not reliably create deep parents).
# --------------------------------------------------------------------------- #
for entry in \
    01_validate \
    02_train_detection \
    02a_train_vignette \
    03_eval_export_detection \
    04_autolabel_ocr \
    04_train_ocr \
    05_finetune_ocr \
    05_eval_ocr \
    06_pipeline_benchmark; do
  mkdir -p "$VIGNOCR_LOGS_DIR/$entry"
done
vlog "centralized SLURM log tree ready: $VIGNOCR_LOGS_DIR"

# --------------------------------------------------------------------------- #
# 7. Next steps.
# --------------------------------------------------------------------------- #
cat >&2 <<EOF

[vignocr] =========================== SETUP COMPLETE ===========================
  venv         : $VIGNOCR_VENV_DIR
  dataset      : $VIGNOCR_DATA_ROOT
  runs/output  : $VIGNOCR_RUNS_DIR   (per-stage, timestamp+gitSHA, resumable)
  pip freeze   : $VIGNOCR_SCRATCH_DIR/pip_freeze.setup.txt

  NEXT — submit the full training DAG (validate -> train -> eval/export -> ocr
  train -> ocr eval -> benchmark), chained with --dependency=afterok:

      export VIGNOCR_ACCOUNT=$VIGNOCR_ACCOUNT
      export VIGNOCR_PI=$VIGNOCR_PI
      bash slurm/submit_all.sh

  Dry-run every job's directives without consuming an allocation:

      bash slurm/submit_all.sh --test-only

  Run a single stage standalone (example: just detection training):

      sbatch --account=\$VIGNOCR_ACCOUNT slurm/02_train_detection.sbatch

  Switch to the real dataset (after annotation) without editing code:

      export VIGNOCR_DATA_ACTIVE=real
[vignocr] ======================================================================
EOF
