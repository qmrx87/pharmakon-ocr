#!/usr/bin/env bash
# =============================================================================
# scripts/fetch_pretrained_v2.sh — LOGIN-NODE prefetch for the V2 variants.
#
# Compute nodes have NO internet; every pretrained artefact must be cached on
# the login node first (same pattern as scripts/fetch_pretrained.sh for v1):
#
#   1. Donut base (v2a)      -> $HF_HOME (lib.sh points it into
#                               $VIGNOCR_PRETRAINED_DIR/huggingface)
#   2. docTR det + PARSeq rec (v2b + the VLM dataset builder)
#                            -> $DOCTR_CACHE_DIR ($VIGNOCR_PRETRAINED_DIR/doctr)
#   3. RF-DETR nano COCO (v2 shared vignette cropper)
#                            -> $VIGNOCR_PRETRAINED_DIR (via fetch_pretrained.sh)
#
# USAGE (login node):
#   export VIGNOCR_ACCOUNT=def-khenni VIGNOCR_PI=khenni
#   bash scripts/fetch_pretrained_v2.sh
#
# Idempotent: HF + docTR both skip files already in cache.
# =============================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
# shellcheck source=../slurm/lib.sh
source "$REPO_ROOT/slurm/lib.sh"

vlog "=== VignOCR V2 pretrained prefetch (LOGIN NODE) ==="
vignocr_require_env
vignocr_paths

# Load cluster modules so that Python and system libraries (like OpenCV) are loaded before the venv is activated.
vignocr_load_modules

# The fetch itself must be ONLINE — undo lib.sh's offline-by-default stance for
# THIS process only (compute jobs still get HF_HUB_OFFLINE=1 from lib.sh).
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

if [[ -d "$VIGNOCR_VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "$VIGNOCR_VENV_DIR/bin/activate"
  vlog "venv active: $(command -v python) ($(python --version 2>&1))"
else
  vdie "venv not found at $VIGNOCR_VENV_DIR — run scripts/setup_narval.sh first"
fi

# Make sure the v2 deps are present (transformers / sentencepiece / doctr).
# The canonical installer is scripts/setup_narval.sh; this is a belt-and-braces
# top-up. CRITICAL: the online fallback must NEVER replace the Compute Canada
# CUDA torch with a generic PyPI wheel, so doctr's online path uses --no-deps
# (torch/torchvision are already installed from [ml]).
TORCH_CUDA_BEFORE="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo none)"
vlog "ensuring [v2] deps (transformers/sentencepiece/python-doctr) — wheelhouse first"
python -m pip install --no-index transformers sentencepiece \
  || python -m pip install transformers sentencepiece \
  || vwarn "transformers/sentencepiece install incomplete"
if ! python -m pip install --no-index "python-doctr[torch]" 2>/dev/null; then
  vwarn "python-doctr not in wheelhouse; online install with --no-deps (protects CUDA torch)"
  python -m pip install --no-deps python-doctr \
    || vwarn "python-doctr online install failed — v2b unavailable"
  python -m pip install --no-index pypdfium2 anyascii langdetect defusedxml h5py shapely scipy tqdm \
    || vwarn 'some docTR deps missing from the wheelhouse — import doctr will name them'
fi
TORCH_CUDA_AFTER="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo none)"
if [[ "$TORCH_CUDA_BEFORE" != "none" && "$TORCH_CUDA_AFTER" == "none" ]]; then
  vdie "a v2 install REPLACED the CUDA torch with a CPU build (cuda: ${TORCH_CUDA_BEFORE} -> none). Recreate the venv: bash scripts/setup_narval.sh --recreate"
fi

DONUT_BASE="${VIGNOCR_DONUT_BASE:-naver-clova-ix/donut-base}"

vlog "1/3 prefetch Donut base ($DONUT_BASE) -> $HF_HOME"
python - "$DONUT_BASE" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1])
print(f"    cached: {path}")
PY

vlog "2/3 prefetch docTR weights (det=db_mobilenet_v3_large, rec=parseq) -> ${DOCTR_CACHE_DIR}"
python - <<'PY'
# Constructing the predictors triggers docTR's own download-into-cache.
from doctr.models import ocr_predictor, recognition_predictor
ocr_predictor(det_arch="db_mobilenet_v3_large", reco_arch="parseq", pretrained=True)
recognition_predictor("parseq", pretrained=True)
print("    docTR det+rec cached")
PY

vlog "3/3 prefetch RF-DETR nano COCO weights (v2 vignette cropper)"
VIGNOCR_RFDETR_SIZE=nano bash "$SCRIPT_DIR/fetch_pretrained.sh" \
  || vwarn "nano fetch failed — the v2 cropper training will refuse to start until cached"

vlog "=== V2 prefetch complete ==="
vlog "  HF cache    : $HF_HOME"
vlog "  docTR cache : $DOCTR_CACHE_DIR"
vlog "  rfdetr nano : $VIGNOCR_PRETRAINED_DIR/rfdetr_nano_coco.pth"
vlog "Submit the v2 DAG next:  bash slurm/submit_v2.sh"
