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

# Make sure the v2 extra is installed (transformers / sentencepiece / doctr).
vlog "ensuring [v2] deps (transformers/sentencepiece/python-doctr)"
if python -m pip install --no-index transformers sentencepiece python-doctr 2>/dev/null; then
  vlog "v2 deps installed from the wheelhouse"
else
  vlog "wheelhouse install incomplete; attempting online install from PyPI"
  python -m pip install transformers sentencepiece "python-doctr[torch]" \
    || vwarn "wheelhouse and online install incomplete — check: avail_wheels transformers sentencepiece python_doctr"
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
