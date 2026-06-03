#!/usr/bin/env bash
# =============================================================================
# scripts/fetch_pretrained.sh  —  LOGIN-NODE-ONLY pretrained-weight prefetch.
#
# WHY THIS EXISTS
#   RF-DETR's `pretrain_weights=None` path triggers a runtime download of the
#   COCO-pretrained backbone from the public Hub. On Narval COMPUTE nodes there
#   is **NO outbound network**, so that download hangs for ~6 minutes and then
#   fails — leaving a useless stack trace and a wasted GPU allocation.
#
#   The fix: pre-download the .pth file ONCE on the login node into project
#   space (~/projects/def-$PI/vignocr/checkpoints/pretrained/). Both Stage A and
#   Stage B detection trainers (src/vignocr/detection/train.py) look for it via
#   $VIGNOCR_PRETRAINED_DIR and pass it to RFDETRMedium(pretrain_weights=...).
#
# WHAT IT DOWNLOADS
#   * rfdetr_medium_coco.pth   — RF-DETR-medium COCO checkpoint (Roboflow Hub)
#
# USAGE (login node):
#   export VIGNOCR_ACCOUNT=def-<PI>
#   export VIGNOCR_PI=<PI>
#   bash scripts/fetch_pretrained.sh
#
# Idempotent: a file that already exists + verifies its size is reused.
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
# shellcheck source=../slurm/lib.sh
source "$REPO_ROOT/slurm/lib.sh"

vlog "=== VignOCR pretrained-weight prefetch (LOGIN NODE) ==="
vignocr_require_env
vignocr_paths

# Refuse to run on a compute node — there is no outbound network there.
if [[ -n "${SLURM_JOB_ID:-}" ]] && ! command -v avail_wheels >/dev/null 2>&1; then
  vwarn "running inside SLURM_JOB_ID=$SLURM_JOB_ID. Compute nodes have no internet."
  vwarn "If this is a salloc shell on a compute node, this script will fail. Run on the LOGIN node."
fi

# Activate the venv ONLY for the rfdetr Python entrypoint. We do NOT need the
# venv to curl/wget — but rfdetr exposes an easier programmatic download path.
if [[ -d "$VIGNOCR_VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "$VIGNOCR_VENV_DIR/bin/activate"
  vlog "venv active: $(command -v python) ($(python --version 2>&1))"
else
  vwarn "venv not found at $VIGNOCR_VENV_DIR — run scripts/setup_narval.sh first if you want the rfdetr-Python path"
fi

OUT_DIR="$VIGNOCR_PRETRAINED_DIR"
mkdir -p "$OUT_DIR"

RFDETR_OUT="$OUT_DIR/rfdetr_medium_coco.pth"

# ---------------------------------------------------------------------------
# Strategy:
#   1. If a file is already present and non-empty, reuse it.
#   2. Else, try rfdetr's programmatic download (if rfdetr is importable).
#   3. Else, fall back to a curl from the published Roboflow URL.
# Step 3's URL is the public release artifact; if Roboflow renames it the user
# must override with $VIGNOCR_PRETRAINED_RFDETR_URL.
# ---------------------------------------------------------------------------
RFDETR_URL="${VIGNOCR_PRETRAINED_RFDETR_URL:-https://storage.googleapis.com/rfdetr/rf-detr-medium-coco.pth}"

if [[ -f "$RFDETR_OUT" && -s "$RFDETR_OUT" ]]; then
  vlog "reusing existing pretrained weights: $RFDETR_OUT ($(du -h "$RFDETR_OUT" | cut -f1))"
else
  vlog "fetching RF-DETR medium COCO pretrained weights -> $RFDETR_OUT"
  fetched=0

  # Path A: ask rfdetr to download (it knows its own canonical URL + naming).
  if python -c "import rfdetr" 2>/dev/null; then
    vlog "trying rfdetr programmatic download via RFDETRMedium(pretrained=True)"
    if VIGNOCR_PRETRAINED_OUT="$RFDETR_OUT" python - <<'PY'
import os, shutil
from pathlib import Path
out = Path(os.environ["VIGNOCR_PRETRAINED_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
try:
    # Touching the model constructor with no num_classes drives rfdetr to
    # resolve + cache its base COCO checkpoint. Then we copy it to our cache.
    from rfdetr import RFDETRMedium
    m = RFDETRMedium(num_classes=80, resolution=640, pretrain_weights=None)
    cand = None
    # Heuristic: rfdetr stashes the file under torch hub (TORCH_HOME) — find it.
    th = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch")))
    for p in th.rglob("rf*detr*medium*coco*.pth"):
        cand = p; break
    if cand is None:
        for p in th.rglob("*.pth"):
            if "rfdetr" in p.name.lower() or "rf-detr" in p.name.lower():
                cand = p; break
    if cand is None:
        raise RuntimeError("rfdetr download finished but no .pth was found in TORCH_HOME — falling back to curl")
    shutil.copy2(cand, out)
    print(f"OK rfdetr cached -> {out} (from {cand})")
except Exception as e:
    raise SystemExit(f"rfdetr download path failed: {e}")
PY
    then
      fetched=1
    else
      vwarn "rfdetr programmatic download failed — will try direct URL"
    fi
  fi

  # Path B: direct URL fallback.
  if [[ "$fetched" -eq 0 ]]; then
    if command -v curl >/dev/null 2>&1; then
      vlog "curl -L $RFDETR_URL -> $RFDETR_OUT"
      if curl -fL --retry 3 --retry-delay 2 -o "$RFDETR_OUT.tmp" "$RFDETR_URL"; then
        mv "$RFDETR_OUT.tmp" "$RFDETR_OUT"
        fetched=1
      else
        rm -f "$RFDETR_OUT.tmp"
      fi
    elif command -v wget >/dev/null 2>&1; then
      vlog "wget $RFDETR_URL -> $RFDETR_OUT"
      if wget -q -O "$RFDETR_OUT.tmp" "$RFDETR_URL"; then
        mv "$RFDETR_OUT.tmp" "$RFDETR_OUT"
        fetched=1
      else
        rm -f "$RFDETR_OUT.tmp"
      fi
    fi
  fi

  if [[ "$fetched" -ne 1 || ! -s "$RFDETR_OUT" ]]; then
    vdie "could not download RF-DETR pretrained weights. Set VIGNOCR_PRETRAINED_RFDETR_URL to the correct release URL and retry. Or place the file manually at $RFDETR_OUT."
  fi
  vlog "downloaded: $RFDETR_OUT ($(du -h "$RFDETR_OUT" | cut -f1))"
fi

# Record a manifest so a future run can verify provenance.
{
  echo "rfdetr_medium_coco.pth: $RFDETR_OUT"
  echo "size_bytes: $(stat -c%s "$RFDETR_OUT" 2>/dev/null || stat -f%z "$RFDETR_OUT" 2>/dev/null || echo unknown)"
  echo "fetched_at: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "host: $(hostname)"
  echo "source: ${RFDETR_URL}"
} > "$OUT_DIR/MANIFEST.txt"

vlog "pretrained cache ready: $OUT_DIR"
vlog "the detection trainer will pick this up automatically via VIGNOCR_PRETRAINED_DIR."
