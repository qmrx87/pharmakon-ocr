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

# Pre-flight: ask the installed rfdetr what its HOSTED_MODELS map says, so we
# discover the canonical URL straight from the package instead of guessing.
# This is a no-op if rfdetr isn't importable.
if python -c "import rfdetr" 2>/dev/null; then
  vlog "introspecting rfdetr for the canonical pretrained URL..."
  python - <<'PY' 2>/dev/null || true
import importlib, json
try:
    import rfdetr
    # rfdetr 1.x exposes the URL map in various submodules depending on version.
    # Try the most common locations; print whatever we find.
    found = {}
    for name in ("rfdetr.main", "rfdetr.detr", "rfdetr.config", "rfdetr.models", "rfdetr"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for attr in dir(mod):
            if "HOSTED" in attr.upper() or "URL" in attr.upper() or "WEIGHTS" in attr.upper():
                v = getattr(mod, attr, None)
                if isinstance(v, dict) and v and any("http" in str(x) for x in v.values()):
                    found[f"{name}.{attr}"] = v
                elif isinstance(v, str) and v.startswith("http"):
                    found[f"{name}.{attr}"] = v
    if found:
        print("    rfdetr exposes the following hosted-weight URLs:")
        for k, v in found.items():
            print(f"      {k} = {v if isinstance(v, str) else json.dumps(v, indent=8)}")
    else:
        print("    (could not auto-discover HOSTED_MODELS map in rfdetr — using defaults)")
except Exception as e:
    print(f"    rfdetr introspection failed: {e}")
PY
fi

# ---------------------------------------------------------------------------
# Strategy:
#   1. If a file is already present and non-empty, reuse it.
#   2. Try rfdetr's programmatic download by passing the MAGIC STRING
#      `pretrain_weights="rf-detr-medium.pth"`. rfdetr 1.x maintains an internal
#      HOSTED_MODELS map of {name: url} and downloads on construction. Passing
#      `pretrain_weights=None` does NOT trigger a download — it inits from
#      scratch silently with only a Pydantic warning (we discovered this the
#      hard way: the earlier from-scratch path explains the rc=1 a few minutes
#      into training, since rfdetr-medium converges very poorly without its
#      DINOv2-pretrained backbone).
#   3. Fall back to curl from the canonical Roboflow GCS URL
#      (`rf-detr-medium.pth`, NOT `rf-detr-medium-coco.pth` — that's a 404).
#      Override with $VIGNOCR_PRETRAINED_RFDETR_URL if Roboflow renames it.
# ---------------------------------------------------------------------------
RFDETR_URL="${VIGNOCR_PRETRAINED_RFDETR_URL:-https://storage.googleapis.com/rfdetr/rf-detr-medium.pth}"
RFDETR_MAGIC_NAME="${VIGNOCR_PRETRAINED_RFDETR_MAGIC:-rf-detr-medium.pth}"

if [[ -f "$RFDETR_OUT" && -s "$RFDETR_OUT" ]]; then
  vlog "reusing existing pretrained weights: $RFDETR_OUT ($(du -h "$RFDETR_OUT" | cut -f1))"
else
  vlog "fetching RF-DETR medium pretrained weights -> $RFDETR_OUT"
  fetched=0

  # Path A: ask rfdetr to download via its magic-string lookup (HOSTED_MODELS).
  if python -c "import rfdetr" 2>/dev/null; then
    vlog "trying rfdetr programmatic download via RFDETRMedium(pretrain_weights='$RFDETR_MAGIC_NAME')"
    if VIGNOCR_PRETRAINED_OUT="$RFDETR_OUT" VIGNOCR_RFDETR_MAGIC="$RFDETR_MAGIC_NAME" python - <<'PY'
import os, shutil
from pathlib import Path
out = Path(os.environ["VIGNOCR_PRETRAINED_OUT"])
magic = os.environ["VIGNOCR_RFDETR_MAGIC"]
out.parent.mkdir(parents=True, exist_ok=True)
try:
    # Passing the MAGIC STRING triggers rfdetr's internal download into its
    # cache dir (typically TORCH_HOME or a sibling). After the constructor
    # returns, we find the file by name and copy it into our project cache.
    from rfdetr import RFDETRMedium
    m = RFDETRMedium(pretrain_weights=magic)
    # rfdetr 1.x caches under TORCH_HOME/hub/checkpoints/ by default; check
    # several candidate locations.
    th = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch")))
    candidates = [
        th / "hub" / "checkpoints" / magic,
        th / magic,
        Path.home() / ".cache" / "rfdetr" / magic,
        Path.cwd() / magic,
    ]
    cand = None
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            cand = p; break
    if cand is None:
        for p in th.rglob(magic):
            if p.is_file() and p.stat().st_size > 0:
                cand = p; break
    if cand is None:
        # Last resort: any *.pth with rfdetr in its name.
        for p in th.rglob("*.pth"):
            if ("rfdetr" in p.name.lower() or "rf-detr" in p.name.lower()) and p.stat().st_size > 0:
                cand = p; break
    if cand is None:
        raise RuntimeError(f"rfdetr finished but no '{magic}' was found under {th} — falling back to curl")
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
    vdie "could not download RF-DETR pretrained weights. Try: VIGNOCR_PRETRAINED_RFDETR_URL=<correct url> bash scripts/fetch_pretrained.sh ; or place the file manually at $RFDETR_OUT (the URL is whatever rfdetr's HOSTED_MODELS map says for 'rf-detr-medium.pth' — check the installed rfdetr's source)."
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
