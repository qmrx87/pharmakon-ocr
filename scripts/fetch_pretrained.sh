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

# Load cluster modules so that Python and system libraries are loaded before the venv is activated.
vignocr_load_modules

if [[ -d "$VIGNOCR_VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "$VIGNOCR_VENV_DIR/bin/activate"
  vlog "venv active: $(command -v python) ($(python --version 2>&1))"
else
  vwarn "venv not found at $VIGNOCR_VENV_DIR — run scripts/setup_narval.sh first if you want the rfdetr-Python path"
fi

OUT_DIR="$VIGNOCR_PRETRAINED_DIR"
mkdir -p "$OUT_DIR"

# Which RF-DETR size to fetch (medium = v1 default; v2 may use nano/small).
#   VIGNOCR_RFDETR_SIZE=nano bash scripts/fetch_pretrained.sh
# The trainer resolves the cache per-variant (rfdetr_<size>_coco.pth), so fetch
# once per size you train.
RFDETR_SIZE="${VIGNOCR_RFDETR_SIZE:-medium}"
RFDETR_OUT="$OUT_DIR/rfdetr_${RFDETR_SIZE}_coco.pth"

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
RFDETR_URL="${VIGNOCR_PRETRAINED_RFDETR_URL:-https://storage.googleapis.com/rfdetr/rf-detr-${RFDETR_SIZE}.pth}"
RFDETR_MAGIC_NAME="${VIGNOCR_PRETRAINED_RFDETR_MAGIC:-rf-detr-${RFDETR_SIZE}.pth}"

# rfdetr's CANONICAL cache location (discovered the hard way: rfdetr 1.x
# downloads its hosted models into ~/.roboflow/models/<name>.pth, NOT into
# TORCH_HOME). Always check this path FIRST — if rfdetr ever pulled the file
# (e.g. during a previous successful construction), it's sitting right here.
ROBOFLOW_CACHE="${ROBOFLOW_MODELS_DIR:-$HOME/.roboflow/models}"
if [[ -f "$ROBOFLOW_CACHE/$RFDETR_MAGIC_NAME" && -s "$ROBOFLOW_CACHE/$RFDETR_MAGIC_NAME" ]]; then
  vlog "found pretrained weights in rfdetr's own cache: $ROBOFLOW_CACHE/$RFDETR_MAGIC_NAME"
  mkdir -p "$(dirname "$RFDETR_OUT")"
  cp -f "$ROBOFLOW_CACHE/$RFDETR_MAGIC_NAME" "$RFDETR_OUT"
  vlog "copied -> $RFDETR_OUT ($(du -h "$RFDETR_OUT" | cut -f1))"
  # Record provenance + bail out (we're done).
  {
    echo "rfdetr_medium_coco.pth: $RFDETR_OUT"
    echo "size_bytes: $(stat -c%s "$RFDETR_OUT" 2>/dev/null || stat -f%z "$RFDETR_OUT" 2>/dev/null || echo unknown)"
    echo "fetched_at: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "host: $(hostname)"
    echo "source: $ROBOFLOW_CACHE/$RFDETR_MAGIC_NAME (rfdetr internal cache)"
  } > "$OUT_DIR/MANIFEST.txt"
  vlog "pretrained cache ready: $OUT_DIR"
  vlog "the detection trainer will pick this up automatically via VIGNOCR_PRETRAINED_DIR."
  exit 0
fi

if [[ -f "$RFDETR_OUT" && -s "$RFDETR_OUT" ]]; then
  vlog "reusing existing pretrained weights: $RFDETR_OUT ($(du -h "$RFDETR_OUT" | cut -f1))"
else
  vlog "fetching RF-DETR medium pretrained weights -> $RFDETR_OUT"
  fetched=0

  # Path A: ask rfdetr to download via its magic-string lookup (HOSTED_MODELS).
  # We try THREE inner strategies in sequence, all inside one Python invocation:
  #   A1: construct RFDETRMedium(pretrain_weights=MAGIC) → find the .pth rfdetr
  #       just cached anywhere under TORCH_HOME / ~/.cache / CWD.
  #   A2: construct RFDETRMedium() with NO args (relies on rfdetr's default
  #       download path) → find the cached file.
  #   A3: dump the constructed model's state_dict directly via torch.save. This
  #       is the guaranteed-success path: as long as RFDETRMedium initialises
  #       AT ALL, we get a loadable .pth (rfdetr accepts either full ckpt or
  #       state_dict). It won't have optimizer/EMA state, but for a FRESH
  #       fine-tune (which is what Stage A/B do) only the model weights matter.
  if python -c "import rfdetr" 2>/dev/null; then
    vlog "trying rfdetr programmatic download (3 inner strategies)"
    if VIGNOCR_PRETRAINED_OUT="$RFDETR_OUT" VIGNOCR_RFDETR_MAGIC="$RFDETR_MAGIC_NAME" VIGNOCR_RFDETR_SIZE="$RFDETR_SIZE" python - <<'PY'
import os, shutil, sys, traceback
from pathlib import Path

out = Path(os.environ["VIGNOCR_PRETRAINED_OUT"])
magic = os.environ["VIGNOCR_RFDETR_MAGIC"]
out.parent.mkdir(parents=True, exist_ok=True)


def _candidate_dirs():
    # rfdetr's REAL cache directory (1.x writes here, not under TORCH_HOME).
    yield Path(os.environ.get("ROBOFLOW_MODELS_DIR", str(Path.home() / ".roboflow" / "models")))
    th = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch")))
    yield th / "hub" / "checkpoints"
    yield th
    yield Path.home() / ".cache" / "rfdetr"
    yield Path.home() / ".cache" / "huggingface" / "hub"
    yield Path.cwd()


def _find_cached(magic_name):
    """Return the path of the most-recently-modified .pth that matches magic_name
    or rf-detr-medium under any candidate dir."""
    matches = []
    for d in _candidate_dirs():
        if not d.is_dir():
            continue
        # Exact filename
        p = d / magic_name
        if p.is_file() and p.stat().st_size > 0:
            matches.append(p)
        # Glob anywhere under d
        try:
            for p in d.rglob("*.pth"):
                n = p.name.lower()
                if ("rfdetr" in n or "rf-detr" in n or "rf_detr" in n) and p.stat().st_size > 0:
                    matches.append(p)
        except OSError:
            pass
    if not matches:
        return None
    # Newest mtime wins.
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _save_state_dict(model_obj, out_path):
    """Last-resort: dump the constructed model's state_dict to disk.

    rfdetr's RFDETRMedium wraps a `ModelContext` (NOT a torch.nn.Module). The
    actual torch.nn.Module lives several levels deep. We walk likely attribute
    paths to find an object that has `.state_dict()` — typically:
        m -> m.model              # ModelContext
        m -> m.model.model        # the inner DETR
        m -> m.model.model.module # if DataParallel-wrapped
    """
    import torch

    def _find_nn_module(obj, depth=0, max_depth=5):
        if depth > max_depth:
            return None
        # An object whose state_dict() returns a non-empty dict and which has
        # actual parameters is what we want.
        sd_fn = getattr(obj, "state_dict", None)
        if callable(sd_fn):
            try:
                sd = sd_fn()
                if isinstance(sd, dict) and len(sd) > 0:
                    return obj
            except Exception:
                pass
        # Probe likely wrapper attributes.
        for attr in ("model", "module", "net", "backbone_with_pos_enc",
                     "_orig_model", "transformer", "detr"):
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                found = _find_nn_module(child, depth + 1, max_depth)
                if found is not None:
                    return found
        return None

    mod = _find_nn_module(model_obj)
    if mod is None:
        raise AttributeError(
            f"could not find a torch.nn.Module under {type(model_obj).__name__}; "
            "walked .model / .module / .net / .backbone_with_pos_enc / ..."
        )
    sd = mod.state_dict()
    ckpt = {"model": sd}
    torch.save(ckpt, str(out_path))
    return out_path


import rfdetr

# Pick the wrapper class for the requested size (medium = v1 default).
_size = os.environ.get("VIGNOCR_RFDETR_SIZE", "medium").lower()
_cls_name = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
             "base": "RFDETRBase", "large": "RFDETRLarge"}.get(_size, "RFDETRMedium")
ModelCls = getattr(rfdetr, _cls_name, None)
if ModelCls is None:
    print(f"  installed rfdetr does not export {_cls_name}; falling back to RFDETRMedium")
    from rfdetr import RFDETRMedium as ModelCls

errors = []

# A1: explicit magic-string download
try:
    print(f"  [A1] {ModelCls.__name__}(pretrain_weights={magic!r})")
    m = ModelCls(pretrain_weights=magic)
    cand = _find_cached(magic)
    if cand:
        shutil.copy2(cand, out)
        print(f"  [A1] OK: copied {cand} -> {out}")
        sys.exit(0)
    print("  [A1] no cached file found after construction; trying A2")
except Exception as e:
    errors.append(("A1", e, traceback.format_exc()))
    print(f"  [A1] error: {e}")

# A2: no-arg construction (rfdetr default pretrained behaviour)
try:
    print(f"  [A2] {ModelCls.__name__}()  # no args → default pretrained download")
    m = ModelCls()
    cand = _find_cached(magic)
    if cand:
        shutil.copy2(cand, out)
        print(f"  [A2] OK: copied {cand} -> {out}")
        sys.exit(0)
    print("  [A2] still no cached file found; trying A3 (state_dict save)")
except Exception as e:
    errors.append(("A2", e, traceback.format_exc()))
    print(f"  [A2] error: {e}")

# A3: save the constructed model's state_dict directly. The model object from
# A1 or A2 will be in `m` if either succeeded. Otherwise construct from scratch
# (which gives random init — still useful for a runs-end-to-end smoke test).
try:
    print("  [A3] torch.save(model.state_dict(), out)  # guaranteed-success path")
    if "m" not in locals() or m is None:
        m = ModelCls(pretrain_weights=None)
    _save_state_dict(m, out)
    print(f"  [A3] OK: wrote {out} ({out.stat().st_size} bytes)")
    sys.exit(0)
except Exception as e:
    errors.append(("A3", e, traceback.format_exc()))
    print(f"  [A3] error: {e}")

print("\nALL APPROACHES FAILED. Captured errors:")
for label, e, tb in errors:
    print(f"\n--- {label} ---\n{tb}")
sys.exit(1)
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
