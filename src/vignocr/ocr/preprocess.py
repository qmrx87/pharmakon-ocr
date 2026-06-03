"""Field-crop preprocessing for recognition.

Two responsibilities:

1. :func:`orient` — rotate a crop so its text is upright *before* recognition.
   The ``vin`` strip fields (``num_lot``, ``date_fab``, ``date_exp``) are printed
   rotated 90° on the vignette; we rotate the crop back to a horizontal,
   left-to-right reading orientation so the recognizer never sees vertical text.

2. :func:`preprocess_for_type` — the field-type-aware pipeline (grayscale,
   deskew, denoise, band-aware contrast, pad + height-normalize) configured per
   field *type* in ``configs/ocr/recognition.yaml: field_types``.

Everything here is **PIL + NumPy**. OpenCV (``cv2``) is an *optional* accelerator
for deskew/denoise — lazy-imported and silently skipped when absent, so this
module imports and runs on CPU without the ``[ml]`` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from vignocr.common import get_logger

log = get_logger(__name__)

# Orientation tokens come from classes.yaml (`orientation:` per class). We treat
# anything that isn't a normal left-to-right field as needing rotation.
_VERTICAL = {"vertical"}
_HORIZONTAL = {"horizontal"}


# --------------------------------------------------------------------------- #
# Orientation correction (the rotated vin fields)
# --------------------------------------------------------------------------- #


def orient(crop: Image.Image, orientation: str, *, ccw: bool = True) -> Image.Image:
    """Return ``crop`` rotated so its text is upright for recognition.

    The ``vin`` strip (``num_lot``/``date_fab``/``date_exp``, and the ``vin``
    region itself) is printed vertically; for those ``orientation == "vertical"``
    we rotate 90° so the text reads left-to-right before OCR. Horizontal fields
    are returned unchanged. Unknown orientations are passed through (with a
    debug log) rather than guessed.

    Args:
        crop: the field crop (RGB or L PIL image) from Stage-1 detection.
        orientation: the field's ``orientation`` from ``configs/classes.yaml``
            (``"horizontal"`` | ``"vertical"`` | ``"diagonal"``).
        ccw: rotate counter-clockwise (default). Vignette ``vin`` text reads
            bottom-to-top, so a 90° CCW rotation makes it left-to-right. Exposed
            as a flag rather than hardcoded so a future real-data inspection can
            flip it without touching call sites.

    Returns:
        The upright crop. ``expand=True`` so no pixels are cropped by the rotate.
    """
    o = (orientation or "").strip().lower()
    if o in _VERTICAL:
        # PIL ``rotate`` is counter-clockwise for positive angles.
        angle = 90 if ccw else -90
        return crop.rotate(angle, expand=True)
    if o not in _HORIZONTAL and o:
        # diagonal (color_band) or unknown — recognition doesn't run on these,
        # but be explicit instead of silently rotating.
        log.debug("orient.passthrough", orientation=o)
    return crop


# --------------------------------------------------------------------------- #
# Horizontal-field enhancement hooks (deskew / denoise / band-aware contrast)
# --------------------------------------------------------------------------- #


def to_grayscale(crop: Image.Image) -> Image.Image:
    """Convert to single-channel L (idempotent)."""
    return crop if crop.mode == "L" else crop.convert("L")


def deskew(crop: Image.Image, *, max_angle: float = 10.0) -> Image.Image:
    """Correct small in-plane skew on a (horizontal) text crop.

    Uses OpenCV's minimum-area-rect estimate when ``cv2`` is available; otherwise
    returns the crop unchanged (deskew is a refinement, not a correctness
    requirement — the detector already delivers axis-aligned crops). The estimate
    is clamped to ``±max_angle`` so a bad estimate never violently rotates text.
    """
    cv2 = _try_cv2()
    if cv2 is None:
        return crop
    gray = np.asarray(to_grayscale(crop))
    # Foreground = dark text on light background -> invert so ink is "on".
    inv = 255 - gray
    coords = np.column_stack(np.where(inv > inv.mean()))
    if coords.shape[0] < 10:
        return crop
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    # minAreaRect angle is in (-90, 0]; normalize to a small correction.
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return crop
    arr = np.asarray(crop)
    h, w = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        arr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(rotated)


def denoise(crop: Image.Image) -> Image.Image:
    """Light denoise. OpenCV fast-NL-means when present, else a PIL median filter."""
    cv2 = _try_cv2()
    if cv2 is not None:
        arr = np.asarray(to_grayscale(crop))
        out = cv2.fastNlMeansDenoising(arr, h=7)
        return Image.fromarray(out)
    return crop.filter(ImageFilter.MedianFilter(size=3))


def band_aware_contrast(crop: Image.Image) -> Image.Image:
    """Normalize contrast for text that sits on/near the coloured reimbursability band.

    The colour band (green/red/orange) must NOT be hue-rotated — that signal is
    read separately by the reimbursability head. So we operate on **luminance
    only**: convert to grayscale and autocontrast. This lifts faint glyphs off a
    saturated background without ever flipping green↔red (no channel/hue ops).
    """
    return ImageOps.autocontrast(to_grayscale(crop), cutoff=1)


def pad(crop: Image.Image, ratio: float) -> Image.Image:
    """Add a proportional white border so glyphs aren't clipped after cropping."""
    if ratio <= 0:
        return crop
    w, h = crop.size
    bx, by = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    fill = 255 if crop.mode == "L" else (255, 255, 255)
    return ImageOps.expand(crop, border=(bx, by, bx, by), fill=fill)


def resize_height(crop: Image.Image, target_height: int) -> Image.Image:
    """Resize to a fixed height (aspect-preserving) to match the rec head input."""
    if target_height <= 0:
        return crop
    w, h = crop.size
    if h == target_height or h == 0:
        return crop
    new_w = max(1, int(round(w * (target_height / h))))
    return crop.resize((new_w, target_height), Image.BICUBIC)


def sharpen(crop: Image.Image) -> Image.Image:
    """Light unsharp mask — helps small printed codes."""
    return crop.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2))


# --------------------------------------------------------------------------- #
# The per-field-type pipeline (driven by recognition.yaml: field_types)
# --------------------------------------------------------------------------- #


def preprocess_for_type(
    crop: Image.Image,
    orientation: str,
    type_cfg: dict[str, Any],
) -> Image.Image:
    """Orient + apply the configured preprocessing pipeline for a field type.

    Order: orientation correction first (so deskew/denoise act on upright text),
    then the hooks declared in ``type_cfg['preprocess']``. Every step is config-
    gated; an absent/false flag skips that step. Pure PIL/NumPy (cv2 optional).

    Args:
        crop: raw field crop from detection.
        orientation: the field's orientation from ``classes.yaml``.
        type_cfg: one ``field_types.<type>`` block from ``recognition.yaml``.

    Returns:
        The preprocessed crop ready for the recognizer.
    """
    pp: dict[str, Any] = (type_cfg or {}).get("preprocess", {}) or {}

    out = orient(crop, orientation)
    if pp.get("band_aware_contrast"):
        out = band_aware_contrast(out)
    elif pp.get("to_grayscale", True):
        out = to_grayscale(out)
    if pp.get("deskew"):
        out = deskew(out)
    if pp.get("denoise"):
        out = denoise(out)
    if pp.get("sharpen"):
        out = sharpen(out)
    out = pad(out, float(pp.get("pad_ratio", 0.0)))
    out = resize_height(out, int(pp.get("target_height", 0)))
    return out


# --------------------------------------------------------------------------- #
# Lazy optional OpenCV
# --------------------------------------------------------------------------- #


def _try_cv2() -> Any | None:
    """Return the ``cv2`` module if installed, else ``None`` (never raises).

    OpenCV is an *optional* accelerator here, so unlike the backend in
    ``infer.py`` its absence is not an error — we degrade to PIL/NumPy.
    """
    try:
        import cv2  # noqa: PLC0415  (lazy by design — keeps the core CPU-only)

        return cv2
    except ImportError:
        return None
