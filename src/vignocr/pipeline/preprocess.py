"""Whole-image preprocessing hooks run *before* detection.

The detector and recognizer each do their own field-crop preprocessing
(``vignocr.ocr.preprocess``); this module is the **page-level** pass applied once
to the full vignette at the very top of the pipeline: light deskew, denoise,
grayscale-luminance normalization, and a *band-aware* contrast lift.

The single hard constraint is the same one the augmentation config enforces: the
reimbursability ``color_band`` signal lives entirely in HUE, so we must never
hue-rotate, channel-swap, or grayscale the image we hand downstream — doing so
would let a green band read red (or vice-versa). Every hook here therefore either
preserves the RGB hue exactly (deskew/denoise operate geometrically or on
luminance and re-merge) or is applied to a *throwaway* luminance copy. The image
returned to the detector/reimbursability stages keeps its original colour.

Everything is **PIL + NumPy**. OpenCV (``cv2``) is an optional accelerator for
deskew/denoise — lazy-imported and silently skipped when absent — so this module
imports and runs on CPU without the ``[ml]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from vignocr.common import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

log = get_logger(__name__)

# Default page-level pipeline. Conservative on purpose: the synthetic fixture is
# already clean, and on real captures the band hue must survive untouched.
_DEFAULT_PIPELINE: dict[str, Any] = {
    "deskew": False,  # off by default — a bad full-page estimate is worse than none
    "denoise": False,
    "normalize_contrast": True,  # luminance-only autocontrast (hue-preserving)
    "max_skew_deg": 5.0,
}


def normalize(image: Image.Image, cfg: dict[str, Any] | None = None) -> Image.Image:
    """Run the configured page-level preprocessing pipeline on a full vignette.

    Args:
        image: the input vignette (any PIL mode; coerced to RGB so the band hue
            is carried through every stage).
        cfg: optional ``pipeline.preprocess`` block. Missing keys fall back to
            :data:`_DEFAULT_PIPELINE`. Each hook is config-gated and skipped when
            its flag is false/absent.

    Returns:
        A preprocessed **RGB** image. Hue is preserved end-to-end (the band-colour
        reimbursability signal is never altered).
    """
    pp = {**_DEFAULT_PIPELINE, **(cfg or {})}
    out = image if image.mode == "RGB" else image.convert("RGB")

    if pp.get("deskew"):
        out = deskew(out, max_angle=float(pp.get("max_skew_deg", 5.0)))
    if pp.get("denoise"):
        out = denoise(out)
    if pp.get("normalize_contrast"):
        out = band_aware_contrast(out)

    log.debug(
        "pipeline.preprocess",
        size=out.size,
        deskew=bool(pp.get("deskew")),
        denoise=bool(pp.get("denoise")),
        normalize_contrast=bool(pp.get("normalize_contrast")),
    )
    return out


def deskew(image: Image.Image, *, max_angle: float = 5.0) -> Image.Image:
    """Correct a small whole-page in-plane skew, preserving colour.

    Estimates the dominant text angle on a luminance copy (OpenCV's
    minimum-area-rect when ``cv2`` is present) and rotates the **colour** image by
    that angle. The estimate is clamped to ``±max_angle`` so a bad estimate can
    never violently rotate the page. Without ``cv2`` the image is returned
    unchanged (deskew is a refinement, not a correctness requirement).
    """
    cv2 = _try_cv2()
    if cv2 is None:
        return image
    gray = np.asarray(image.convert("L"))
    inv = 255 - gray  # dark ink -> "on"
    coords = np.column_stack(np.where(inv > inv.mean()))
    if coords.shape[0] < 10:
        return image
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return image
    # Rotate the colour image (expand=False keeps the canvas; white fill avoids
    # introducing a coloured border that a naive band read might pick up).
    return image.rotate(angle, expand=False, fillcolor=(255, 255, 255), resample=Image.BICUBIC)


def denoise(image: Image.Image) -> Image.Image:
    """Light, hue-preserving denoise.

    Uses OpenCV's coloured fast-NL-means when present (operates per-channel, so
    hue is preserved); otherwise a mild PIL median filter on the RGB image.
    """
    cv2 = _try_cv2()
    if cv2 is not None:
        arr = np.asarray(image.convert("RGB"))
        out = cv2.fastNlMeansDenoisingColored(arr, None, 5, 5, 7, 21)
        return Image.fromarray(out)
    return image.filter(ImageFilter.MedianFilter(size=3))


def band_aware_contrast(image: Image.Image) -> Image.Image:
    """Lift global contrast without touching hue (the band signal must survive).

    Autocontrast is applied to the **luminance (Y) channel only** via a YCbCr
    round-trip: the chroma channels (Cb/Cr) — which carry the green/red/orange
    band hue — are left untouched, so green can never drift toward red. This is
    strictly stronger than a grayscale autocontrast: it keeps colour while still
    normalizing brightness/contrast for faint captures.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    ycbcr = image.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = ImageOps.autocontrast(y, cutoff=1)
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def _try_cv2() -> Any | None:
    """Return the ``cv2`` module if installed, else ``None`` (never raises).

    OpenCV is an *optional* accelerator here; its absence degrades gracefully to
    PIL/NumPy so the core stays CPU-only and ``[ml]``-free.
    """
    try:
        import cv2  # noqa: PLC0415  (lazy by design — keeps the core CPU-only)

        return cv2
    except ImportError:
        return None
