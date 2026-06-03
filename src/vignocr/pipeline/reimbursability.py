"""Reimbursability classification from the coloured band — a non-OCR signal.

CHIFA eligibility is encoded by the colour of the diagonal ``color_band`` on the
vignette, NOT by any text: green = remboursable, red = non remboursable, orange =
à vérifier. This module reads that band region's dominant **hue** and maps it to a
:class:`~vignocr.common.schemas.Reimbursability` via the palette declared in
``configs/classes.yaml: reimbursability.colors`` — nothing is hardcoded.

Hue (not RGB distance) is the discriminator on purpose: it is robust to the
brightness/contrast variation of phone captures, and it is exactly the channel the
augmentation/preprocess rules are forbidden from touching, so the train-time and
inference-time notions of "the band colour" stay identical.

``orange`` and an unrecognized hue both yield ``eligible=None`` (abstain → "à
vérifier"): a reimbursability decision is never *guessed*. Pure NumPy/PIL.
"""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING, Any

import numpy as np

from vignocr.common import BBox, Reimbursability, get_classes, get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

log = get_logger(__name__)

# Reference hue (degrees, 0..360) for each palette key. These are the canonical
# positions on the colour wheel; the actual decision is "nearest reference hue
# among the *configured* colours", so adding/removing a palette colour in
# classes.yaml changes the behaviour without code edits.
_REFERENCE_HUE_DEG: dict[str, float] = {
    "green": 120.0,
    "red": 0.0,
    "orange": 30.0,
}

# A pixel only counts toward the band hue if it is colourful and bright enough —
# this rejects the light ``entete`` background that the axis-aligned band bbox
# unavoidably also covers (the band itself is a thin diagonal line).
_MIN_SATURATION = 0.25
_MIN_VALUE = 0.20
# Max angular distance (deg) from a reference hue to accept the colour at all.
# Beyond this the band hue is "unknown" -> abstain rather than mis-classify.
_MAX_HUE_DISTANCE_DEG = 45.0


def classify_band(
    image: Image.Image,
    color_band_bbox: BBox | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> Reimbursability:
    """Classify CHIFA reimbursability from the band region's dominant hue.

    Args:
        image: the (preprocessed) vignette, any PIL mode — coerced to RGB so hue
            is read faithfully.
        color_band_bbox: the detected ``color_band`` region in input-image pixels
            (COCO ``xywh``). ``None`` (band not localized) → ``unknown`` / abstain.
        cfg: optional override of ``classes.yaml: reimbursability`` (the colour
            palette + labels). Defaults to the schema's reimbursability block.

    Returns:
        A :class:`~vignocr.common.schemas.Reimbursability`: ``color`` is the
        matched palette key (or ``"unknown"``), ``eligible`` is ``True``/``False``
        for green/red and ``None`` for orange/unknown, ``label`` comes from the
        palette, and ``confidence`` reflects how cleanly the hue matched.
    """
    palette = _palette(cfg)

    if color_band_bbox is None:
        log.debug("reimbursability.no_band")
        return _result("unknown", palette, confidence=0.0)

    region = _crop_region(image, color_band_bbox)
    if region is None:
        return _result("unknown", palette, confidence=0.0)

    hue_deg, sat, frac_coloured = _dominant_hue(region)
    if hue_deg is None:
        # No colourful pixels in the band region (e.g. a blank/over-exposed crop).
        log.debug("reimbursability.no_coloured_pixels", coloured_fraction=round(frac_coloured, 4))
        return _result("unknown", palette, confidence=0.0)

    color, distance = _nearest_palette_color(hue_deg, palette)
    if color is None or distance > _MAX_HUE_DISTANCE_DEG:
        log.debug(
            "reimbursability.hue_unmatched",
            hue_deg=round(hue_deg, 1),
            nearest=color,
            distance_deg=None if distance is None else round(distance, 1),
        )
        return _result("unknown", palette, confidence=0.0)

    # Confidence: how cleanly the hue matched (1 at the reference, 0 at the cap),
    # tempered by saturation and how much of the region was actually coloured.
    hue_score = max(0.0, 1.0 - distance / _MAX_HUE_DISTANCE_DEG)
    confidence = float(hue_score * min(1.0, sat / _MIN_SATURATION) * min(1.0, frac_coloured * 4.0))
    confidence = max(0.0, min(1.0, confidence))

    log.debug(
        "reimbursability.classified",
        color=color,
        hue_deg=round(hue_deg, 1),
        distance_deg=round(distance, 1),
        confidence=round(confidence, 3),
    )
    return _result(color, palette, confidence=confidence)


# --------------------------------------------------------------------------- #
# Palette + result helpers
# --------------------------------------------------------------------------- #


def _palette(cfg: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Resolve the ``{color_key: {eligible, label}}`` palette from config."""
    block = cfg if cfg is not None else get_classes().reimbursability
    colors = (block or {}).get("colors", {}) or {}
    return {str(k): dict(v or {}) for k, v in colors.items()}


def _result(
    color: str,
    palette: dict[str, dict[str, Any]],
    *,
    confidence: float,
) -> Reimbursability:
    """Build a ``Reimbursability`` for ``color`` using the palette's label/eligible."""
    spec = palette.get(color, {})
    label = spec.get("label")
    eligible = spec.get("eligible")  # may be True / False / None (orange)
    if color == "unknown" and label is None:
        # The schema palette has no explicit 'unknown' row; use the model default
        # label ("À vérifier") for an unrecognized band.
        return Reimbursability(color="unknown", eligible=None, confidence=confidence)
    return Reimbursability(
        color=color,  # validated against BandColor literal by the model
        eligible=eligible,
        confidence=confidence,
        label=label if label is not None else "À vérifier",
    )


# --------------------------------------------------------------------------- #
# Pixel reads (NumPy)
# --------------------------------------------------------------------------- #


def _crop_region(image: Image.Image, bbox: BBox) -> Image.Image | None:
    """Crop the band bbox from the image, clamped to bounds (None if degenerate)."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    w, h = rgb.size
    x0 = max(0, int(round(bbox.x)))
    y0 = max(0, int(round(bbox.y)))
    x1 = min(w, int(round(bbox.x + bbox.w)))
    y1 = min(h, int(round(bbox.y + bbox.h)))
    if x1 <= x0 or y1 <= y0:
        log.debug("reimbursability.degenerate_bbox", bbox=[bbox.x, bbox.y, bbox.w, bbox.h])
        return None
    return rgb.crop((x0, y0, x1, y1))


def _dominant_hue(region: Image.Image) -> tuple[float | None, float, float]:
    """Mean hue (deg) of the *coloured* pixels in ``region``.

    Returns ``(hue_deg | None, mean_saturation, coloured_fraction)``. Only pixels
    above the saturation/value floors contribute (rejecting the light background
    the axis-aligned band box also spans). Hue is averaged **circularly** (via
    unit vectors) so the wrap-around at 0°/360° (red) is handled correctly. When
    no pixel qualifies, the hue is ``None``.
    """
    arr = np.asarray(region, dtype=np.float32) / 255.0  # HWC RGB in [0,1]
    flat = arr.reshape(-1, 3)
    maxc = flat.max(axis=1)
    minc = flat.min(axis=1)
    value = maxc
    delta = maxc - minc
    # Saturation (HSV): delta / value, guarding value==0.
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(value > 0, delta / np.maximum(value, 1e-6), 0.0)

    mask = (sat >= _MIN_SATURATION) & (value >= _MIN_VALUE) & (delta > 1e-6)
    coloured_fraction = float(mask.mean()) if mask.size else 0.0
    if not mask.any():
        return None, 0.0, coloured_fraction

    sel = flat[mask]
    # Per-pixel hue in [0,1) via colorsys (vectorized over the selected pixels).
    hues = np.array([colorsys.rgb_to_hsv(r, g, b)[0] for r, g, b in sel], dtype=np.float64)
    angles = hues * 2.0 * np.pi
    mean_x = float(np.cos(angles).mean())
    mean_y = float(np.sin(angles).mean())
    mean_angle = np.arctan2(mean_y, mean_x)  # (-pi, pi]
    hue_deg = float((np.degrees(mean_angle)) % 360.0)
    mean_sat = float(sat[mask].mean())
    return hue_deg, mean_sat, coloured_fraction


def _nearest_palette_color(
    hue_deg: float, palette: dict[str, dict[str, Any]]
) -> tuple[str | None, float | None]:
    """Nearest configured palette colour to ``hue_deg`` + its angular distance.

    Only palette keys with a known reference hue are considered; the distance is
    the circular (wrap-around) hue distance in degrees.
    """
    best_color: str | None = None
    best_dist: float | None = None
    for color in palette:
        ref = _REFERENCE_HUE_DEG.get(color)
        if ref is None:
            continue
        dist = _hue_distance_deg(hue_deg, ref)
        if best_dist is None or dist < best_dist:
            best_dist, best_color = dist, color
    return best_color, best_dist


def _hue_distance_deg(a: float, b: float) -> float:
    """Smallest angular distance between two hues on the 0..360 wheel."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)
