"""Deterministic fixture-backed stand-ins for the detector and recognizer.

These let the **whole** end-to-end pipeline run on a CPU-only box today — no
``torch``/``rfdetr``/``paddleocr`` and no trained weights — by reading the
synthetic ground truth (``fixtures/synthetic/ground_truth.json``, via
:func:`vignocr.data.synthetic.load_ground_truth`) instead of a model.

* :class:`StubDetector` implements the same surface as
  ``vignocr.detection.infer.Detector.detect`` — ``detect(image) -> list[Detection]``
  — returning the ground-truth boxes for the image at confidence ``0.99``.
* :class:`StubRecognizer` implements the same surface as
  ``vignocr.ocr.infer.Recognizer.read`` — ``read(crop, field_type, orientation,
  *, flow=None, field_name=None) -> FieldRead`` — returning the ground-truth text
  for ``field_name`` at confidence ``0.99``.

Crucially the recognizer returns the text **exactly as it is drawn on the
vignette** (e.g. ``"PPA = 341.34 DA"``, ``"PRIX 306.49 DA"``), not the
already-normalized value — so the deterministic parsing/checksum/PPA layers
downstream are genuinely exercised, exactly as they would be on a real OCR read.

To honour the real, crop-only ``read`` signature (which carries no image id), the
orchestrator *binds* the current image onto the stubs out-of-band via
:meth:`bind_image` before extracting — the same way a real backend is "bound" to
its loaded weights. Both stubs resolve their image id once and serve that image's
ground truth. Image-id resolution (path / ``PIL.Image.filename`` / content hash)
lives in :func:`resolve_image_id`.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vignocr.common import BBox, FieldRead, get_classes, get_logger
from vignocr.data import synthetic

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

    from vignocr.detection.infer import Detection

log = get_logger(__name__)

# Stub reads are emitted at this confidence so they clear every abstention
# threshold (selling's strictest default is 0.90) — the stub asserts "I read this
# perfectly", which is the right semantics for replaying ground truth.
STUB_CONFIDENCE = 0.99

# How money fields are rendered on the synthetic vignette (must mirror
# vignocr.data.synthetic._render_image so the stub replays the *drawn* text, not
# the normalized value). Labels/markers are intentionally included so PPA
# disambiguation and money parsing have something real to chew on.
_MONEY_RENDER: dict[str, str] = {
    "prix": "PRIX {value} DA",
    "shp": "SHP {value} DA",
    "ppa": "PPA = {value} DA",
}


# --------------------------------------------------------------------------- #
# Image-id resolution (shared by the orchestrator + both stubs)
# --------------------------------------------------------------------------- #


def resolve_image_id(image: Image.Image | str | Path, root: str | Path | None = None) -> str:
    """Resolve the ground-truth key (image ``file_name``) for an input image.

    Resolution order:
      1. a path-like input → its base name;
      2. a ``PIL.Image`` carrying ``.filename`` (set when opened from disk) → its
         base name;
      3. otherwise an in-memory image → matched by **content hash** against the
         fixture images under ``root`` (the serving path, where the upload has no
         filename). Falls back to a synthetic ``"sha1:<digest>"`` id if nothing
         matches, so the caller still gets a stable, honest identifier.
    """
    if isinstance(image, str | Path):
        return Path(image).name

    filename = getattr(image, "filename", None)
    if filename:
        return Path(filename).name

    # In-memory image: hash its pixels and look it up in the fixture hash index.
    digest = _image_digest(image)
    if root is not None:
        by_hash = _fixture_hash_index(str(Path(root)))
        hit = by_hash.get(digest)
        if hit is not None:
            return hit
    return f"sha1:{digest}"


def _image_digest(image: Image.Image) -> str:
    """Stable SHA-1 over an image's RGB pixel bytes (mode/size-independent key)."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return hashlib.sha1(rgb.tobytes()).hexdigest()  # noqa: S324 - non-cryptographic id


@lru_cache(maxsize=8)
def _fixture_hash_index(root: str) -> dict[str, str]:
    """Map ``rgb-pixel-sha1 -> file_name`` for every fixture image under ``root``.

    Built once per root (cached). Used only on the in-memory (serving) path; the
    common path / filename routes never touch it.
    """
    from PIL import Image as PILImage  # noqa: PLC0415

    index: dict[str, str] = {}
    gt = _load_gt(root)
    root_path = Path(root)
    for file_name in gt:
        for img_path in root_path.rglob(file_name):
            try:
                with PILImage.open(img_path) as im:
                    index[_image_digest(im)] = file_name
            except (OSError, ValueError):  # pragma: no cover - corrupt fixture
                log.debug("stub.fixture_hash_failed", path=str(img_path))
            break  # first match for this file_name is enough
    log.debug("stub.fixture_hash_index", root=root, images=len(index))
    return index


@lru_cache(maxsize=8)
def _load_gt(root: str) -> dict[str, dict[str, Any]]:
    """Cached ground-truth load for a dataset root."""
    return synthetic.load_ground_truth(root)


# --------------------------------------------------------------------------- #
# Stub detector
# --------------------------------------------------------------------------- #


class StubDetector:
    """Fixture-backed detector: replays ground-truth boxes as detections.

    Mirrors ``vignocr.detection.infer.Detector``: construct it with the dataset
    ``root`` (where ``ground_truth.json`` lives), optionally bind an image id, then
    call :meth:`detect`. Every drawn class box is returned as a
    :class:`~vignocr.detection.infer.Detection` at confidence :data:`STUB_CONFIDENCE`.
    """

    backend = "stub"

    def __init__(self, root: str | Path) -> None:
        self.root = str(Path(root))
        self._bound_image_id: str | None = None
        log.info("stub_detector.init", root=self.root)

    def bind_image(self, image_id: str) -> StubDetector:
        """Bind the ground-truth key the next :meth:`detect` call should serve."""
        self._bound_image_id = image_id
        return self

    def detect(self, image: Image.Image | str | Path) -> list[Detection]:
        """Return the ground-truth boxes for the (bound or resolved) image.

        Args:
            image: a ``PIL.Image`` or path. If an image id was bound via
                :meth:`bind_image` it takes precedence; otherwise the id is
                resolved from ``image`` (path / filename / content hash).

        Returns:
            One :class:`~vignocr.detection.infer.Detection` per drawn class, each
            at confidence :data:`STUB_CONFIDENCE`. Unknown image → ``[]``.
        """
        from vignocr.detection.infer import Detection  # noqa: PLC0415 - same lazy graph

        image_id = self._bound_image_id or resolve_image_id(image, self.root)
        entry = _load_gt(self.root).get(image_id)
        if entry is None:
            log.warning("stub_detector.unknown_image", image_id=image_id, root=self.root)
            return []

        schema = get_classes()
        known = set(schema.names)
        dets: list[Detection] = []
        for name, box in entry.get("boxes", {}).items():
            if name not in known:  # defensive: only schema classes
                continue
            x, y, w, h = (float(v) for v in box)
            dets.append(
                Detection(
                    name=name,
                    score=STUB_CONFIDENCE,
                    bbox=BBox(x=x, y=y, w=w, h=h),
                )
            )
        log.debug("stub_detector.detect", image_id=image_id, n=len(dets))
        return dets


# --------------------------------------------------------------------------- #
# Stub recognizer
# --------------------------------------------------------------------------- #


class StubRecognizer:
    """Fixture-backed recognizer: replays ground-truth text for a field crop.

    Mirrors ``vignocr.ocr.infer.Recognizer.read``. Because the real signature is
    crop-only (no image id), the orchestrator binds the current image via
    :meth:`bind_image` before recognizing its fields; :meth:`read` then returns the
    ground-truth text for ``field_name`` at confidence :data:`STUB_CONFIDENCE`.

    Region/non-text types (``color_band``/``entete``/``vin``) abstain with no text,
    exactly like the real recognizer.
    """

    backend = "stub"

    # Field *types* that carry no recognizable text (mirrors Recognizer._NON_TEXT_TYPES).
    _NON_TEXT_TYPES = {"region"}

    def __init__(self, root: str | Path) -> None:
        self.root = str(Path(root))
        self._bound_image_id: str | None = None
        log.info("stub_recognizer.init", root=self.root)

    def bind_image(self, image_id: str) -> StubRecognizer:
        """Bind the ground-truth key whose field text subsequent reads serve."""
        self._bound_image_id = image_id
        return self

    def read(
        self,
        crop: Image.Image,  # noqa: ARG002 - kept for interface parity with Recognizer.read
        field_type: str,
        orientation: str,  # noqa: ARG002 - parity; the GT text is already upright
        *,
        flow: str | None = None,  # noqa: ARG002 - parity; abstention is moot at conf 0.99
        field_name: str | None = None,
    ) -> FieldRead:
        """Return the ground-truth :class:`FieldRead` for ``field_name``.

        The returned ``raw``/``value`` is the text **as drawn on the vignette**
        (money fields keep their ``PRIX``/``SHP``/``PPA =`` markers and ``DA``
        suffix), so downstream parsing/PPA/checksum run for real. Region types and
        unknown fields abstain — never a silent guess.
        """
        name = field_name or field_type

        # Structural/colour regions are not text — abstain (matches the real path).
        if field_type in self._NON_TEXT_TYPES:
            return FieldRead(
                name=name, value=None, raw=None, confidence=0.0, status="abstain", source="none"
            )

        image_id = self._bound_image_id
        if image_id is None:
            log.warning("stub_recognizer.unbound", field=name)
            return FieldRead(name=name, status="missing", source="none")

        entry = _load_gt(self.root).get(image_id)
        if entry is None:
            log.warning("stub_recognizer.unknown_image", image_id=image_id, field=name)
            return FieldRead(name=name, status="missing", source="none")

        text = self._rendered_text(name, entry.get("fields", {}))
        if text is None:
            # The field has no ground-truth value (e.g. a class with no text).
            return FieldRead(name=name, status="missing", source="none")

        return FieldRead(
            name=name,
            value=text,
            raw=text,
            confidence=STUB_CONFIDENCE,
            status="ok",
            source="ocr",
        )

    @staticmethod
    def _rendered_text(field_name: str, gt_fields: dict[str, str]) -> str | None:
        """Ground-truth text for a field, rendered as it appears on the vignette."""
        value = gt_fields.get(field_name)
        if value is None:
            return None
        template = _MONEY_RENDER.get(field_name)
        return template.format(value=value) if template else str(value)
