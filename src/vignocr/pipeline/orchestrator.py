"""The end-to-end VignOCR pipeline — detection → recognition → parse → correct.

:class:`VignocrPipeline` composes every stage into a single
``extract(image, flow=...) -> ExtractionRecord``:

    preprocess (page-level, hue-preserving)
      → detect            (real RF-DETR Detector if [ml]+weights, else StubDetector)
      → per-field crop    (one crop per detected field box)
      → recognize         (real Recognizer if [ml], else StubRecognizer)
      → PPA disambiguate  (pick the final "= XXX,XX DA" among PPA reads)
      → record.build      (money checksum + per-flow abstention -> partial record)
      → nomenclature      (load + structural match on the anchor + policy correct)
      → reimbursability   (band-colour -> CHIFA eligibility, a non-OCR signal)
      → assemble          (fields + reports + model_versions + timings_ms)

**Backend selection** is config-driven (``pipeline.backend``: ``auto`` | ``stub`` |
``real``), overridable by env ``VIGNOCR_PIPELINE_BACKEND``. ``auto`` uses the real
backends when ``torch`` *and* a detector checkpoint are available, else the
deterministic fixture stubs — so the same code runs unchanged on a CPU-only box
today and on a GPU host with trained weights. Heavy ML libs are imported lazily
*inside* the real backends, never here; importing this module is CPU-safe.

The orchestrator accepts the serving layer's plain-``dict`` config
(``{detector_path, recognizer_path, nomenclature_csv, default_flow, ...}``) and
exposes ``model_versions()`` so it satisfies ``serving.deps.PipelineLike``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vignocr.common import (
    BBox,
    ExtractionRecord,
    FieldRead,
    Reimbursability,
    get_classes,
    get_logger,
    load_config,
)
from vignocr.common.config import get_active_dataset
from vignocr.common.schemas import Flow, NomenclatureReport
from vignocr.nomenclature import correct as nomenclature_correct
from vignocr.nomenclature import loader as nomenclature_loader
from vignocr.nomenclature import match as nomenclature_match
from vignocr.ocr import orient
from vignocr.parsing import ppa as ppa_mod
from vignocr.parsing import record as record_mod
from vignocr.pipeline import preprocess as preprocess_mod
from vignocr.pipeline import reimbursability as reimbursability_mod
from vignocr.pipeline.stubs import StubDetector, StubRecognizer, resolve_image_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

    from vignocr.detection.infer import Detection

log = get_logger(__name__)

_ENV_BACKEND = "VIGNOCR_PIPELINE_BACKEND"
_VALID_BACKENDS = {"auto", "stub", "real"}

# classes.yaml `type` values that are structural/colour regions, not OCR fields.
# (Mirrors ocr.infer.Recognizer._NON_TEXT_TYPES.) These are excluded from the
# recognized `fields` map; their boxes are still used elsewhere (reimbursability).
_REGION_TYPES = {"region"}


class VignocrPipeline:
    """Compose detection + recognition + parsing + correction into one extractor.

    Args:
        cfg: pipeline configuration. Either a plain dict (as the serving layer
            passes — ``detector_path`` / ``recognizer_path`` / ``nomenclature_csv``
            / ``default_flow`` / ``backend``), or ``None`` to use defaults. A
            ``backend`` / ``pipeline.backend`` of ``auto`` | ``stub`` | ``real``
            selects the backend; env ``VIGNOCR_PIPELINE_BACKEND`` overrides it.

    Constructing a pipeline is cheap and import-safe: no model is loaded and no ML
    lib is imported until :meth:`extract` actually runs a real backend.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg: dict[str, Any] = dict(cfg or {})
        self._schema = get_classes()

        # --- resolution: paths, dataset root, default flow, backend mode -------
        self.detector_path: str | None = self.cfg.get("detector_path")
        self.recognizer_path: str | None = self.cfg.get("recognizer_path")
        self.nomenclature_csv: str | None = self.cfg.get("nomenclature_csv")
        # Stage A (vignette detector) is OPTIONAL. When configured, extract()
        # runs it first to crop the vignette region out of a wider drug-box
        # photo; when None, the input is assumed to already be a vignette
        # (the back-compat behaviour synthetic tests rely on).
        self.vignette_detector_path: str | None = self.cfg.get("vignette_detector_path")
        self.vignette_cfg_path: str = self.cfg.get(
            "vignette_cfg_path", "detection/rfdetr_vignette"
        )
        # Crop target for Stage A. In the data2 annotation `vin` is the WIDE
        # vignette body (~39% of the photo) and `entete` is the narrow vertical
        # strip INSIDE it holding num_lot/date_fab/date_exp (~7%). The previous
        # default ("entete") cropped to the strip — discarding every other field
        # before Stage B ever ran. Stage B must see the whole vignette: `vin`.
        self.vignette_class: str = self.cfg.get("vignette_class", "vin")
        # Relative margin added around the Stage A crop so a slightly-tight box
        # never clips edge fields (fraction of box size per side).
        self.vignette_crop_margin: float = float(self.cfg.get("vignette_crop_margin", 0.04))
        self.default_flow: Flow = self.cfg.get("default_flow", "selling")
        self._preprocess_cfg: dict[str, Any] = self.cfg.get("preprocess", {}) or {}
        self._root: str = self._resolve_root()
        self._backend_mode: str = self._resolve_backend_mode()

        # --- lazily-built handles ---------------------------------------------
        self._detector: Any | None = None
        self._vignette_detector: Any | None = None
        self._recognizer: Any | None = None
        self._nomenclature_index: Any | None = None
        self._resolved_backend: str | None = None  # "real" | "stub", set on first use

        log.info(
            "pipeline.init",
            backend_mode=self._backend_mode,
            detector_path=self.detector_path,
            root=self._root,
            default_flow=self.default_flow,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(
        self, image: Image.Image | str | Path, *, flow: Flow | None = None
    ) -> ExtractionRecord:
        """Run one vignette end-to-end and return its :class:`ExtractionRecord`.

        Args:
            image: a ``PIL.Image`` or a path to a vignette image.
            flow: ``"selling"`` (stricter abstention) or ``"receiving"``. Defaults
                to the configured ``default_flow``.

        Returns:
            A fully-assembled :class:`ExtractionRecord` — per-field reads (value,
            confidence, status, source, bbox), the money checksum report, the
            nomenclature match/conflict report, the reimbursability classification,
            the per-flow abstention list, and ``model_versions`` + ``timings_ms``.
        """
        flow = flow or self.default_flow
        timings: dict[str, float] = {}

        pil = self._as_pil(image)
        image_id = resolve_image_id(image, self._root)

        # 0) STAGE A — vignette region detection (optional, configured via
        #    `vignette_detector_path`). Crops a wider drug-box photo down to the
        #    vignette region before the field detector runs. When unconfigured,
        #    the input is treated as an already-cropped vignette (back-compat).
        with _timed(timings, "vignette"):
            pil = self._pre_crop_vignette(pil, image_id)

        # 1) Page-level preprocess (hue-preserving). Keep the original for any
        #    stub content-hashing (already resolved image_id above) and band read.
        with _timed(timings, "preprocess"):
            prepared = preprocess_mod.normalize(pil, self._preprocess_cfg)

        # 2) Detect field boxes (real or stub).
        with _timed(timings, "detect"):
            detections = self._detect(prepared, image_id)

        # 3) Per-field orient + crop -> 4) recognize.
        with _timed(timings, "recognize"):
            field_reads, ppa_candidates = self._recognize_fields(
                prepared, detections, image_id, flow
            )

        # 5) PPA disambiguation: choose the final "= XXX,XX DA" among PPA reads,
        #    BEFORE the checksum so it anchors prix+shp==ppa with the right value.
        with _timed(timings, "ppa"):
            if ppa_candidates:
                field_reads["ppa"] = ppa_mod.disambiguate(ppa_candidates)

        # 5b) Real-data reconciliation: the Algeria-Drug-label v1 annotation puts
        #     prix + shp inside ONE `ppa_shp` box. Split it here so the standard
        #     prix/shp/ppa checksum runs unchanged. `setdefault` means synthetic
        #     (which already carries native prix/shp reads) is never overwritten.
        with _timed(timings, "ppa_shp"):
            combined = field_reads.get("ppa_shp")
            if combined is not None:
                from vignocr.parsing import ppa_shp as ppa_shp_mod

                prix_fr, shp_fr = ppa_shp_mod.split(combined)
                field_reads.setdefault("prix", prix_fr)
                field_reads.setdefault("shp", shp_fr)

        # 6) Parsing record: money checksum (verify/repair/flag) + per-flow
        #    abstention. record.build owns checksum.verify_and_repair on the
        #    prix/shp/ppa triple, so it runs exactly once here.
        with _timed(timings, "parse"):
            partial = record_mod.build(field_reads, flow, image_id=image_id)
            fields = dict(partial.fields)
            # Canonicalize money field *values* to the centime-quantized Decimal
            # strings the checksum already parsed (the contract: money as "250.00").
            # The recognizer is a faithful transcriber (value == raw OCR text), and
            # checksum.verify_and_repair only rewrites a value when it *repairs* it
            # — so on an "ok"/"mismatch" verdict prix/shp keep their raw markers
            # ("PRIX 306.49 DA"). We promote them to canonical here using parsing's
            # own parse output (no re-implementation of money parsing).
            fields = self._canonicalize_money(fields, partial.checksum)

        # 7) Nomenclature: structural match on the anchor + safety-policy correct
        #    (never touches ppa/tr; flags dispensing-critical conflicts).
        with _timed(timings, "nomenclature"):
            fields, nomenclature_report = self._apply_nomenclature(fields)

        # 8) Reimbursability from the colour band (a non-OCR, hue-based signal).
        with _timed(timings, "reimbursability"):
            reimb = self._classify_reimbursability(prepared, detections)

        # 9) Assemble. Recompute abstentions from FINAL statuses (nomenclature may
        #    have resolved or flagged fields after the parsing-stage abstention).
        abstentions = sorted(name for name, fr in fields.items() if fr.status == "abstain")

        record = ExtractionRecord(
            image_id=image_id,
            fields=fields,
            reimbursability=reimb,
            checksum=partial.checksum,
            nomenclature=nomenclature_report,
            abstentions=abstentions,
            flow=flow,
            model_versions=self.model_versions(),
            timings_ms={k: round(v, 3) for k, v in timings.items()},
        )
        log.info(
            "pipeline.extract",
            image_id=image_id,
            flow=flow,
            backend=self._resolved_backend,
            n_fields=len(fields),
            n_abstentions=len(abstentions),
            checksum=record.checksum.verdict,
            reimbursability=reimb.color,
            total_ms=round(sum(timings.values()), 1),
        )
        return record

    def model_versions(self) -> dict[str, str]:
        """Report the live model/source versions (real weights or ``"stub"``).

        The backend is resolved lazily; before the first :meth:`extract` this
        reflects the *configured* mode (``auto`` → the resolved backend once known,
        else the would-be backend), so ``/ready`` is honest about stub vs real.
        """
        backend = self._resolved_backend or self._would_be_backend()
        if backend == "real":
            return {
                "detector": _version_tag(self.detector_path, "rfdetr_medium"),
                "recognizer": _version_tag(self.recognizer_path, self._ocr_backend_name()),
                "nomenclature_version": self._nomenclature_version(),
            }
        return {
            "detector": "stub",
            "recognizer": "stub",
            "nomenclature_version": self._nomenclature_version(),
        }

    # ------------------------------------------------------------------ #
    # Stage helpers
    # ------------------------------------------------------------------ #

    def _detect(self, image: Image.Image, image_id: str) -> list[Detection]:
        """Detect field boxes, building (and binding) the right backend lazily."""
        detector = self._get_detector()
        if isinstance(detector, StubDetector):
            detector.bind_image(image_id)
        return detector.detect(image)

    def _recognize_fields(
        self,
        image: Image.Image,
        detections: list[Detection],
        image_id: str,
        flow: Flow,
    ) -> tuple[dict[str, FieldRead], list[FieldRead]]:
        """Crop each detection, recognize it, and collect reads keyed by field name.

        Per the schema, ``rotated_fields`` (the vertical vin strip) are oriented
        upright before recognition. The real recognizer also orients internally;
        we pass the *raw* crop so behaviour matches whichever backend is active.

        Returns ``(field_reads, ppa_candidates)`` — ``ppa_candidates`` are kept
        separate so PPA disambiguation can choose the final value when the band
        carries both an intermediate and a final PPA line.
        """
        recognizer = self._get_recognizer()
        if isinstance(recognizer, StubRecognizer):
            recognizer.bind_image(image_id)

        field_reads: dict[str, FieldRead] = {}
        ppa_candidates: list[FieldRead] = []

        for det in detections:
            name = det.name
            spec = self._safe_spec(name)
            if spec is None:
                continue
            field_type = str(spec.get("type", "text"))
            orientation = str(spec.get("orientation", "horizontal"))

            # Structural/colour regions (entete/vin/color_band) carry no OCR field:
            # the reimbursability head reads color_band's hue separately (from the
            # detections, not from `fields`), and entete/vin are layout. Skipping
            # them keeps `fields` exactly the recognizable set and `abstentions`
            # free of non-field clutter.
            if field_type in _REGION_TYPES:
                continue

            crop = self._crop(image, det.bbox)
            if crop is None:
                continue

            read = recognizer.read(
                crop, field_type=field_type, orientation=orientation, flow=flow, field_name=name
            )
            # Attach the source box so the consumer can locate/redraw the field.
            read = read.model_copy(update={"bbox": det.bbox})

            if name == "ppa":
                ppa_candidates.append(read)
                # Seed the slot; disambiguation may replace it from the candidates.
                field_reads.setdefault("ppa", read)
            else:
                # Highest-confidence read wins if a class is detected more than once.
                prev = field_reads.get(name)
                if prev is None or read.confidence >= prev.confidence:
                    field_reads[name] = read

        return field_reads, ppa_candidates

    def _apply_nomenclature(
        self, fields: dict[str, FieldRead]
    ) -> tuple[dict[str, FieldRead], NomenclatureReport]:
        """Match the anchor against the register and apply the correction policy."""
        cfg = load_config("nomenclature/correction")
        index = self._get_nomenclature_index()

        anchor = fields.get(self._anchor_field())
        anchor_value = anchor.value if anchor else None
        row, _conf = nomenclature_match.find(anchor_value, index, cfg)
        return nomenclature_correct.apply(fields, row, cfg)

    def _pre_crop_vignette(self, pil: Image.Image, image_id: str) -> Image.Image:
        """Stage A: crop the input down to the detected vignette region.

        Pass-through when no Stage A model is configured (the synthetic test
        path). When configured, lazy-loads a ``Detector`` bound to the vignette
        config, picks the highest-scoring ``vignette_class`` (default ``vin`` —
        the whole vignette body in the data2 annotation; ``entete`` is only the
        lot/dates strip inside it), expands the box by ``vignette_crop_margin``
        per side, and returns the cropped PIL image. If no qualifying box is
        found, falls back to the original image and logs.
        """
        if not self.vignette_detector_path:
            return pil
        if self._vignette_detector is None:
            try:
                from vignocr.detection.infer import Detector
            except ImportError:
                log.warning("pipeline.vignette_skipped_no_ml", image_id=image_id)
                return pil
            self._vignette_detector = Detector(
                self.vignette_detector_path, cfg_path=self.vignette_cfg_path
            )
            log.info(
                "pipeline.vignette_loaded",
                path=self.vignette_detector_path,
                cfg=self.vignette_cfg_path,
                target_class=self.vignette_class,
            )

        try:
            dets = self._vignette_detector.detect(pil)
        except Exception as exc:  # noqa: BLE001
            log.warning("pipeline.vignette_inference_failed", image_id=image_id, err=str(exc))
            return pil

        targets = [d for d in dets if d.name == self.vignette_class]
        if not targets:
            log.warning(
                "pipeline.vignette_not_found",
                image_id=image_id,
                target=self.vignette_class,
                n_detections=len(dets),
            )
            return pil
        best = max(targets, key=lambda d: d.score)
        # Expand the box by the configured margin so a tight detection never
        # clips fields sitting on the vignette border (clamped to the image).
        mx = best.bbox.w * self.vignette_crop_margin
        my = best.bbox.h * self.vignette_crop_margin
        x0 = max(0, int(round(best.bbox.x - mx)))
        y0 = max(0, int(round(best.bbox.y - my)))
        x1 = min(pil.width, int(round(best.bbox.x + best.bbox.w + mx)))
        y1 = min(pil.height, int(round(best.bbox.y + best.bbox.h + my)))
        if x1 <= x0 or y1 <= y0:
            return pil
        log.info(
            "pipeline.vignette_cropped",
            image_id=image_id,
            bbox=(x0, y0, x1, y1),
            score=round(best.score, 3),
        )
        return pil.crop((x0, y0, x1, y1))

    def _canonicalize_money(
        self, fields: dict[str, FieldRead], checksum: Any
    ) -> dict[str, FieldRead]:
        """Set each money field's ``value`` to the checksum's parsed Decimal string.

        Uses the canonical strings the checksum already computed via
        ``parsing.money.parse`` (``checksum.prix``/``shp``/``ppa``), so no money
        parsing is re-implemented here. ``raw`` is preserved (the faithful OCR
        text); only ``value`` is normalized. A field whose amount could not be
        parsed (report value ``None``) is left untouched.
        """
        out = dict(fields)
        for name in self._money_fields():
            fr = out.get(name)
            if fr is None:
                continue
            canonical = getattr(checksum, name, None)
            if canonical is not None and fr.value != canonical:
                out[name] = fr.model_copy(update={"value": canonical})
        return out

    def _classify_reimbursability(
        self, image: Image.Image, detections: list[Detection]
    ) -> Reimbursability:
        """Read the band colour from the detected ``color_band`` region."""
        region_name = self._reimbursability_region()
        band_bbox: BBox | None = next((d.bbox for d in detections if d.name == region_name), None)
        return reimbursability_mod.classify_band(image, band_bbox)

    # ------------------------------------------------------------------ #
    # Backend construction (lazy) + selection
    # ------------------------------------------------------------------ #

    def _get_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        if self._use_real():
            self._detector = self._build_real_detector()
            self._resolved_backend = "real"
        else:
            self._detector = StubDetector(self._root)
            self._resolved_backend = "stub"
        return self._detector

    def _get_recognizer(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        if self._use_real():
            self._recognizer = self._build_real_recognizer()
            self._resolved_backend = "real"
        else:
            self._recognizer = StubRecognizer(self._root)
            self._resolved_backend = "stub"
        return self._recognizer

    def _build_real_detector(self) -> Any:
        """Construct the real RF-DETR detector (lazy ML import inside Detector)."""
        from vignocr.detection.infer import Detector  # noqa: PLC0415

        if not self.detector_path:
            raise ImportError(
                "Real detection backend selected but no detector weights configured "
                "(set detector_path / VIGNOCR_DETECTOR_PATH). For a CPU-only run use "
                "pipeline.backend=stub."
            )
        return Detector(self.detector_path)

    def _build_real_recognizer(self) -> Any:
        """Construct the real OCR recognizer (lazy backend import inside Recognizer)."""
        from vignocr.ocr.infer import Recognizer  # noqa: PLC0415

        return Recognizer()

    def _use_real(self) -> bool:
        """Decide whether to use the real backends (resolves ``auto``)."""
        if self._backend_mode == "real":
            return True
        if self._backend_mode == "stub":
            return False
        # auto: real only if torch is importable AND a detector checkpoint exists.
        return _torch_available() and bool(self.detector_path)

    def _would_be_backend(self) -> str:
        """The backend ``model_versions`` reports before the first extract."""
        return "real" if self._use_real() else "stub"

    # ------------------------------------------------------------------ #
    # Config / schema resolution
    # ------------------------------------------------------------------ #

    def _resolve_backend_mode(self) -> str:
        """Resolve the backend mode from env > cfg > nested pipeline cfg > 'auto'."""
        env = os.environ.get(_ENV_BACKEND)
        raw = (
            env
            or self.cfg.get("backend")
            or (self.cfg.get("pipeline", {}) or {}).get("backend")
            or "auto"
        )
        mode = str(raw).strip().lower()
        if mode not in _VALID_BACKENDS:
            log.warning("pipeline.bad_backend_mode", value=mode, fallback="auto")
            return "auto"
        return mode

    def _resolve_root(self) -> str:
        """Resolve the dataset root (where the synthetic ground truth lives)."""
        explicit = self.cfg.get("root") or self.cfg.get("dataset_root")
        if explicit:
            return str(explicit)
        try:
            return str(get_active_dataset()["root"])
        except Exception:  # pragma: no cover - config best-effort
            log.debug("pipeline.root_fallback")
            return "fixtures/synthetic"

    def _safe_spec(self, name: str) -> dict[str, Any] | None:
        try:
            return self._schema.by_name(name)
        except KeyError:
            log.debug("pipeline.unknown_class", name=name)
            return None

    def _anchor_field(self) -> str:
        return self._schema.role("anchor_field") or "num_enregistrement"

    def _money_fields(self) -> list[str]:
        return list(self._schema.role("money_fields") or ["ppa", "prix", "shp"])

    def _reimbursability_region(self) -> str:
        return self._schema.role("reimbursability_region") or "color_band"

    def _ocr_backend_name(self) -> str:
        try:
            return str(load_config("ocr/recognition").get("backend", "baseline"))
        except Exception:  # pragma: no cover
            return "baseline"

    def _nomenclature_version(self) -> str:
        """A stable version tag for the nomenclature source (csv path basename)."""
        path = self.nomenclature_csv
        if not path:
            try:
                path = load_config("nomenclature/correction").get("csv", {}).get("path")
            except Exception:  # pragma: no cover
                path = None
        return Path(path).name if path else "unknown"

    def _get_nomenclature_index(self) -> Any:
        """Load (once) the nomenclature index from the configured/overridden CSV."""
        if self._nomenclature_index is None:
            self._nomenclature_index = nomenclature_loader.load_csv(self.nomenclature_csv)
        return self._nomenclature_index

    # ------------------------------------------------------------------ #
    # Image helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_pil(image: Image.Image | str | Path) -> Image.Image:
        from PIL import Image as PILImage  # noqa: PLC0415

        if isinstance(image, str | Path):
            return PILImage.open(image).convert("RGB")
        return image.convert("RGB") if image.mode != "RGB" else image

    @staticmethod
    def _crop(image: Image.Image, bbox: BBox) -> Image.Image | None:
        """Crop ``bbox`` (COCO xywh) from ``image``, clamped to bounds."""
        w, h = image.size
        x0 = max(0, int(round(bbox.x)))
        y0 = max(0, int(round(bbox.y)))
        x1 = min(w, int(round(bbox.x + bbox.w)))
        y1 = min(h, int(round(bbox.y + bbox.h)))
        if x1 <= x0 or y1 <= y0:
            return None
        return image.crop((x0, y0, x1, y1))


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #

# Re-exported for callers that orient crops themselves; the orchestrator relies on
# the recognizer's internal orientation but exposes the hook for parity/testing.
__all__ = ["VignocrPipeline", "orient"]


class _timed:
    """Context manager recording elapsed wall-time (ms) into ``store[key]``."""

    def __init__(self, store: dict[str, float], key: str) -> None:
        self._store = store
        self._key = key
        self._t0 = 0.0

    def __enter__(self) -> _timed:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._store[self._key] = (time.perf_counter() - self._t0) * 1000.0


def _torch_available() -> bool:
    """True if ``torch`` can be imported (the [ml] extra is installed)."""
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("torch") is not None


def _version_tag(path: str | None, fallback: str) -> str:
    """A compact version tag for a model artifact (basename) or a sensible default."""
    if path:
        return Path(path).name
    return fallback
