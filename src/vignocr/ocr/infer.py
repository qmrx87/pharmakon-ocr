"""Recognition inference — turn a field crop into a :class:`FieldRead`.

:class:`Recognizer` wraps the configured OCR backend (PaddleOCR baseline, or a
TrOCR/Donut transformer behind the config switch), preprocesses each crop for
its field type, recognizes the text, scores confidence, and applies the
**abstention gate**: when confidence is below the active flow's threshold the
read is returned with ``status="abstain"`` (raw text kept, flagged à vérifier)
instead of being silently accepted.

The backend is **lazy-imported** inside :meth:`Recognizer._backend`; importing
this module never requires the ``[ml]`` extra. The thresholds are read from
``configs/parsing/fields.yaml: abstention`` (single source of truth) — selling
is stricter than receiving.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

from vignocr.common import FieldRead, get_classes, get_logger, load_config
from vignocr.ocr.preprocess import preprocess_for_type

if TYPE_CHECKING:  # type-only import; never executed at runtime
    from PIL import Image

log = get_logger(__name__)

# Field *types* (classes.yaml `type`) that carry no recognizable text. These are
# structural/colour regions handled elsewhere (reimbursability head, layout), so
# the recognizer abstains on them rather than feeding them to OCR.
_NON_TEXT_TYPES = {"region"}

_WHITESPACE = re.compile(r"\s+")


class Recognizer:
    """Per-field text recognizer driven by ``configs/ocr/recognition.yaml``.

    Args:
        cfg: the parsed ``recognition.yaml`` mapping. ``None`` (default) loads it
            via :func:`vignocr.common.load_config`. Passing a dict keeps the
            class testable and lets the pipeline inject an already-loaded config.

    The heavy backend is constructed lazily on first :meth:`read`, so building a
    ``Recognizer`` is cheap and import-safe on a CPU-only box.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg: dict[str, Any] = cfg if cfg is not None else load_config("ocr/recognition")
        # Backend selection is env-overridable (12-factor) without editing yaml.
        self.backend_name: str = os.environ.get(
            "VIGNOCR_OCR_BACKEND", self.cfg.get("backend", "baseline")
        )
        self._field_types: dict[str, Any] = self.cfg.get("field_types", {}) or {}
        self._conf_cfg: dict[str, Any] = self.cfg.get("confidence", {}) or {}
        self._abstention = _load_abstention(self._conf_cfg)
        self._backend_obj: Any | None = None  # built lazily
        self._classes = get_classes()
        log.info("recognizer.init", backend=self.backend_name)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def read(
        self,
        crop: Image.Image,
        field_type: str,
        orientation: str,
        *,
        flow: str | None = None,
        field_name: str | None = None,
    ) -> FieldRead:
        """Recognize ``crop`` and return a :class:`FieldRead`.

        Args:
            crop: the field crop (PIL image) from Stage-1 detection.
            field_type: the field ``type`` from ``classes.yaml``
                (``money`` | ``code`` | ``date`` | ``text`` | ``region``).
            orientation: the field ``orientation`` from ``classes.yaml``; vertical
                crops are rotated upright before recognition (see ``preprocess.orient``).
            flow: ``"selling"`` | ``"receiving"`` — which abstention profile to
                apply. Defaults to ``confidence.default_flow`` from the config.
                Selling is stricter (a wrong dispense is unacceptable).
            field_name: optional concrete field name (e.g. ``"num_lot"``) used to
                pick a per-field abstention override if one is configured.

        Returns:
            ``FieldRead`` with ``raw`` = backend text, ``value`` = the same text
            (deterministic normalization to Decimal/date/code is the ``parsing``
            module's job — recognition stays a faithful transcriber), a
            ``confidence`` in ``[0, 1]``, ``source="ocr"`` and ``status`` either
            ``"ok"`` (>= threshold) or ``"abstain"`` (< threshold). Region/non-text
            types and empty reads abstain.
        """
        flow = flow or self._conf_cfg.get("default_flow", "selling")
        name = field_name or field_type

        # Structural/colour regions are not text — abstain without OCR.
        if field_type in _NON_TEXT_TYPES:
            return FieldRead(
                name=name, value=None, raw=None, confidence=0.0, status="abstain", source="none"
            )

        type_cfg = self._field_types.get(field_type, {})
        prepared = preprocess_for_type(crop, orientation, type_cfg)

        text, score = self._recognize(prepared, type_cfg)
        text = _collapse_ws(text)
        confidence = self._confidence(text, score)

        threshold = self._threshold(flow, name)
        status = "ok" if confidence >= threshold else "abstain"
        if status == "abstain":
            log.debug("recognizer.abstain", field=name, flow=flow, conf=confidence, tau=threshold)

        # Keep raw text either way; on abstain the downstream HITL still sees it.
        return FieldRead(
            name=name,
            value=text or None,
            raw=text or None,
            confidence=confidence,
            status=status,
            source="ocr",
        )

    # ------------------------------------------------------------------ #
    # Confidence + abstention
    # ------------------------------------------------------------------ #

    def _confidence(self, text: str, score: float) -> float:
        """Map a backend score to ``FieldRead.confidence`` per ``confidence.scoring``."""
        if not text.strip():
            return float(self._conf_cfg.get("empty_read_confidence", 0.0))
        # ``backend_score`` (baseline): PaddleOCR already returns a 0..1 line score.
        # ``mean_token_prob`` (transformer): the seq2seq path computes its own and
        # passes it through ``score`` — identity here either way.
        return max(0.0, min(1.0, float(score)))

    def _threshold(self, flow: str, field_name: str) -> float:
        """Resolve the abstention threshold for ``flow`` (+ per-field override)."""
        profile = self._abstention.get(flow, {})
        if field_name in profile:  # explicit per-field override, e.g. selling.num_lot
            return float(profile[field_name])
        if "default" in profile:
            return float(profile["default"])
        # Fail safe: if a flow profile is missing, abstain on everything (1.0)
        # rather than silently accepting unvetted reads.
        log.warning("recognizer.no_threshold", flow=flow)
        return 1.0

    # ------------------------------------------------------------------ #
    # Backend dispatch (lazy)
    # ------------------------------------------------------------------ #

    def _recognize(self, crop: Image.Image, type_cfg: dict[str, Any]) -> tuple[str, float]:
        """Run the configured backend on a preprocessed crop -> (text, score)."""
        if self.backend_name == "baseline":
            return self._recognize_paddle(crop, type_cfg)
        if self.backend_name == "transformer":
            return self._recognize_transformer(crop, type_cfg)
        raise ValueError(
            f"unknown OCR backend {self.backend_name!r}; "
            "expected 'baseline' or 'transformer' (configs/ocr/recognition.yaml: backend)"
        )

    def _backend(self) -> Any:
        """Build (once) and return the heavy backend handle. Lazy-imports the lib."""
        if self._backend_obj is not None:
            return self._backend_obj
        if self.backend_name == "baseline":
            self._backend_obj = self._build_paddle()
        elif self.backend_name == "transformer":
            self._backend_obj = self._build_transformer()
        else:  # pragma: no cover - guarded in _recognize
            raise ValueError(f"unknown OCR backend {self.backend_name!r}")
        return self._backend_obj

    # ---- PaddleOCR (baseline) ----------------------------------------- #

    def _build_paddle(self) -> Any:
        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415  (lazy: keeps core CPU-only)
        except ImportError as exc:  # pragma: no cover - exercised only without [ml]
            raise ImportError(
                "PaddleOCR is required for the 'baseline' OCR backend but is not "
                "installed. Install the ML extra:  pip install -e .[ml]\n"
                "(or switch configs/ocr/recognition.yaml: backend -> transformer)."
            ) from exc

        b = (self.cfg.get("backends", {}) or {}).get("baseline", {}) or {}
        use_gpu = os.environ.get("VIGNOCR_OCR_USE_GPU", "1" if b.get("use_gpu") else "0") == "1"
        # Recognition-only: detection is RF-DETR's job upstream.
        return PaddleOCR(
            lang=b.get("lang", "fr"),
            use_angle_cls=bool(b.get("use_angle_cls", False)),
            det=bool(b.get("use_detection", False)),
            rec_model_dir=b.get("rec_model_dir") or None,
            drop_score=float(b.get("drop_score", 0.0)),
            use_gpu=use_gpu,
            show_log=False,
        )

    def _recognize_paddle(self, crop: Image.Image, type_cfg: dict[str, Any]) -> tuple[str, float]:
        import numpy as np  # noqa: PLC0415  (core dep, imported lazily for symmetry)

        ocr = self._backend()
        arr = np.asarray(crop.convert("RGB") if crop.mode != "RGB" else crop)
        # det=False -> recognition-only; PaddleOCR returns [[(text, score)], ...].
        result = ocr.ocr(arr, det=False, cls=False)
        text, score = _first_paddle_line(result)
        # NOTE: char_whitelist (type_cfg['char_whitelist']) is enforced at the
        # decoder level by shipping a per-field char dict to a fine-tuned rec
        # model (see train.run). With the stock model we keep the raw read and
        # let parsing/nomenclature apply the confusion map — we never silently
        # delete out-of-alphabet glyphs here, so confidence stays honest.
        return text, score

    # ---- TrOCR / Donut (transformer scaffold) ------------------------- #

    def _build_transformer(self) -> Any:
        """Build the seq2seq processor+model. Scaffolded; needs the [ml] extra + transformers.

        TODO(transformer): ``transformers`` is not yet pinned in pyproject ``[ml]``.
        Add it there before enabling this backend in production.
        """
        try:
            import torch  # noqa: F401, PLC0415  (lazy)
            from transformers import (  # noqa: PLC0415
                AutoProcessor,
                VisionEncoderDecoderModel,
            )
        except ImportError as exc:  # pragma: no cover - scaffold path
            raise ImportError(
                "The 'transformer' OCR backend needs torch + transformers, which "
                "are not installed. Install the ML extra and transformers:\n"
                "  pip install -e .[ml] transformers\n"
                "(or use the default 'baseline' PaddleOCR backend)."
            ) from exc

        t = (self.cfg.get("backends", {}) or {}).get("transformer", {}) or {}
        engine = t.get("engine", "trocr")
        variant = (t.get("variants", {}) or {}).get(engine, {}) or {}
        model_name = variant.get("model_name")
        # TODO(transformer): the exact processor/model classes differ between
        # TrOCR (VisionEncoderDecoder) and Donut (also VisionEncoderDecoder but
        # with a task prompt + different decoding). Confirmed for TrOCR; Donut
        # decoding (task_prompt -> structured tokens) is assumed and untested.
        processor = AutoProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.eval()
        return {"engine": engine, "variant": variant, "processor": processor, "model": model}

    def _recognize_transformer(
        self, crop: Image.Image, type_cfg: dict[str, Any]
    ) -> tuple[str, float]:
        # Build first: _build_transformer carries the friendly ImportError guard,
        # so a missing torch/transformers surfaces the "pip install -e .[ml]"
        # message rather than a bare ModuleNotFoundError from the line below.
        handle = self._backend()
        import torch  # noqa: PLC0415

        processor, model = handle["processor"], handle["model"]
        variant = handle["variant"]
        max_new_tokens = int(variant.get("max_new_tokens", 32))

        pixel_values = processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            out = model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        # Mean per-token softmax prob over the generated steps -> a 0..1 confidence.
        # TODO(transformer): validate this scoring against held-out reads; the
        # exact ``generate`` score API can vary across transformers versions.
        score = _seq2seq_confidence(out, torch)
        return text, score


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_abstention(conf_cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the abstention thresholds referenced by ``recognition.yaml: confidence``.

    The values live in ``parsing/fields.yaml`` (single source of truth); we only
    hold the *reference* in our config and resolve it here.
    """
    cfg_name = conf_cfg.get("abstention_thresholds_config", "parsing/fields")
    key = conf_cfg.get("abstention_thresholds_key", "abstention")
    parsing_cfg = load_config(cfg_name)
    return parsing_cfg.get(key, {}) or {}


def _collapse_ws(text: str | None) -> str:
    """Trim + collapse internal whitespace runs to single spaces."""
    if not text:
        return ""
    return _WHITESPACE.sub(" ", text).strip()


def _first_paddle_line(result: Any) -> tuple[str, float]:
    """Extract ``(text, score)`` from PaddleOCR's recognition-only output.

    PaddleOCR (det=False) returns ``[[(text, score)], ...]`` but the exact nesting
    has shifted across releases, so we probe defensively and fall back to an empty
    low-confidence read rather than raising on an unexpected shape.
    """
    try:
        if not result:
            return "", 0.0
        line = result[0]
        pair = line[0] if isinstance(line, list | tuple) and line else line
        if isinstance(pair, list | tuple) and len(pair) >= 2:
            return str(pair[0]), float(pair[1])
    except (IndexError, TypeError, ValueError):
        log.warning("paddle.unexpected_result_shape")
    return "", 0.0


def _seq2seq_confidence(out: Any, torch: Any) -> float:
    """Mean per-step max-softmax probability over generated tokens (0..1)."""
    try:
        scores = getattr(out, "scores", None)
        if not scores:
            return 0.0
        probs = [torch.softmax(s[0], dim=-1).max().item() for s in scores]
        return float(sum(probs) / len(probs)) if probs else 0.0
    except (RuntimeError, ValueError, TypeError, IndexError):  # pragma: no cover - scaffold
        return 0.0
