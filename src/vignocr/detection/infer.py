"""Detection inference — backend-agnostic ``Detector`` over RF-DETR.

A single class loads *either* an RF-DETR torch checkpoint (``.pth``/``.pt``) or an
exported ONNX graph (``.onnx``) and yields a uniform ``list[Detection]``. Heavy
libs (torch / rfdetr / onnxruntime / numpy-heavy paths) are imported lazily inside
methods so importing this module never pulls the ``[ml]`` stack — the core CPU
package depends on it (``vignocr.detection`` re-exports ``Detection``/``Detector``).

Class index -> field name is resolved through ``vignocr.common.get_classes`` (the
single source of truth); no class name or count is hardcoded here.

NOTE: this module contains NO fixture/stub detector. The pipeline owns the
deterministic stub used when the ``[ml]`` extra is absent (see
``vignocr.pipeline``); ``Detector`` always talks to a real model/graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vignocr.common import BBox, get_logger, load_config
from vignocr.detection._resolve import resolve_class_schema, resolve_dataset

if TYPE_CHECKING:  # type-only imports; never executed at runtime (no ml dep on import)
    from PIL.Image import Image

log = get_logger(__name__)

_ML_HINT = "Detection inference needs the ML extra. Run: pip install -e .[ml]"


class _DetectorSchemaView:
    """Schema-shaped view (``num_classes`` + ``name_of``) over a flat class list.

    Each detection stage binds to its own class list (Stage A: 3 vignette classes;
    Stage B: 17 field classes from ``classes.yaml``). This tiny view lets the
    Detector treat both uniformly without depending on the field-schema dataclass.
    """

    def __init__(self, names: list[str], n: int) -> None:
        self._names: list[str] = list(names)
        self.num_classes: int = int(n)

    def name_of(self, idx: int) -> str:
        if 0 <= idx < len(self._names):
            return self._names[idx]
        return f"class_{idx}"


@dataclass(frozen=True)
class Detection:
    """One detected field box.

    Attributes:
        name:  class name from ``configs/classes.yaml`` (e.g. ``"ppa"``).
        score: confidence in ``[0, 1]``.
        bbox:  COCO ``[x, y, w, h]`` box in pixels of the *input* image.
    """

    name: str
    score: float
    bbox: BBox


def _is_onnx(path: Path) -> bool:
    return path.suffix.lower() == ".onnx"


def _load_label_map(ckpt_path: Path) -> list[str] | None:
    """Load the ordered id->name list persisted beside the checkpoint at train time.

    Looks for ``class_names.json`` in the checkpoint's directory (and its parent,
    in case the ckpt sits one level deeper than the run dir). Returns ``names``
    (index == class_id) or ``None`` if absent/unreadable. Stdlib only — keeps this
    module import-safe without the ``[ml]`` stack.
    """
    import json

    for d in (ckpt_path.parent, ckpt_path.parent.parent):
        f = d / "class_names.json"
        if f.is_file():
            try:
                names = json.loads(f.read_text(encoding="utf-8")).get("names")
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
            except (OSError, ValueError):
                continue
    return None


class Detector:
    """Loads an RF-DETR checkpoint or ONNX graph and detects vignette fields.

    Args:
        ckpt_or_onnx: path to a ``.onnx`` graph (-> onnxruntime backend) or an
            RF-DETR ``.pth``/``.pt`` checkpoint (-> torch backend).
        cfg_path: detector config (defaults to ``detection/rfdetr_medium``); used
            for resolution and the default score threshold.
        score_threshold: override the config's ``eval.score_threshold``.
        device: torch device for the torch backend (default ``"cpu"``). Ignored
            by the ONNX backend (uses available ONNX Runtime providers).

    The heavy model/session is loaded lazily on first ``detect`` (or via
    :meth:`load`), so merely constructing a ``Detector`` is cheap and import-safe.
    """

    def __init__(
        self,
        ckpt_or_onnx: str | Path,
        *,
        cfg_path: str = "detection/rfdetr_medium",
        score_threshold: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.path = Path(ckpt_or_onnx)
        if not self.path.exists():
            raise FileNotFoundError(f"detector weights not found: {self.path}")
        self._cfg = load_config(cfg_path)
        # Per-config dataset + class binding (Stage A uses 3 classes from
        # data.yaml/vignette; Stage B uses 17 from classes.yaml).
        _ds = resolve_dataset(self._cfg)
        _n, _names = resolve_class_schema(self._cfg, _ds)
        # AUTHORITATIVE label map: training persists `class_names.json` next to the
        # checkpoint (id->name from the COCO RF-DETR actually trained on, aliases
        # applied). It is the source of truth for class_id decoding — classes.yaml
        # ordering does NOT match a reconciled/real COCO. Fall back to the config
        # schema only when the sidecar is absent (e.g. an externally-trained ckpt).
        _mapped = _load_label_map(self.path)
        if _mapped:
            self._schema = _DetectorSchemaView(_mapped, len(_mapped))
            log.info("detector.label_map.loaded", n=len(_mapped), source="class_names.json")
        else:
            self._schema = _DetectorSchemaView(_names, _n)
            log.warning(
                "detector.label_map.fallback",
                note="no class_names.json beside checkpoint; decoding ids via config "
                "schema — verify this matches the model's training categories.",
            )
        self.resolution: int = int(self._cfg.get("model", {}).get("resolution", 640))
        eval_cfg = self._cfg.get("eval", {})
        self.score_threshold: float = (
            float(score_threshold)
            if score_threshold is not None
            else float(eval_cfg.get("score_threshold", 0.30))
        )
        self.device = device
        self.backend: str = "onnx" if _is_onnx(self.path) else "torch"
        # Lazily-populated handles (torch model OR onnxruntime session).
        self._model: Any | None = None
        self._session: Any | None = None
        log.info(
            "detector.init",
            path=str(self.path),
            backend=self.backend,
            resolution=self.resolution,
            score_threshold=self.score_threshold,
        )

    # ------------------------------------------------------------------ #
    # Loading (lazy; safe to call repeatedly)
    # ------------------------------------------------------------------ #
    def load(self) -> Detector:
        """Eagerly load the backend model/session. Returns self (chainable)."""
        if self.backend == "onnx":
            self._load_onnx()
        else:
            self._load_torch()
        return self

    def _load_torch(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401  (lazy; ensures clear error if missing)

            from vignocr.detection._resolve import resolve_model_class

            model_cls = resolve_model_class(self._cfg)
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_ML_HINT) from exc

        # rfdetr loads a trained checkpoint by passing the weights path to the
        # model constructor (pretrain_weights=...). The model CLASS follows
        # cfg.model.name so a nano/small checkpoint is loaded into the matching
        # architecture (previously hardcoded RFDETRMedium).
        num_classes = self._schema.num_classes
        self._model = model_cls(
            pretrain_weights=str(self.path),
            num_classes=num_classes,
            resolution=self.resolution,
        )
        # Put the underlying torch module in eval mode on the requested device when
        # the wrapper exposes it; RF-DETR keeps the nn.Module under `.model`.
        inner = getattr(self._model, "model", None)
        if inner is not None and hasattr(inner, "eval"):
            inner.eval()
            if hasattr(inner, "to"):
                inner.to(self.device)
        log.info("detector.torch.loaded", num_classes=num_classes, device=self.device)

    def _load_onnx(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_ML_HINT) from exc

        providers = ort.get_available_providers()
        # Prefer CUDA when present, always keep CPU as the fallback provider.
        ordered = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
        self._session = ort.InferenceSession(str(self.path), providers=ordered or providers)
        log.info("detector.onnx.loaded", providers=ordered or providers)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def detect(self, image: Image | str | Path) -> list[Detection]:
        """Detect vignette fields in ``image``.

        Args:
            image: a ``PIL.Image`` or a path to an image file.

        Returns:
            Detections with score >= ``score_threshold``, each carrying a
            class name from ``classes.yaml`` and a pixel-space ``BBox`` in the
            ORIGINAL image's coordinates (letterbox/resize is undone internally).
        """
        pil = self._as_pil(image)
        if self.backend == "onnx":
            return self._detect_onnx(pil)
        return self._detect_torch(pil)

    # ---- torch backend ------------------------------------------------ #
    def _detect_torch(self, pil: Image) -> list[Detection]:
        self._load_torch()
        assert self._model is not None
        # TODO(rfdetr-api): rfdetr's predict() returns a `supervision.Detections`
        # with .xyxy (abs pixels in the input image), .confidence, .class_id.
        # Confirm the threshold kwarg name (`threshold=`) against the installed
        # version; older builds used `conf=`/`confidence=`.
        result = self._model.predict(pil, threshold=self.score_threshold)
        xyxy = self._to_list(getattr(result, "xyxy", []))
        scores = self._to_list(getattr(result, "confidence", []))
        class_ids = self._to_list(getattr(result, "class_id", []))
        dets: list[Detection] = []
        for box, score, cid in zip(xyxy, scores, class_ids, strict=False):
            if score < self.score_threshold:
                continue
            name = self._safe_name(int(cid))
            if name is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append(
                Detection(
                    name=name,
                    score=float(score),
                    bbox=BBox(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1)),
                )
            )
        log.info("detector.detect", backend="torch", n=len(dets))
        return dets

    # ---- onnx backend ------------------------------------------------- #
    def _detect_onnx(self, pil: Image) -> list[Detection]:
        self._load_onnx()
        assert self._session is not None
        import numpy as np

        orig_w, orig_h = pil.size
        tensor, scale, pad = self._preprocess_onnx(pil)  # CHW float32 [1,3,R,R]
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: tensor})

        # RF-DETR ONNX emits two heads: bbox logits/boxes and class logits. Their
        # order can vary by export; identify by trailing dim (4 == boxes,
        # num_classes == logits) rather than by index, so we stay export-robust.
        boxes_raw, logits_raw = self._split_detr_outputs(outputs, self._schema.num_classes)
        boxes = np.asarray(boxes_raw, dtype="float32").reshape(-1, 4)
        logits = np.asarray(logits_raw, dtype="float32").reshape(boxes.shape[0], -1)

        # DETR-family: per-query sigmoid scores; take the top class per query.
        probs = 1.0 / (1.0 + np.exp(-logits))
        class_ids = probs.argmax(axis=1)
        scores = probs.max(axis=1)

        dets: list[Detection] = []
        for box, score, cid in zip(boxes, scores, class_ids, strict=False):
            if float(score) < self.score_threshold:
                continue
            name = self._safe_name(int(cid))
            if name is None:
                continue
            # Boxes come back as normalized cxcywh in [0,1] (DETR convention);
            # convert to xyxy at model resolution, undo letterbox, clamp to image.
            bbox = self._decode_box_to_original(box, scale, pad, orig_w, orig_h)
            dets.append(Detection(name=name, score=float(score), bbox=bbox))
        log.info("detector.detect", backend="onnx", n=len(dets))
        return dets

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_pil(image: Image | str | Path) -> Image:
        from PIL import Image as PILImage

        if isinstance(image, str | Path):
            return PILImage.open(image).convert("RGB")
        # Already a PIL image — ensure 3-channel RGB (band colour must survive).
        return image.convert("RGB") if image.mode != "RGB" else image

    def _preprocess_onnx(self, pil: Image) -> tuple[Any, float, tuple[int, int]]:
        """Letterbox to a square ``resolution`` and return CHW float32 [1,3,R,R].

        Returns ``(tensor, scale, (pad_x, pad_y))`` so boxes can be mapped back to
        the original image. Aspect ratio is preserved (no hue/colour change).
        """
        import numpy as np
        from PIL import Image as PILImage

        r = self.resolution
        w, h = pil.size
        scale = min(r / w, r / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = pil.resize((new_w, new_h), PILImage.BILINEAR)
        canvas = PILImage.new("RGB", (r, r), (114, 114, 114))  # neutral grey pad
        pad_x, pad_y = (r - new_w) // 2, (r - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))

        arr = np.asarray(canvas, dtype="float32") / 255.0  # HWC, [0,1]
        # ImageNet normalization (RF-DETR/DETR backbones expect it).
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
        chw = np.transpose(arr, (2, 0, 1))[None, ...]  # [1,3,R,R]
        return np.ascontiguousarray(chw, dtype="float32"), scale, (pad_x, pad_y)

    def _decode_box_to_original(
        self,
        box: Any,
        scale: float,
        pad: tuple[int, int],
        orig_w: int,
        orig_h: int,
    ) -> BBox:
        """Map one normalized cxcywh box (model space) back to original-image xywh."""
        r = self.resolution
        cx, cy, bw, bh = (float(v) for v in box)
        # Heuristic: normalized DETR boxes are in [0,1]; if values already exceed
        # ~1.5 we assume they are absolute pixels at model resolution.
        if max(cx, cy, bw, bh) <= 1.5:
            cx, cy, bw, bh = cx * r, cy * r, bw * r, bh * r
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        # Undo letterbox padding + scale.
        pad_x, pad_y = pad
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        w = bw / scale
        h = bh / scale
        # Clamp into the original image.
        x1 = min(max(0.0, x1), float(orig_w))
        y1 = min(max(0.0, y1), float(orig_h))
        w = max(0.0, min(w, float(orig_w) - x1))
        h = max(0.0, min(h, float(orig_h) - y1))
        return BBox(x=x1, y=y1, w=w, h=h)

    @staticmethod
    def _split_detr_outputs(outputs: list[Any], num_classes: int) -> tuple[Any, Any]:
        """Pick (boxes, logits) from the ONNX outputs by trailing dimension.

        Robust to export ordering: the 4-wide tensor is boxes; the
        ``num_classes``-wide tensor is the class logits.
        """
        import numpy as np

        boxes = logits = None
        for out in outputs:
            arr = np.asarray(out)
            last = arr.shape[-1] if arr.ndim else 0
            if last == 4 and boxes is None:
                boxes = arr
            elif last == num_classes and logits is None:
                logits = arr
        if boxes is None or logits is None:
            # Fall back to positional (boxes, logits) as a last resort.
            if len(outputs) >= 2:
                boxes = boxes if boxes is not None else np.asarray(outputs[0])
                logits = logits if logits is not None else np.asarray(outputs[1])
            else:
                raise ValueError(
                    "Unexpected ONNX outputs: could not identify boxes (last-dim 4) "
                    f"and logits (last-dim {num_classes}) among {len(outputs)} tensors."
                )
        return boxes, logits

    def _safe_name(self, cid: int) -> str | None:
        """Class idx -> name via classes.yaml; drop out-of-range ids (e.g. no-object)."""
        try:
            return self._schema.name_of(cid)
        except KeyError:
            return None

    @staticmethod
    def _to_list(obj: Any) -> list:
        """Coerce a tensor/ndarray/sequence to a plain Python list (no torch import)."""
        if obj is None:
            return []
        for attr in ("tolist", "cpu"):
            if hasattr(obj, attr):
                try:
                    return obj.tolist() if attr == "tolist" else obj.cpu().numpy().tolist()
                except Exception:  # noqa: BLE001 - fall through to generic handling
                    pass
        return list(obj)
