"""OCR recognition (Stage 2) — read text from per-field crops.

Stage 1 (``vignocr.detection``) localizes each field and hands us a crop + its
class. This package preprocesses that crop (orientation correction for the
rotated ``vin`` fields, field-type-aware grayscale/deskew/denoise/contrast),
recognizes the text with the configured backend, scores confidence, and either
returns ``status="ok"`` or **abstains** (``status="abstain"``) when confidence
falls below the flow's threshold — never a silent guess.

Public API (see ``docs/INTERFACES.md``)::

    from vignocr.ocr import Recognizer, orient
    rec = Recognizer(cfg)                 # cfg = configs/ocr/recognition.yaml dict (or None -> load)
    field_read = rec.read(crop, field_type="money", orientation="horizontal", flow="selling")

The OCR backend (PaddleOCR / TrOCR / Donut) is **lazy-imported** inside the
functions that use it, so this package imports and runs on CPU without the
``[ml]`` extra. Config drives everything; nothing here hardcodes class names,
thresholds, or alphabets.
"""

from vignocr.ocr.infer import Recognizer
from vignocr.ocr.preprocess import orient

__all__ = ["Recognizer", "orient"]
