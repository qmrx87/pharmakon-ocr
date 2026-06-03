"""End-to-end pipeline — compose every stage into one extractor.

``VignocrPipeline.extract(image, flow=...)`` runs the full flow and returns an
:class:`~vignocr.common.schemas.ExtractionRecord`:

    preprocess → detect → per-field orient+crop → recognize → PPA disambiguate →
    money checksum + abstention (record.build) → nomenclature correct →
    reimbursability(color_band) → assemble (model_versions + timings_ms)

The detector and recognizer are the real RF-DETR / OCR backends when the ``[ml]``
extra and trained weights are present, and deterministic fixture-backed **stubs**
(:class:`~vignocr.pipeline.stubs.StubDetector` /
:class:`~vignocr.pipeline.stubs.StubRecognizer`) otherwise — so the whole pipeline
runs on a CPU-only box today. Backend selection is config-driven
(``pipeline.backend``: ``auto`` | ``stub`` | ``real``). Importing this package
loads **no** ML libs (they are lazy-imported inside the real backends).

Public API (see ``docs/INTERFACES.md``)::

    from vignocr.pipeline import VignocrPipeline
    record = VignocrPipeline(cfg).extract(image, flow="selling")
"""

from __future__ import annotations

from vignocr.pipeline.orchestrator import VignocrPipeline
from vignocr.pipeline.reimbursability import classify_band
from vignocr.pipeline.stubs import StubDetector, StubRecognizer

__all__ = [
    "VignocrPipeline",
    "StubDetector",
    "StubRecognizer",
    "classify_band",
]
