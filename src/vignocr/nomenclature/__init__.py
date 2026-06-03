"""Nomenclature correction — repair identity fields from the official register.

Pure CPU, ``rapidfuzz``-backed. The engine keys off the ``num_enregistrement``
anchor: it normalizes the OCR-read code, fuzzy-matches it against the nomenclature
index (structural edit distance with per-block weights), and then applies the
correction **policy** from ``configs/nomenclature/correction.yaml``.

Safety core (policy, never hardcoded here):
    * ``never_overwrite`` (ppa, tr)            -> nomenclature MUST NOT touch.
    * ``repair_always`` (product_name, lab.)   -> nomenclature is source of truth.
    * ``repair_if_ocr_low_or_agree`` (dci)     -> use nomenclature only if OCR
                                                  abstained/low-conf OR they agree.
    * ``flag_on_conflict`` (dosage, forme)     -> dispensing-critical: on a confident
                                                  disagreement, KEEP OCR + flag it.

Public API (import from ``vignocr.nomenclature``)::

    loader.load_csv(path) -> NomenclatureIndex
    match.find(norm_code, index, cfg) -> (row | None, confidence)
    correct.apply(fields, row, cfg) -> (dict[str, FieldRead], NomenclatureReport)
"""

from __future__ import annotations

from vignocr.nomenclature.correct import apply
from vignocr.nomenclature.loader import NomenclatureIndex, load_csv
from vignocr.nomenclature.match import find, normalize_code

__all__ = [
    "NomenclatureIndex",
    "load_csv",
    "find",
    "normalize_code",
    "apply",
]
