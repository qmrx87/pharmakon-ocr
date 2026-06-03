"""Deterministic parsing layer — pure CPU, no ML.

Normalizes raw OCR reads into canonical, validated values and runs the business
checks that don't need a model: money parsing (``Decimal`` end-to-end), date
parsing, enregistrement-code normalization, the ``prix + shp == ppa`` checksum,
PPA disambiguation, and partial-record assembly with the per-flow abstention
policy. Every rule comes from ``configs/parsing/fields.yaml`` via
``vignocr.common.load_config``.

Public API (see ``docs/INTERFACES.md``)::

    money.parse(text) -> Decimal | None
    dates.parse(text, formats) -> date | None ; dates.is_after(exp, fab) -> bool
    codes.normalize_enregistrement(text) -> str | None
    checksum.verify_and_repair(prix, shp, ppa) -> (dict[str, FieldRead], ChecksumReport)
    ppa.disambiguate(candidates) -> FieldRead
    record.build(fields, flow) -> ExtractionRecord   # partial (no nomenclature)
"""

from vignocr.parsing import checksum, codes, dates, money, ppa, record

__all__ = ["money", "dates", "codes", "checksum", "ppa", "record"]
