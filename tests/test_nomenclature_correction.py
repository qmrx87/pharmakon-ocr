"""Nomenclature correction — the medical-safety core.

Driven entirely by ``configs/nomenclature/correction.yaml: policy`` (never
hardcoded here). The fixtures pick a real row from the generated index at
runtime, so the tests are seed-independent.

Asserted behaviours:
* a clean match repairs ``product_name`` / ``laboratoire`` from the register;
* ``ppa`` / ``tr`` are NEVER changed (vignette-specific);
* a dosage CONFLICT (confident OCR disagreeing with the register) is FLAGGED
  (``status == "conflict"``) with the OCR value KEPT — never overwritten;
* a partially-misread code still matches within ``max_edit_distance``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vignocr.common import FieldRead, load_config
from vignocr.nomenclature import correct as correct_mod
from vignocr.nomenclature import loader, match


@pytest.fixture(scope="module")
def correction_cfg() -> dict[str, Any]:
    """The ``correction.yaml`` policy/match/csv config."""
    return load_config("nomenclature/correction")


@pytest.fixture
def index(nomenclature_csv: Path) -> loader.NomenclatureIndex:
    """The nomenclature index loaded from the generated fixture CSV."""
    idx = loader.load_csv(nomenclature_csv)
    assert not idx.is_empty, "fixture nomenclature CSV produced an empty index"
    return idx


@pytest.fixture
def sample_row(index: loader.NomenclatureIndex) -> dict[str, str]:
    """A deterministic register row to correct against (first by sorted key)."""
    key = sorted(index.by_key)[0]
    return index.by_key[key]


def _anchor_field(row: dict[str, str], cfg: dict[str, Any]) -> FieldRead:
    """A confident OCR read of the row's enregistrement code (the anchor)."""
    key_col = cfg["csv"]["key_column"]
    return FieldRead(
        name="num_enregistrement",
        value=row[key_col],
        raw=row[key_col],
        confidence=0.99,
        status="ok",
        source="ocr",
    )


# --------------------------------------------------------------------------- #
# Clean match: identity repaired; ppa/tr never touched
# --------------------------------------------------------------------------- #


def test_clean_match_repairs_identity_fields(
    index: loader.NomenclatureIndex, sample_row: dict, correction_cfg: dict
) -> None:
    """An exact code match repairs product_name and laboratoire from the register."""
    code = sample_row[correction_cfg["csv"]["key_column"]]
    row, conf = match.find(code, index, correction_cfg)
    assert row is not None and conf == pytest.approx(1.0)

    fields = {
        "num_enregistrement": _anchor_field(sample_row, correction_cfg),
        # product_name misread + low confidence -> repair_always overwrites it.
        "product_name": FieldRead(
            name="product_name", value="MISREAD", raw="MISREAD",
            confidence=0.40, status="ok", source="ocr",
        ),
        # laboratoire missing -> filled from the register.
        "laboratoire": FieldRead(name="laboratoire", status="missing", source="none"),
    }
    out, report = correct_mod.apply(fields, row, correction_cfg)

    assert report.matched is True
    assert out["product_name"].value == sample_row["product_name"]
    assert out["product_name"].status == "corrected"
    assert out["product_name"].source == "nomenclature"

    assert out["laboratoire"].value == sample_row["laboratoire"]
    assert out["laboratoire"].status == "corrected"
    assert out["laboratoire"].source == "nomenclature"


def test_ppa_and_tr_are_never_overwritten(
    index: loader.NomenclatureIndex, sample_row: dict, correction_cfg: dict
) -> None:
    """``ppa`` and ``tr`` are in ``never_overwrite``: the register cannot touch them.

    Even though the matched row carries a ``tr`` value, the OCR ``ppa`` read must
    be returned byte-for-byte identical, and ``tr`` must not be injected as a field.
    """
    code = sample_row[correction_cfg["csv"]["key_column"]]
    row, _ = match.find(code, index, correction_cfg)

    ppa_read = FieldRead(
        name="ppa", value="123.45", raw="PPA = 123.45 DA",
        confidence=0.99, status="ok", source="ocr",
    )
    fields = {
        "num_enregistrement": _anchor_field(sample_row, correction_cfg),
        "ppa": ppa_read,
    }
    out, _report = correct_mod.apply(fields, row, correction_cfg)

    # ppa returned unchanged (same value/status/source), never corrected.
    assert out["ppa"].value == "123.45"
    assert out["ppa"].status == "ok"
    assert out["ppa"].source == "ocr"
    # tr is never materialized as a corrected field from the register.
    assert "tr" not in out or out["tr"].source != "nomenclature"


# --------------------------------------------------------------------------- #
# Dispensing-critical conflict: KEEP OCR + flag (never overwrite)
# --------------------------------------------------------------------------- #


def test_dosage_conflict_is_flagged_and_ocr_kept(
    index: loader.NomenclatureIndex, sample_row: dict, correction_cfg: dict
) -> None:
    """A confident OCR dosage that disagrees with the register is flagged, not overwritten.

    ``dosage`` is in ``flag_on_conflict`` (dispensing-critical). On a confident
    disagreement the OCR value is KEPT, status becomes ``conflict``, and the
    register value is reported as a conflict — never silently substituted.
    """
    code = sample_row[correction_cfg["csv"]["key_column"]]
    row, _ = match.find(code, index, correction_cfg)

    register_dosage = sample_row["dosage"]
    wrong_dosage = register_dosage + " XL"  # confidently different from the register
    fields = {
        "num_enregistrement": _anchor_field(sample_row, correction_cfg),
        "dosage": FieldRead(
            name="dosage", value=wrong_dosage, raw=wrong_dosage,
            confidence=0.97, status="ok", source="ocr",
        ),
    }
    out, report = correct_mod.apply(fields, row, correction_cfg)

    # OCR value KEPT (never overwritten), but flagged for human review.
    assert out["dosage"].value == wrong_dosage
    assert out["dosage"].status == "conflict"
    assert out["dosage"].source == "ocr"

    # The conflict is reported with both sides and a 'flagged' action.
    flagged = [c for c in report.conflicts if c.field == "dosage"]
    assert len(flagged) == 1
    assert flagged[0].ocr == wrong_dosage
    assert flagged[0].nomenclature == register_dosage
    assert flagged[0].action == "flagged"


def test_dosage_agreement_is_high_trust(
    index: loader.NomenclatureIndex, sample_row: dict, correction_cfg: dict
) -> None:
    """When OCR dosage agrees with the register, it is confirmed (no conflict)."""
    code = sample_row[correction_cfg["csv"]["key_column"]]
    row, _ = match.find(code, index, correction_cfg)

    fields = {
        "num_enregistrement": _anchor_field(sample_row, correction_cfg),
        "dosage": FieldRead(
            name="dosage", value=sample_row["dosage"], raw=sample_row["dosage"],
            confidence=0.97, status="ok", source="ocr",
        ),
    }
    out, report = correct_mod.apply(fields, row, correction_cfg)
    assert out["dosage"].value == sample_row["dosage"]
    assert out["dosage"].status == "ok"
    assert out["dosage"].source == "ocr+nomenclature"
    assert [c for c in report.conflicts if c.field == "dosage"] == []


# --------------------------------------------------------------------------- #
# Fuzzy matching: a partially-misread code still matches within max_edit_distance
# --------------------------------------------------------------------------- #


def test_partial_misread_matches_within_edit_distance(
    index: loader.NomenclatureIndex, sample_row: dict, correction_cfg: dict
) -> None:
    """A single corrupted digit still resolves to the right register row."""
    key_col = correction_cfg["csv"]["key_column"]
    code = sample_row[key_col]

    # Corrupt exactly one digit in the trailing block (within max_edit_distance).
    blocks = match.split_blocks(match.normalize_code(code, correction_cfg))
    assert blocks is not None, "sample code must be structurally parseable"
    digit = blocks["e"][-1]
    swapped = "9" if digit != "9" else "8"
    corrupted = code[:-1] + swapped
    assert corrupted != code

    row, conf = match.find(corrupted, index, correction_cfg)
    assert row is not None, "a 1-edit misread should still match"
    assert row[key_col] == code
    assert conf >= correction_cfg["match"]["min_match_confidence"]


def test_far_misread_does_not_match(
    index: loader.NomenclatureIndex, correction_cfg: dict
) -> None:
    """A code far from every register key (beyond max_edit_distance) does NOT match."""
    # A structurally-valid code with many edits from any real key.
    row, _conf = match.find("11/11/11Z111/111", index, correction_cfg)
    assert row is None


def test_unmatched_anchor_keeps_ocr_values(
    index: loader.NomenclatureIndex, correction_cfg: dict
) -> None:
    """When no row matches, identity fields keep their OCR values and matched=False."""
    fields = {
        "num_enregistrement": FieldRead(
            name="num_enregistrement", value="11/11/11Z111/111",
            confidence=0.99, status="ok", source="ocr",
        ),
        "product_name": FieldRead(
            name="product_name", value="OCRNAME", raw="OCRNAME",
            confidence=0.95, status="ok", source="ocr",
        ),
    }
    row, _ = match.find("11/11/11Z111/111", index, correction_cfg)
    out, report = correct_mod.apply(fields, row, correction_cfg)
    assert report.matched is False
    assert out["product_name"].value == "OCRNAME"  # unchanged
    assert out["product_name"].source == "ocr"
