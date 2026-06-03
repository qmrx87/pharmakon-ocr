"""Deterministic parsing: money locales, the ``prix+shp==ppa`` checksum, PPA.

Money is :class:`decimal.Decimal` end-to-end and serialized as a centime-quantized
string (``"250.00"``). These tests assert on those Decimal-strings, never floats,
and cover the business-critical behaviours:

* golden money parses across locales (``","`` / ``"."``, thousands marks, ``DA``);
* ``prix + shp == ppa`` exact to the centime (ok);
* a REPAIR case — one field missing, recomputed exactly from the other two;
* a MISMATCH case — all three present but inconsistent, flagged (not accepted);
* PPA disambiguation — the final ``= XXX,XX DA`` value wins over the
  intermediate ``a + b`` form, even when their numbers differ.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from vignocr.common import FieldRead, money_str
from vignocr.parsing import checksum as checksum_mod
from vignocr.parsing import codes, money
from vignocr.parsing import ppa as ppa_mod

# --------------------------------------------------------------------------- #
# Money parsing — golden locale cases (Decimal, never float)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("250,00", "250.00"),          # comma decimal
        ("250.00", "250.00"),          # dot decimal
        ("1234.56", "1234.56"),        # plain integer-part, dot decimal
        ("1 234,56 DA", "1234.56"),    # space thousands + comma decimal + currency
        ("1 234,56", "1234.56"),       # no-break space thousands separator
        ("1'234.50", "1234.50"),       # apostrophe thousands separator
        ("702,56 DA", "702.56"),       # trailing currency token
        ("700.06", "700.06"),          # the intermediate-style dotted value
        ("PPA = 262.31 DA", "262.31"), # labelled + currency: capture the run only
        ("0,00", "0.00"),              # the min bound
    ],
)
def test_money_parse_golden(text: str, expected: str) -> None:
    """Money parses to the expected centime-quantized Decimal."""
    value = money.parse(text)
    assert isinstance(value, Decimal)
    assert money_str(value) == expected
    # Quantized to exactly two decimal places.
    assert value == value.quantize(Decimal("0.01"))


@pytest.mark.parametrize("text", ["", "abc", "DA", "   ", None])
def test_money_parse_rejects_non_money(text: str | None) -> None:
    """Non-money text yields ``None`` (never a silent zero or a guess)."""
    assert money.parse(text) is None


def test_money_parse_out_of_bounds_abstains() -> None:
    """A value above the configured ``max_value`` sanity bound abstains (``None``)."""
    # 100000.00 is the configured ceiling; above it -> abstain.
    assert money.parse("250000.00") is None


def test_money_is_decimal_not_float() -> None:
    """The parser returns ``Decimal`` (the contract forbids float money)."""
    value = money.parse("1 234,56 DA")
    assert type(value) is Decimal


# --------------------------------------------------------------------------- #
# Checksum: prix + shp == ppa (ok / repaired / mismatch)
# --------------------------------------------------------------------------- #


def _fr(name: str, value: str | None, conf: float, status: str = "ok") -> FieldRead:
    return FieldRead(name=name, value=value, raw=value, confidence=conf, status=status, source="ocr")


def test_checksum_ok_exact_to_the_centime() -> None:
    """All three present and consistent -> verdict ``ok``; values unchanged."""
    fields, report = checksum_mod.verify_and_repair(
        _fr("prix", "235.99", 0.95),
        _fr("shp", "26.32", 0.95),
        _fr("ppa", "262.31", 0.95),
    )
    assert report.verdict == "ok"
    assert report.repaired_field is None
    assert (report.prix, report.shp, report.ppa) == ("235.99", "26.32", "262.31")
    for name in ("prix", "shp", "ppa"):
        assert fields[name].status == "ok"


def test_checksum_repairs_missing_shp_exactly() -> None:
    """One field missing + two confident anchors -> recompute it to the centime."""
    fields, report = checksum_mod.verify_and_repair(
        _fr("prix", "235.99", 0.95),
        FieldRead(name="shp", value=None, raw=None, confidence=0.0, status="missing"),
        _fr("ppa", "262.31", 0.95),
    )
    assert report.verdict == "repaired"
    assert report.repaired_field == "shp"
    # 262.31 - 235.99 == 26.32, exactly.
    assert fields["shp"].value == "26.32"
    assert fields["shp"].status == "corrected"
    assert fields["shp"].source == "checksum"
    assert report.shp == "26.32"


def test_checksum_repairs_missing_ppa_exactly() -> None:
    """PPA recomputed from prix + shp when PPA is the missing field."""
    fields, report = checksum_mod.verify_and_repair(
        _fr("prix", "306.49", 0.95),
        _fr("shp", "12.50", 0.95),
        FieldRead(name="ppa", value=None, raw=None, confidence=0.0, status="missing"),
    )
    assert report.verdict == "repaired"
    assert report.repaired_field == "ppa"
    assert fields["ppa"].value == "318.99"  # 306.49 + 12.50
    assert fields["ppa"].source == "checksum"


def test_checksum_mismatch_is_flagged_not_accepted() -> None:
    """Three present but inconsistent -> verdict ``mismatch`` and fields flagged.

    A failed checksum must never be silently accepted: every involved field is
    marked ``status="conflict"``.
    """
    fields, report = checksum_mod.verify_and_repair(
        _fr("prix", "100.00", 0.95),
        _fr("shp", "5.00", 0.95),
        _fr("ppa", "999.99", 0.95),  # 100 + 5 != 999.99
    )
    assert report.verdict == "mismatch"
    assert report.repaired_field is None
    for name in ("prix", "shp", "ppa"):
        assert fields[name].status == "conflict", f"{name} should be flagged on mismatch"


def test_checksum_low_confidence_does_not_anchor_a_repair() -> None:
    """Two confident anchors are required to repair; a low-conf pair cannot.

    With prix missing and only a low-confidence shp+ppa, there are two
    non-anchored fields, so no repair fires and the verdict stays ``incomplete``.
    """
    fields, report = checksum_mod.verify_and_repair(
        FieldRead(name="prix", value=None, raw=None, confidence=0.0, status="missing"),
        _fr("shp", "5.00", 0.10),
        _fr("ppa", "999.99", 0.10),
    )
    assert report.verdict == "incomplete"
    assert report.repaired_field is None
    assert fields["prix"].value is None


# --------------------------------------------------------------------------- #
# PPA disambiguation: final "= XXX,XX DA" wins over intermediate "a + b"
# --------------------------------------------------------------------------- #


def test_ppa_prefers_final_over_intermediate() -> None:
    """The final ``= XXX,XX DA`` value is chosen over the intermediate ``a + b``.

    The two candidates carry *different* numbers (sum=702.56 vs final=999.99) so
    the test genuinely proves the final line wins, not a coincidental equality.
    """
    candidates = [
        # Intermediate additive form (a + b = 702.56), highest confidence.
        FieldRead(name="ppa", raw="PPA: 700,06+2,50", value=None, confidence=0.99, status="ok"),
        # Final total — distinct value.
        FieldRead(name="ppa", raw="PPA = 999,99 DA", value=None, confidence=0.80, status="ok"),
    ]
    chosen = ppa_mod.disambiguate(candidates)
    assert chosen.value == "999.99", "must pick the final '= XXX,XX DA' value"
    assert chosen.status == "ok"


def test_ppa_resolves_intermediate_when_no_final() -> None:
    """With only the intermediate ``a + b`` form, it resolves to the sum."""
    chosen = ppa_mod.disambiguate(
        [FieldRead(name="ppa", raw="700,06+2,50", value=None, confidence=0.9, status="ok")]
    )
    assert chosen.value == "702.56"  # 700.06 + 2.50
    assert chosen.status == "ok"


def test_ppa_plain_value_when_no_markers() -> None:
    """A plain money read (no '='/'+') is parsed as-is."""
    chosen = ppa_mod.disambiguate(
        [FieldRead(name="ppa", raw="262,31", value=None, confidence=0.9, status="ok")]
    )
    assert chosen.value == "262.31"


def test_ppa_empty_candidates_is_missing() -> None:
    """No candidates -> a ``missing`` PPA placeholder (checksum may anchor later)."""
    chosen = ppa_mod.disambiguate([])
    assert chosen.name == "ppa"
    assert chosen.status == "missing"
    assert chosen.value is None


# --------------------------------------------------------------------------- #
# Enregistrement code normalization (deterministic, confusion-corrected)
# --------------------------------------------------------------------------- #


def test_code_normalize_strips_spacing() -> None:
    """Spaces around the letter block are stripped to the canonical form."""
    assert codes.normalize_enregistrement("18/97/14G 061/003") == "18/97/14G061/003"
    assert codes.normalize_enregistrement("16/99/17D034/022") == "16/99/17D034/022"


def test_code_normalize_applies_confusion_map() -> None:
    """OCR letter->digit confusions in digit slots are corrected (O->0, etc.)."""
    # Every confusable maps onto its digit; the single letter block is preserved.
    assert codes.normalize_enregistrement("OO/O7/3OP126/317") == "00/07/30P126/317"


def test_code_normalize_rejects_malformed() -> None:
    """A read that cannot be coerced into the structure returns ``None``."""
    assert codes.normalize_enregistrement("not-a-code") is None
    assert codes.normalize_enregistrement(None) is None
