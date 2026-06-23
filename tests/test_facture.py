"""CPU-only tests for FactureOCR — the arithmetic verifier and the (mocked) extractor.

Numbers are taken from the real supplier invoices the user provided, so the
arithmetic check is exercised against genuine data. The anthropic SDK is mocked.
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from vignocr.facture.claude_extract import FactureExtractor
from vignocr.facture.verify import to_decimal, verify_facture, verify_line

# --------------------------------------------------------------------------- #
# Money parsing (Algerian / French formats)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1 247.92", "1247.92"),
        ("1 497,50", "1497.50"),
        ("24 958.40", "24958.40"),
        ("295 578,51", "295578.51"),
        ("20", "20"),
        ("122 129,16 DA", "122129.16"),
    ],
)
def test_to_decimal_french_formats(raw: str, expected: str) -> None:
    assert str(to_decimal(raw)) == expected


@pytest.mark.parametrize("raw", ["", "  ", "DA", "-", None])
def test_to_decimal_non_numbers(raw: Any) -> None:
    assert to_decimal(raw) is None


# --------------------------------------------------------------------------- #
# Per-line arithmetic (qty x unit_price == line_total)
# --------------------------------------------------------------------------- #


def test_verify_line_ok_real_invoice_rows() -> None:
    # APROVASC: 20 x 1 247.92 = 24 958.40 (VECOPHARM invoice)
    v = verify_line({"quantity": "20", "unit_price_ht": "1 247.92", "line_total": "24 958.40"})
    assert v["checkable"] and v["ok"] is True
    # CLOFENAL: 30 x 111.31 = 3 339.30
    assert verify_line({"quantity": "30", "unit_price_ht": "111.31", "line_total": "3 339.30"})["ok"]


def test_verify_line_detects_digit_error() -> None:
    v = verify_line({"quantity": "20", "unit_price_ht": "1 247.92", "line_total": "9 999.00"})
    assert v["checkable"] and v["ok"] is False and v["rel_error"] > 0.02


def test_verify_line_uncheckable_when_cell_missing() -> None:
    v = verify_line({"quantity": "", "unit_price_ht": "1 247.92", "line_total": ""})
    assert v["checkable"] is False and v["ok"] is None


# --------------------------------------------------------------------------- #
# Whole-invoice verification + review flagging
# --------------------------------------------------------------------------- #


def _facture() -> dict[str, Any]:
    return {
        "supplier": "VECOPHARM",
        "invoice_number": "26/FA241955",
        "invoice_date": "09/06/2026",
        "client": "LALA HABIB",
        "lines": [
            {"designation": "APROVASC 150MG/5MG COMP.PELLI B/30", "quantity": "20",
             "unit_price_ht": "1 247.92", "line_total": "24 958.40", "lot": "GNN0194",
             "expiry": "02/29", "confidence": 0.96},
            {"designation": "BEKLOMIL 100UG/DOSE", "quantity": "3",
             "unit_price_ht": "600.00", "line_total": "1 800.00", "lot": "668001",
             "expiry": "04/28", "confidence": 0.93},
            # digit error AND low confidence -> must be flagged for review
            {"designation": "CLOFENAL 100MG SUPP B/10", "quantity": "30",
             "unit_price_ht": "111.31", "line_total": "5 000.00", "lot": "721",
             "expiry": "04/29", "confidence": 0.55},
        ],
        "totals": {"total_ht": "26 758.40", "net_ht": "26 758.40", "net_a_payer": "26 758.40"},
    }


def test_verify_facture_flags_mismatch_and_low_conf() -> None:
    out = verify_facture(_facture())
    rep = out["verification"]
    assert rep["n_lines"] == 3
    assert rep["n_checkable_lines"] == 3
    assert rep["n_line_mismatches"] == 1          # the CLOFENAL digit error
    assert rep["n_low_confidence_lines"] == 1
    assert rep["needs_review"] is True
    # the bad line carries its own verdict
    assert out["lines"][2]["_verify"]["ok"] is False
    assert out["lines"][0]["_verify"]["ok"] is True


def test_verify_facture_totals_ok_when_lines_sum_to_net() -> None:
    data = _facture()
    # Fix the bad line so the sum matches total_ht (24958.40 + 1800.00 + 3339.30).
    data["lines"][2]["line_total"] = "3 339.30"
    data["lines"][2]["confidence"] = 0.95
    data["totals"]["total_ht"] = "30 097.70"
    out = verify_facture(data)
    rep = out["verification"]
    assert rep["n_line_mismatches"] == 0
    assert rep["totals_ok"] is True
    assert rep["needs_review"] is False


# --------------------------------------------------------------------------- #
# Extractor (anthropic mocked)
# --------------------------------------------------------------------------- #


def _resp(text: str, stop_reason: str = "end_turn") -> Any:
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, stop_details=None,
    )


class _FakeClient:
    def __init__(self, resp: Any) -> None:
        self.messages = types.SimpleNamespace(create=lambda **kw: self._capture(kw))
        self._resp = resp
        self.calls: list[dict[str, Any]] = []

    def _capture(self, kw: dict[str, Any]) -> Any:
        self.calls.append(kw)
        return self._resp


def _img() -> Any:
    from PIL import Image

    return Image.new("RGB", (400, 600), (255, 255, 255))


def test_facture_extract_and_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    ex = FactureExtractor()
    payload = {
        "supplier": "adpp", "invoice_number": "FC 287", "invoice_date": "17/06/2026",
        "client": "PHARMACIE LALA HABIB",
        "lines": [
            {"designation": "LEVOTHYROX 25µg COMP B/30", "dci": "", "quantity": "10",
             "lot": "G032TL", "expiry": "30/09/2028", "unit_price_ht": "85,07",
             "ppa": "127,61", "shp": "", "line_total": "850,70", "confidence": 0.92},
        ],
        "totals": {"net_a_payer": "122 129,16", "total_ht": "850,70", "net_ht": "850,70"},
    }
    ex._client = _FakeClient(_resp(json.dumps(payload)))

    out = ex.extract_and_verify(_img())
    assert out["header"]["supplier"] == "adpp"
    assert out["lines"][0]["lot"] == "G032TL"
    # 10 x 85,07 = 850,70 -> line verifies
    assert out["lines"][0]["_verify"]["ok"] is True
    assert out["verification"]["needs_review"] is False

    kw = ex._client.calls[0]
    assert kw["model"].startswith("claude")
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert any(b.get("type") == "image" for b in kw["messages"][0]["content"])
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_facture_refusal_returns_empty() -> None:
    ex = FactureExtractor()
    ex._client = _FakeClient(_resp("", stop_reason="refusal"))
    out = ex.extract(_img())
    assert out["lines"] == [] and out["supplier"] == ""
