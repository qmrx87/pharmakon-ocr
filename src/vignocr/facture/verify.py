"""Deterministic arithmetic verification of a read invoice.

A supplier invoice carries its own redundancy: ``quantity x unit_price`` should
equal the printed ``line_total``, and the line totals should sum to the printed
net. Checking that arithmetic CATCHES OCR DIGIT ERRORS on money without a second
model — the facture analogue of the vignette ``prix + shp == ppa`` checksum.

Pure Python (Decimal, no ML / no network) so it runs anywhere and is unit-tested
on CPU. Money is parsed in Algerian/French style: space (or NBSP) thousands and
either ``,`` or ``.`` as the decimal separator.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

__all__ = ["to_decimal", "verify_line", "verify_facture"]

_NBSP = " "
_NARROW_NBSP = " "


def to_decimal(raw: Any) -> Decimal | None:
    """Parse an Algerian/French money or quantity string to ``Decimal``.

    Handles space/NBSP thousands and ``,`` or ``.`` decimals; strips currency
    letters (DA/DZD) and stray symbols. Returns ``None`` when not a number.

        "1 247.92" -> 1247.92   "1 497,50" -> 1497.50   "295 578,51" -> 295578.51
    """
    if raw is None:
        return None
    s = str(raw).replace(_NBSP, "").replace(_NARROW_NBSP, "").strip()
    s = re.sub(r"[^0-9.,\-]", "", s)  # drop spaces, DA, %, etc.
    if not s or s in {"-", ".", ","}:
        return None
    if "," in s and "." in s:
        # The LAST separator is the decimal; the other is a thousands grouping.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    # else: only "." (or none) — already a valid Decimal literal.
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def verify_line(line: dict[str, Any], tolerance: float = 0.02) -> dict[str, Any]:
    """Check ``quantity * unit_price_ht`` against the printed ``line_total``.

    Returns ``{checkable, ok, expected, printed, rel_error}``. ``checkable`` is
    False (``ok=None``) when any of the three numbers is missing/unparseable —
    a missing cell is not a math error, just unverifiable.
    """
    qty = to_decimal(line.get("quantity"))
    unit = to_decimal(line.get("unit_price_ht"))
    total = to_decimal(line.get("line_total"))
    res: dict[str, Any] = {
        "checkable": False,
        "ok": None,
        "expected": None,
        "printed": (str(total) if total is not None else None),
        "rel_error": None,
    }
    if qty is None or unit is None or total is None:
        return res
    expected = qty * unit
    res["checkable"] = True
    res["expected"] = str(expected)
    if total == 0:
        res["ok"] = expected == 0
        res["rel_error"] = 0.0 if expected == 0 else None
        return res
    rel = abs(expected - total) / abs(total)
    res["rel_error"] = float(rel)
    res["ok"] = rel <= tolerance
    return res


def verify_facture(
    data: dict[str, Any],
    *,
    line_tolerance: float = 0.02,
    total_tolerance: float = 0.02,
    low_conf: float = 0.75,
) -> dict[str, Any]:
    """Annotate every line with its arithmetic check and add a summary report.

    Returns ``{header, lines, totals, verification}``. Each line gains a
    ``_verify`` block. ``verification.needs_review`` is True when any line's math
    mismatches or any line is low-confidence — those MUST be human-checked before
    the stock is committed. The Σlines-vs-net check is best-effort and reported
    separately (remises / TVA / mixed price bases can legitimately break a naive
    sum), so it does not by itself force per-line review.
    """
    lines = list(data.get("lines", []) or [])
    out_lines: list[dict[str, Any]] = []
    sum_total = Decimal(0)
    n_checkable = n_mismatch = n_low_conf = 0

    for ln in lines:
        v = verify_line(ln, line_tolerance)
        t = to_decimal(ln.get("line_total"))
        if t is not None:
            sum_total += t
        if v["checkable"]:
            n_checkable += 1
            if v["ok"] is False:
                n_mismatch += 1
        try:
            if float(ln.get("confidence", 1.0)) < low_conf:
                n_low_conf += 1
        except (TypeError, ValueError):
            n_low_conf += 1
        out_lines.append({**ln, "_verify": v})

    totals = data.get("totals", {}) or {}
    # Compare line-total sum against the HT total (same base as Mont.HT) first.
    net = (
        to_decimal(totals.get("total_ht"))
        or to_decimal(totals.get("net_ht"))
        or to_decimal(totals.get("net_a_payer"))
    )
    totals_ok: bool | None = None
    totals_rel: float | None = None
    if net is not None and net != 0 and sum_total != 0:
        totals_rel = float(abs(sum_total - net) / abs(net))
        totals_ok = totals_rel <= total_tolerance

    report = {
        "n_lines": len(lines),
        "n_checkable_lines": n_checkable,
        "n_line_mismatches": n_mismatch,
        "n_low_confidence_lines": n_low_conf,
        "sum_line_total": str(sum_total),
        "printed_net": (str(net) if net is not None else None),
        "totals_ok": totals_ok,          # best-effort; informational
        "totals_rel_error": totals_rel,
        # Per-line math errors and low confidence force review; a totals-only
        # mismatch is surfaced but doesn't force it (remise/TVA can explain it).
        "needs_review": n_mismatch > 0 or n_low_conf > 0,
    }
    return {
        "header": {
            k: data.get(k)
            for k in ("supplier", "invoice_number", "invoice_date", "client")
        },
        "lines": out_lines,
        "totals": totals,
        "verification": report,
    }
