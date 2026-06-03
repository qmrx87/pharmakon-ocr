"""Deterministic money parsing — ``Decimal`` end-to-end, never ``float``.

Vignettes mix locales: ``"1 234,56 DA"``, ``"1234.56"``, ``"700.06"``,
``"702,56 DA"``. This module captures the numeric run, strips configured
thousands separators and currency tokens, normalizes the decimal separator to
the configured canonical form, and returns a centime-quantized ``Decimal``.

All rules (separators, currency tokens, the capture regex, value bounds, and the
money quantum) come from ``configs/parsing/fields.yaml`` — nothing is hardcoded.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any

from vignocr.common import get_logger, load_config

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _money_cfg() -> dict[str, Any]:
    """Resolve and cache the money-relevant slice of ``parsing/fields.yaml``."""
    cfg = load_config("parsing/fields")
    locale = cfg["locale"]
    money = cfg["money"]
    dec_seps = tuple(locale["decimal_separators"])
    # Token matcher built from the *configured* decimal separators. Unlike the
    # config's grouped-thousands gate regex (which fragments un-separated integers
    # like "1234"), this captures the full numeric core once grouping marks have
    # been stripped: integer part + optional decimal part of 1–2 digits.
    dec_class = "".join(re.escape(s) for s in dec_seps)
    token_re = re.compile(rf"\d+(?:[{dec_class}]\d{{1,2}})?")
    return {
        "decimal_separators": dec_seps,
        "canonical_decimal": str(locale["canonical_decimal"]),
        "thousands_separators": tuple(locale["thousands_separators"]),
        "currency_tokens": tuple(locale["currency_tokens"]),
        "quantum": Decimal(str(locale["money_quantum"])),
        "regex": re.compile(money["regex"]),
        "token_re": token_re,
        "min_value": Decimal(str(money["min_value"])),
        "max_value": Decimal(str(money["max_value"])),
    }


def parse(text: str | None) -> Decimal | None:
    """Parse free OCR text into a canonical, centime-quantized ``Decimal``.

    Handles thousands separators (``" "``, no-break space, ``"'"``), both decimal
    separators (``","`` / ``"."``), and trailing/leading currency tokens
    (``"DA"``, ``"DZD"``, ...). The numeric run is located via the configured
    ``money.regex`` so surrounding labels/noise are ignored.

    Returns ``None`` when no numeric value can be parsed, or when the value falls
    outside ``[min_value, max_value]`` (a sanity bound — above it we abstain).
    """
    if text is None:
        return None
    cfg = _money_cfg()
    raw = str(text)

    # Gate: the configured money regex must recognize money-like content.
    if cfg["regex"].search(raw) is None:
        return None

    # Strip configured currency tokens and thousands separators (grouping marks),
    # then locate the numeric core with the separator-driven token matcher. The
    # longest token wins (ignores stray single digits in surrounding labels).
    cleaned = raw
    for token in cfg["currency_tokens"]:
        cleaned = cleaned.replace(token, " ")
    for sep in cfg["thousands_separators"]:
        cleaned = cleaned.replace(sep, "")

    tokens = cfg["token_re"].findall(cleaned)
    if not tokens:
        return None
    run = max(tokens, key=len)

    # Normalize whichever decimal separator was used to the canonical one.
    canonical = cfg["canonical_decimal"]
    for sep in cfg["decimal_separators"]:
        if sep != canonical:
            run = run.replace(sep, canonical)

    if not run or run in {canonical, "-"}:
        return None

    try:
        value = Decimal(run)
    except InvalidOperation:
        log.debug("money.parse: undecimalable run", raw=text, run=run)
        return None

    if value < cfg["min_value"] or value > cfg["max_value"]:
        log.debug("money.parse: out of bounds -> abstain", raw=text, value=str(value))
        return None

    return value.quantize(cfg["quantum"])
