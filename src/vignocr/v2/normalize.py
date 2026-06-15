"""Type-aware value normalizers for v2 evaluation (exact-match fairness).

Two reads of the same vignette field can differ harmlessly ("05/27" vs
"05/2027" only when the format is ambiguous is NOT harmless — but "B-1234 " vs
"B-1234" and "132,21" vs "132.21" are). These normalizers canonicalize values
BEFORE exact-match comparison so the v1/v2a/v2b comparison measures reading
accuracy, not whitespace/punctuation luck.

Pure Python (no ML deps) — shared by the dataset builder, the Donut trainer's
validation metric, and the comparison harness.
"""

from __future__ import annotations

import re

__all__ = ["normalize_value", "FIELD_TYPES"]

# field -> normalizer type. Mirrors configs/classes.yaml `type` for the fields
# v2 reads; kept local so v2 eval has no config-load dependency at import time.
FIELD_TYPES: dict[str, str] = {
    "num_lot": "code",
    "num_enregistrement": "code",
    "date_fab": "date",
    "date_exp": "date",
    "ppa": "money",
    "prix": "money",
    "shp": "money",
    "tr": "money",
    "ppa_shp": "money_pair",
    "product_name": "text",
    "dci": "text",
    "dosage": "text",
    "forme": "text",
    "laboratoire": "text",
}

_DATE_SEP = re.compile(r"[/\-.\s]+")
_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_AMOUNT = re.compile(r"\d+[.,]\d{1,2}|\d+")


def _norm_date(v: str) -> str:
    """Canonicalize a date to its digit groups joined by '/' (e.g. 05/2027).

    Keeps the groups as written (no century guessing — '27' and '2027' stay
    different on purpose; that ambiguity is a real reading difference).
    """
    parts = [p for p in _DATE_SEP.split(v.strip()) if p and p.isdigit()]
    return "/".join(str(int(p)) if len(p) <= 2 else p for p in parts)


def _norm_code(v: str) -> str:
    """Uppercase, strip everything non-alphanumeric (codes compare by content)."""
    return _NON_ALNUM.sub("", v.upper())


def _norm_money(v: str) -> str:
    """First amount in the string, centime-quantized with a '.' separator."""
    m = _AMOUNT.search(v.replace(" ", ""))
    if not m:
        return v.strip()
    amt = m.group(0).replace(",", ".")
    if "." not in amt:
        amt += ".00"
    whole, _, cents = amt.partition(".")
    return f"{int(whole)}.{(cents + '00')[:2]}"


def _norm_money_pair(v: str) -> str:
    """All amounts in the string, each money-normalized, '+'-joined (ppa_shp)."""
    amounts = _AMOUNT.findall(v.replace(" ", ""))
    return "+".join(_norm_money(a) for a in amounts) if amounts else v.strip()


def _norm_text(v: str) -> str:
    return " ".join(v.split()).casefold()


def normalize_value(field: str, value: str) -> str:
    """Normalize ``value`` according to ``field``'s type (unknown -> text rules)."""
    kind = FIELD_TYPES.get(field, "text")
    v = str(value)
    if kind == "date":
        return _norm_date(v)
    if kind == "code":
        return _norm_code(v)
    if kind == "money":
        return _norm_money(v)
    if kind == "money_pair":
        return _norm_money_pair(v)
    return _norm_text(v)
