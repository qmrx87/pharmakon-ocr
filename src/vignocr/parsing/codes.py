"""Deterministic normalization of the N° d'Enregistrement (the anchor code).

Structure (spaces around the letter vary): ``AA / BB / CC <LETTER> DDD / EEE``
e.g. ``16/99/17D034/022``, ``18/97/14G 061/003``, ``09/22 F 018/235``.

The numeric slots (``a, b, c, d, e``) are corrected for common OCR digit/letter
confusions (``O->0``, ``I->1``, ``S->5`` ...) via the configured ``confusion_map``;
the single ``letter`` block is the most identity-bearing slot and is left
untouched by digit-coercion. The canonical form is emitted via the configured
``normalized_format``. All rules come from ``configs/parsing/fields.yaml``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from vignocr.common import get_logger, load_config

log = get_logger(__name__)

# Lenient capture: tolerate confusable letters inside the numeric slots and any
# spacing around the letter block. Slot widths match the strict grammar
# (2/2/2 <letter> 3/3); per-slot confusion correction + a strict re-match enforce
# validity afterwards.
_LENIENT = re.compile(
    r"^\s*"
    r"(?P<a>[\w]{2})\s*/\s*"
    r"(?P<b>[\w]{2})\s*/\s*"
    r"(?P<c>[\w]{2})\s*"
    r"(?P<letter>[A-Za-z])\s*"
    r"(?P<d>[\w]{3})\s*/\s*"
    r"(?P<e>[\w]{3})"
    r"\s*$"
)


@lru_cache(maxsize=1)
def _code_cfg() -> dict[str, Any]:
    """Resolve and cache the ``num_enregistrement`` rules."""
    field = load_config("parsing/fields")["fields"]["num_enregistrement"]
    return {
        "strict": re.compile(field["regex"]),
        "letters": frozenset(field.get("letters", [])),
        "normalized_format": field["normalized_format"],
        "confusion_map": dict(field.get("confusion_map", {})),
    }


def _coerce_digits(slot: str, confusion_map: dict[str, str]) -> str:
    """Map confusable characters in a numeric slot to their digit counterparts."""
    return "".join(confusion_map.get(ch, ch) for ch in slot)


def normalize_enregistrement(text: str | None) -> str | None:
    """Normalize a raw enregistrement read to ``{a}/{b}/{c}{letter}{d}/{e}``.

    Returns the canonical string, or ``None`` if the text cannot be coerced into
    the expected structure (e.g. wrong slot widths, or a digit slot still holds a
    non-digit after applying the confusion map).
    """
    if text is None:
        return None
    cfg = _code_cfg()

    match = _LENIENT.match(str(text))
    if match is None:
        log.debug("codes.normalize: structure mismatch", raw=text)
        return None

    confusion = cfg["confusion_map"]
    a = _coerce_digits(match.group("a"), confusion)
    b = _coerce_digits(match.group("b"), confusion)
    c = _coerce_digits(match.group("c"), confusion)
    d = _coerce_digits(match.group("d"), confusion)
    e = _coerce_digits(match.group("e"), confusion)
    letter = match.group("letter").upper()

    normalized = cfg["normalized_format"].format(a=a, b=b, c=c, letter=letter, d=d, e=e)

    # Re-validate against the strict grammar so the corrected slots are all digits.
    if cfg["strict"].match(normalized) is None:
        log.debug("codes.normalize: post-correction invalid", raw=text, normalized=normalized)
        return None

    return normalized
