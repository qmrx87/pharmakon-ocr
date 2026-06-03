"""Deterministic PPA disambiguation.

The PPA may appear twice on a vignette: an *intermediate* additive form
(``PPA: 700,06+2,50``) and the *final* total (``PPA = 702,56 DA``). The final
value is the dispensable price and must be selected deterministically.

Policy (from ``configs/parsing/fields.yaml: ppa_disambiguation``):
    prefer: final                     -> the ``= XXX,XX DA`` value wins
    intermediate_pattern: "a + b"     -> if only the additive form is present,
                                         resolve it to ``a + b`` (Decimal)
    fallback: sum_prix_shp            -> when no PPA line resolves, the caller
                                         (checksum) anchors PPA from prix + shp

This selector operates only over PPA candidates; the cross-field
``sum_prix_shp`` anchoring is performed in :mod:`vignocr.parsing.checksum`.
"""

from __future__ import annotations

import re
from decimal import Decimal
from functools import lru_cache
from typing import Any

from vignocr.common import FieldRead, get_logger, load_config, money_str
from vignocr.parsing import money

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _ppa_cfg() -> dict[str, Any]:
    cfg = load_config("parsing/fields")["ppa_disambiguation"]
    return {
        "prefer": str(cfg.get("prefer", "final")),
        "final": re.compile(cfg["final_pattern"]),
        "intermediate": re.compile(cfg["intermediate_pattern"]),
        "fallback": str(cfg.get("fallback", "sum_prix_shp")),
    }


def _text_of(c: FieldRead) -> str:
    """Prefer raw OCR text (carries the ``=``/``+`` markers) then the value."""
    return c.raw if c.raw is not None else (c.value or "")


def _as_final(c: FieldRead) -> Decimal | None:
    """Extract the final ``= XXX,XX DA`` amount from a candidate, if present."""
    m = _ppa_cfg()["final"].search(_text_of(c))
    return money.parse(m.group("num")) if m else None


def _as_intermediate(c: FieldRead) -> Decimal | None:
    """Resolve the intermediate ``a + b`` additive form to a sum, if present."""
    m = _ppa_cfg()["intermediate"].search(_text_of(c))
    if not m:
        return None
    a, b = money.parse(m.group("a")), money.parse(m.group("b"))
    if a is None or b is None:
        return None
    return (a + b).quantize(Decimal("0.01"))


def disambiguate(candidates: list[FieldRead]) -> FieldRead:
    """Select the single final PPA from competing reads.

    Resolution order (config ``prefer: final``):
      1. a candidate matching the final ``= XXX,XX DA`` pattern (highest conf wins);
      2. else the intermediate ``a + b`` form, resolved to its sum;
      3. else the highest-confidence candidate parsed as a plain money value;
      4. else a ``missing`` PPA placeholder (checksum may later anchor it).
    """
    if not candidates:
        return FieldRead(name="ppa", status="missing", source="none")

    ranked = sorted(candidates, key=lambda c: c.confidence, reverse=True)

    # 1) Final "= XXX,XX DA" — the dispensable total.
    for cand in ranked:
        final = _as_final(cand)
        if final is not None:
            return cand.model_copy(
                update={"value": money_str(final), "status": "ok", "source": cand.source or "ocr"}
            )

    # 2) Intermediate "a + b" — resolve additively.
    for cand in ranked:
        summed = _as_intermediate(cand)
        if summed is not None:
            log.debug("ppa.disambiguate: resolved intermediate form", value=money_str(summed))
            return cand.model_copy(
                update={"value": money_str(summed), "status": "ok", "source": cand.source or "ocr"}
            )

    # 3) Plain money value on the best candidate.
    best = ranked[0]
    plain = money.parse(_text_of(best))
    if plain is not None:
        status = best.status if best.status in {"ok", "abstain", "corrected", "conflict"} else "ok"
        return best.model_copy(update={"value": money_str(plain), "status": status})

    # 4) Nothing parseable — leave for checksum's sum_prix_shp fallback.
    log.debug("ppa.disambiguate: no parseable PPA candidate", fallback=_ppa_cfg()["fallback"])
    return best.model_copy(update={"value": None, "status": "missing"})
