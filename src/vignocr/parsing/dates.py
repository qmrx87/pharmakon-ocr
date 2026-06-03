"""Deterministic date parsing for vignette FAB/EXP fields.

Vignettes carry month/year (and occasionally day/month/year) dates in mixed
separators: ``"03/2027"``, ``"03-2027"``, ``"2027-03"``, ``"15/03/2027"``.
Parsing is strict (``datetime.strptime`` against an explicit allow-list of
formats) — never fuzzy — so a malformed read fails closed (``None``) rather than
being silently coerced.

The format allow-list is supplied by the caller (the parser reads it from
``configs/parsing/fields.yaml`` per field: ``date_fab`` / ``date_exp``). When
omitted, the union of both fields' configured formats is used as a sensible
default. The EXP-after-FAB sanity rule (``must_be_after``) is exposed via
``is_after``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from functools import lru_cache

from vignocr.common import get_logger, load_config

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _default_formats() -> tuple[str, ...]:
    """Union of configured ``date_fab``/``date_exp`` formats, order-preserving."""
    fields = load_config("parsing/fields").get("fields", {})
    seen: list[str] = []
    for name in ("date_fab", "date_exp"):
        for fmt in fields.get(name, {}).get("formats", []):
            if fmt not in seen:
                seen.append(fmt)
    return tuple(seen)


def parse(text: str | None, formats: Sequence[str] | None = None) -> date | None:
    """Parse ``text`` into a ``date`` using the first matching format.

    ``formats`` is an ordered allow-list of ``strptime`` patterns (e.g.
    ``"%m/%Y"``). Patterns without a day component yield the first of the month.
    Returns ``None`` if no format matches.
    """
    if text is None:
        return None
    candidate = str(text).strip()
    if not candidate:
        return None

    fmts = tuple(formats) if formats is not None else _default_formats()
    for fmt in fmts:
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue

    log.debug("dates.parse: no format matched", raw=text, formats=list(fmts))
    return None


def is_after(exp: date | None, fab: date | None) -> bool:
    """Return whether expiry ``exp`` is strictly after manufacture ``fab``.

    Used for the ``date_exp.must_be_after: date_fab`` sanity check. If either
    date is missing the relation is undefined and ``False`` is returned (the
    caller treats a failed/undefined check as "do not trust the pair").
    """
    if exp is None or fab is None:
        return False
    return exp > fab
