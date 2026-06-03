"""Split the combined ``ppa_shp`` detection box into separate ``prix`` and ``shp``.

The real Algeria-Drug-label annotation uses ONE box for the prix+shp line
(e.g. ``"248.50 + 1.50"`` or ``"Prix 248,50 SHP 1,50"``). After OCR reads the
crop, this module extracts the two money values and emits ``FieldRead`` objects for
``prix`` and ``shp`` so the downstream ``prix + shp == ppa`` checksum runs
unchanged. Money is :class:`decimal.Decimal`; never ``float``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from vignocr.common import FieldRead, get_logger, load_config, money_str
from vignocr.parsing import money

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    """Read ``parsing/fields.yaml: fields.ppa_shp`` once."""
    fields_cfg = load_config("parsing/fields")["fields"]
    return dict(fields_cfg.get("ppa_shp") or {})


def split(combined: FieldRead) -> tuple[FieldRead, FieldRead]:
    """Split a ``ppa_shp`` :class:`FieldRead` into ``(prix, shp)``.

    The detection box already isolates the prix+shp line, so the OCR text in
    ``combined.value`` / ``combined.raw`` carries the two amounts side-by-side
    (typically ``"<prix> + <shp>"`` or ``"<prix> SHP <shp>"``).

    On a clean parse both returned :class:`FieldRead`\s carry ``status="ok"``,
    ``source="ocr"`` and the parent's ``confidence``/``bbox``. On ambiguity each
    becomes ``status="abstain"`` with ``value=None`` so the checksum reports
    ``incomplete`` rather than silently accepting a guess.
    """
    cfg = _cfg()
    text = (combined.value if combined.value is not None else combined.raw) or ""
    pattern = cfg.get("split_regex")

    prix_amt = shp_amt = None
    if pattern:
        m = re.search(pattern, text)
        if m:
            prix_amt = money.parse(m.group("a"))
            shp_amt = money.parse(m.group("b"))

    # Fallback: no usable separator -> two distinct numbers, smaller = shp.
    if prix_amt is None or shp_amt is None:
        nums = [d for d in (money.parse(t) for t in re.findall(r"[\d  .,]+", text)) if d is not None]
        if len(nums) >= 2:
            ordered = sorted(nums, reverse=True)  # prix is the larger value
            prix_amt, shp_amt = ordered[0], ordered[1]
            log.info("ppa_shp.split_fallback_size_order", prix=str(prix_amt), shp=str(shp_amt))

    def _build(name: str, amt: Any) -> FieldRead:
        ok = amt is not None
        return FieldRead(
            name=name,
            value=money_str(amt) if ok else None,
            raw=combined.raw,
            confidence=combined.confidence if ok else 0.0,
            status="ok" if ok else "abstain",
            source="ocr" if ok else "none",
            bbox=combined.bbox,
        )

    return _build("prix", prix_amt), _build("shp", shp_amt)
