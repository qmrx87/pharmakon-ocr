"""The ``prix + shp == ppa`` checksum — one of the strongest accuracy levers.

Behaviour (from ``configs/parsing/fields.yaml: checksum``), Decimal-only:

* **repair** — if exactly one of the three is missing/low-confidence and the
  other two are confident anchors (``>= min_conf_to_anchor``), recompute the
  third arithmetically. The repaired field becomes ``status="corrected"``,
  ``source="checksum"``; verdict ``"repaired"``.
* **mismatch** — if all three are present but ``prix + shp != ppa`` (to the
  centime), flag every involved field ``status="conflict"`` and verdict
  ``"mismatch"``. A failed checksum is never silently accepted.
* **ok** — all three present and the identity holds.
* **incomplete** — too little confident information to verify or repair.

Money is parsed/validated as ``Decimal``; never ``float``.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Any

from vignocr.common import ChecksumReport, FieldRead, get_logger, load_config, money_str
from vignocr.parsing import money

log = get_logger(__name__)

_FIELDS = ("prix", "shp", "ppa")


@lru_cache(maxsize=1)
def _checksum_cfg() -> dict[str, Any]:
    cfg = load_config("parsing/fields")["checksum"]
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "repair": bool(cfg.get("repair", True)),
        "tolerance": Decimal(str(cfg.get("tolerance", "0.00"))),
        "min_conf_to_anchor": float(cfg.get("min_conf_to_anchor", 0.80)),
    }


def _amount(fr: FieldRead) -> Decimal | None:
    """Decimal amount of a field, preferring its normalized value then raw text."""
    return money.parse(fr.value if fr.value is not None else fr.raw)


def _is_anchor(fr: FieldRead, amount: Decimal | None, min_conf: float) -> bool:
    """A field anchors a repair when it has a value and sufficient confidence."""
    return amount is not None and fr.confidence >= min_conf


def verify_and_repair(
    prix: FieldRead, shp: FieldRead, ppa: FieldRead
) -> tuple[dict[str, FieldRead], ChecksumReport]:
    """Apply ``prix + shp == ppa`` to verify, repair, or flag the money triple.

    Returns the (possibly updated) fields keyed by name and a :class:`ChecksumReport`.
    Inputs are never mutated; updated copies are returned.
    """
    cfg = _checksum_cfg()
    fields: dict[str, FieldRead] = {"prix": prix, "shp": shp, "ppa": ppa}

    if not cfg["enabled"]:
        return fields, ChecksumReport(verdict="incomplete")

    amounts = {name: _amount(fields[name]) for name in _FIELDS}
    min_conf = cfg["min_conf_to_anchor"]
    anchors = {name: _is_anchor(fields[name], amounts[name], min_conf) for name in _FIELDS}

    report = ChecksumReport(
        verdict="incomplete",
        prix=money_str(amounts["prix"]),
        shp=money_str(amounts["shp"]),
        ppa=money_str(amounts["ppa"]),
    )

    # ---- Repair: exactly one field unusable, the other two confident anchors ----
    non_anchored = [name for name in _FIELDS if not anchors[name]]
    if cfg["repair"] and len(non_anchored) == 1:
        target = non_anchored[0]
        p, s, a = amounts["prix"], amounts["shp"], amounts["ppa"]
        if target == "ppa":
            recomputed = (p + s).quantize(Decimal("0.01"))
        elif target == "prix":
            recomputed = (a - s).quantize(Decimal("0.01"))
        else:  # shp
            recomputed = (a - p).quantize(Decimal("0.01"))

        fields[target] = fields[target].model_copy(
            update={
                "value": money_str(recomputed),
                "status": "corrected",
                "source": "checksum",
            }
        )
        amounts[target] = recomputed
        report = report.model_copy(
            update={
                "verdict": "repaired",
                "repaired_field": target,
                "prix": money_str(amounts["prix"]),
                "shp": money_str(amounts["shp"]),
                "ppa": money_str(amounts["ppa"]),
            }
        )
        log.info("checksum.repair", repaired=target, value=money_str(recomputed))
        return fields, report

    # ---- Verify / flag: need all three present to assert the identity ----
    if all(amounts[name] is not None for name in _FIELDS):
        residual = abs((amounts["prix"] + amounts["shp"]) - amounts["ppa"])
        if residual <= cfg["tolerance"]:
            return fields, report.model_copy(update={"verdict": "ok"})

        # Arithmetic mismatch — flag every involved field; never silently accept.
        for name in _FIELDS:
            fields[name] = fields[name].model_copy(update={"status": "conflict"})
        log.warning(
            "checksum.mismatch",
            prix=money_str(amounts["prix"]),
            shp=money_str(amounts["shp"]),
            ppa=money_str(amounts["ppa"]),
            residual=str(residual),
        )
        return fields, report.model_copy(update={"verdict": "mismatch"})

    # ---- Otherwise: not enough confident information ----
    return fields, report
