"""Assemble the deterministic, parsing-stage view of an ``ExtractionRecord``.

This produces a **partial** record — fields + checksum + abstentions — with no
nomenclature step yet (the nomenclature layer fills that in later). It:

1. runs the ``prix + shp == ppa`` checksum (repair / verify / flag), then
2. applies the per-flow abstention thresholds from
   ``configs/parsing/fields.yaml: abstention``. The *selling* path is stricter
   than *receiving* (a wrong dispense is unacceptable): any field read below its
   threshold is marked ``status="abstain"`` ("à vérifier") and its name is
   appended to ``abstentions`` — never silently trusted.

Fields already decided by the checksum (``corrected`` / ``conflict``) are not
downgraded to ``abstain``; a missing read stays ``missing`` (distinct from an
explicit low-confidence abstention).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from vignocr.common import (
    ChecksumReport,
    ExtractionRecord,
    FieldRead,
    get_logger,
    load_config,
)
from vignocr.common.schemas import Flow
from vignocr.parsing import checksum as checksum_mod

log = get_logger(__name__)

# Statuses owned by the checksum stage — abstention must not override them.
_CHECKSUM_OWNED = frozenset({"corrected", "conflict"})


@lru_cache(maxsize=1)
def _abstention_cfg() -> dict[str, Any]:
    """Per-flow abstention thresholds (``default`` + optional per-field overrides)."""
    return dict(load_config("parsing/fields").get("abstention", {}))


def _threshold_for(field_name: str, flow: Flow) -> float:
    """Resolve the abstention threshold for a field under a given flow."""
    profile = _abstention_cfg().get(flow, {})
    if field_name in profile:
        return float(profile[field_name])
    return float(profile.get("default", 0.0))


def _apply_abstention(
    fields: dict[str, FieldRead], flow: Flow
) -> tuple[dict[str, FieldRead], list[str]]:
    """Mark low-confidence reads as ``abstain`` and collect their names."""
    out: dict[str, FieldRead] = {}
    abstentions: list[str] = []
    for name, fr in fields.items():
        # Keep checksum verdicts and genuinely-missing reads as-is.
        if fr.status in _CHECKSUM_OWNED or fr.value is None:
            out[name] = fr
            continue
        if fr.confidence < _threshold_for(name, flow):
            out[name] = fr.model_copy(update={"status": "abstain"})
            abstentions.append(name)
        else:
            out[name] = fr
    return out, abstentions


def build(
    fields: dict[str, FieldRead],
    flow: Flow = "selling",
    *,
    image_id: str = "",
) -> ExtractionRecord:
    """Build the partial ``ExtractionRecord`` for the parsing stage.

    Runs the money checksum over any ``prix``/``shp``/``ppa`` present, then applies
    the flow-specific abstention policy. Nomenclature/reimbursability are left at
    their defaults for a later stage to populate.
    """
    work = dict(fields)

    # 1) Checksum the money triple (only when at least one money field is present).
    report = ChecksumReport(verdict="incomplete")
    if any(k in work for k in ("prix", "shp", "ppa")):
        triple = {k: work.get(k, FieldRead(name=k)) for k in ("prix", "shp", "ppa")}
        repaired, report = checksum_mod.verify_and_repair(
            triple["prix"], triple["shp"], triple["ppa"]
        )
        # Write back only fields the caller actually supplied (or that got repaired).
        for k, fr in repaired.items():
            if k in fields or fr.status == "corrected":
                work[k] = fr

    # 2) Abstention policy (selling stricter than receiving).
    work, abstentions = _apply_abstention(work, flow)

    log.debug(
        "record.build",
        flow=flow,
        checksum=report.verdict,
        abstentions=abstentions,
        n_fields=len(work),
    )

    return ExtractionRecord(
        image_id=image_id,
        fields=work,
        checksum=report,
        abstentions=abstentions,
        flow=flow,
    )
