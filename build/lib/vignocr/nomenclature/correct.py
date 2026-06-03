"""Apply the nomenclature correction **policy** — the medical safety core.

Given the OCR field reads and a matched register row, this module repairs,
confirms, or flags each identity field according to
``configs/nomenclature/correction.yaml: policy``. The four buckets are obeyed
exactly and are **never** hardcoded here:

    never_overwrite            ppa, tr        -> never touched (vignette-specific).
    repair_always              product_name,  -> nomenclature is source of truth
                               laboratoire       (status="corrected").
    repair_if_ocr_low_or_agree dci            -> adopt nomenclature only if OCR
                                                 abstained/low-conf OR they already
                                                 agree (source="ocr+nomenclature").
    flag_on_conflict           dosage, forme  -> dispensing-critical: if OCR is
                                                 confident and DISAGREES, keep the
                                                 OCR value and FLAG it (never
                                                 overwrite); otherwise confirm/adopt.

The function returns a shallow-copied field dict (inputs are not mutated) and a
:class:`NomenclatureReport` summarizing the match and any conflicts.
"""

from __future__ import annotations

import re
from typing import Any

from vignocr.common import get_logger, load_config
from vignocr.common.schemas import FieldRead, NomenclatureConflict, NomenclatureReport
from vignocr.nomenclature.match import normalize_code, score

log = get_logger(__name__)

_ANCHOR_FIELD = "num_enregistrement"


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #


def _canon(value: str | None) -> str:
    """Case-insensitive, whitespace-collapsed form for value equality checks."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _agree(ocr_value: str | None, nom_value: str | None) -> bool:
    """True when both sides carry a value and they match after canonicalization."""
    a, b = _canon(ocr_value), _canon(nom_value)
    return bool(a) and bool(b) and a == b


def _is_low_conf(fr: FieldRead | None, threshold: float) -> bool:
    """OCR is 'low' when missing, abstained, or below the confidence threshold."""
    if fr is None:
        return True
    if fr.status in ("abstain", "missing"):
        return True
    if fr.value is None or _canon(fr.value) == "":
        return True
    return fr.confidence < threshold


def _nom_value(row: dict[str, str], field_name: str) -> str | None:
    """Non-empty nomenclature value for ``field_name``, else ``None``."""
    if not row:
        return None
    v = (row.get(field_name) or "").strip()
    return v or None


def _confidence_for(row: dict[str, str], anchor_norm: str, cfg: dict[str, Any]) -> float:
    """Recompute match confidence between the anchor and the matched row's key."""
    key_col = cfg.get("csv", {}).get("key_column", _ANCHOR_FIELD)
    key_norm = normalize_code(row.get(key_col, ""), cfg)
    conf, _raw = score(anchor_norm, key_norm, cfg)
    return conf


# --------------------------------------------------------------------------- #
# Per-bucket field updates (each returns the new FieldRead + optional conflict)
# --------------------------------------------------------------------------- #


def _make_field(
    name: str,
    value: str | None,
    *,
    confidence: float,
    status: str,
    source: str,
    template: FieldRead | None,
) -> FieldRead:
    """Build a FieldRead, preserving the OCR ``raw``/``bbox`` from the template."""
    return FieldRead(
        name=name,
        value=value,
        raw=template.raw if template else None,
        confidence=confidence,
        status=status,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        bbox=template.bbox if template else None,
    )


def _repair_always(
    name: str, current: FieldRead | None, nom_value: str | None, match_conf: float
) -> tuple[FieldRead | None, NomenclatureConflict | None]:
    """Nomenclature is source of truth: overwrite with the register value."""
    if nom_value is None:
        return current, None  # nothing authoritative to write
    field = _make_field(
        name,
        nom_value,
        confidence=match_conf,
        status="corrected",
        source="nomenclature",
        template=current,
    )
    conflict = None
    if current is not None and current.value is not None and not _agree(current.value, nom_value):
        conflict = NomenclatureConflict(
            field=name,
            ocr=current.value,
            nomenclature=nom_value,
            action="kept_nomenclature",
        )
    return field, conflict


def _repair_if_low_or_agree(
    name: str,
    current: FieldRead | None,
    nom_value: str | None,
    match_conf: float,
    low_threshold: float,
) -> tuple[FieldRead | None, NomenclatureConflict | None]:
    """Adopt nomenclature only when OCR is low/absent OR the two already agree."""
    if nom_value is None:
        return current, None
    agree = current is not None and _agree(current.value, nom_value)
    if agree:
        # Highest trust: OCR and register concur.
        field = _make_field(
            name,
            current.value,
            confidence=max(current.confidence, match_conf),
            status="ok",
            source="ocr+nomenclature",
            template=current,
        )
        return field, None
    if _is_low_conf(current, low_threshold):
        # OCR abstained / low — fill from the register.
        field = _make_field(
            name,
            nom_value,
            confidence=match_conf,
            status="corrected",
            source="nomenclature",
            template=current,
        )
        return field, None
    # OCR is confident and disagrees -> for this bucket we keep OCR untouched
    # (it is not dispensing-critical; we don't silently overwrite a confident read).
    return current, None


def _flag_on_conflict(
    name: str,
    current: FieldRead | None,
    nom_value: str | None,
    match_conf: float,
    low_threshold: float,
) -> tuple[FieldRead | None, NomenclatureConflict | None]:
    """Dispensing-critical: confident disagreement -> KEEP OCR + flag; never overwrite."""
    if nom_value is None:
        return current, None
    agree = current is not None and _agree(current.value, nom_value)
    if agree:
        field = _make_field(
            name,
            current.value,
            confidence=max(current.confidence, match_conf),
            status="ok",
            source="ocr+nomenclature",
            template=current,
        )
        return field, None
    if _is_low_conf(current, low_threshold):
        # OCR abstained / low — safe to adopt the register value.
        field = _make_field(
            name,
            nom_value,
            confidence=match_conf,
            status="corrected",
            source="nomenclature",
            template=current,
        )
        return field, None
    # OCR is CONFIDENT and DISAGREES on a dispensing-critical field:
    # keep the OCR value, mark it for human review, and report the conflict.
    flagged = _make_field(
        name,
        current.value,  # KEEP OCR — never overwrite
        confidence=current.confidence,
        status="conflict",
        source="ocr",
        template=current,
    )
    conflict = NomenclatureConflict(
        field=name,
        ocr=current.value,
        nomenclature=nom_value,
        action="flagged",
    )
    return flagged, conflict


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def apply(
    fields: dict[str, FieldRead],
    row: dict[str, str] | None,
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, FieldRead], NomenclatureReport]:
    """Apply the correction policy to ``fields`` using the matched ``row``.

    Args:
        fields: OCR reads keyed by field name (``classes.yaml`` names).
        row: the matched nomenclature row (``{column: value}``), or ``None`` when
            no record matched — in which case fields are returned unchanged.
        cfg: ``correction.yaml`` dict; loaded if omitted.

    Returns:
        ``(updated_fields, report)``. ``updated_fields`` is a new dict (inputs are
        never mutated); ``report`` carries match status, the normalized anchor
        code, the match confidence, and any conflicts.
    """
    cfg = cfg if cfg is not None else load_config("nomenclature/correction")
    policy = cfg.get("policy", {})
    never = set(policy.get("never_overwrite", []))
    repair_always = list(policy.get("repair_always", []))
    repair_low_or_agree = list(policy.get("repair_if_ocr_low_or_agree", []))
    flag_conflict = list(policy.get("flag_on_conflict", []))
    low_threshold = float(policy.get("ocr_low_conf_threshold", 0.75))
    emit_report = bool(policy.get("emit_conflict_report", True))

    out: dict[str, FieldRead] = dict(fields)  # shallow copy; FieldReads are replaced, not mutated

    anchor = fields.get(_ANCHOR_FIELD)
    anchor_norm = normalize_code(anchor.value if anchor else None, cfg) or None

    if not row:
        log.info("nomenclature.apply_unmatched", anchor=anchor_norm)
        return out, NomenclatureReport(
            matched=False,
            num_enregistrement_normalized=anchor_norm,
            match_confidence=0.0,
            conflicts=[],
        )

    match_conf = _confidence_for(row, anchor_norm or "", cfg)
    conflicts: list[NomenclatureConflict] = []

    def _record(name: str, result: tuple[FieldRead | None, NomenclatureConflict | None]) -> None:
        field, conflict = result
        if field is not None:
            out[name] = field
        if conflict is not None and emit_report:
            conflicts.append(conflict)

    # never_overwrite: explicitly left untouched (ppa, tr). No-op by design,
    # but we assert it so the intent is unmistakable and future-proof.
    for name in never:
        if name in fields:
            out[name] = fields[name]  # identical reference; never replaced

    for name in repair_always:
        if name in never:
            continue
        _record(name, _repair_always(name, fields.get(name), _nom_value(row, name), match_conf))

    for name in repair_low_or_agree:
        if name in never:
            continue
        _record(
            name,
            _repair_if_low_or_agree(
                name, fields.get(name), _nom_value(row, name), match_conf, low_threshold
            ),
        )

    for name in flag_conflict:
        if name in never:
            continue
        _record(
            name,
            _flag_on_conflict(
                name, fields.get(name), _nom_value(row, name), match_conf, low_threshold
            ),
        )

    log.info(
        "nomenclature.apply_matched",
        anchor=anchor_norm,
        confidence=round(match_conf, 4),
        conflicts=len(conflicts),
    )
    return out, NomenclatureReport(
        matched=True,
        num_enregistrement_normalized=anchor_norm,
        match_confidence=match_conf,
        conflicts=conflicts,
    )
