"""Canonical data types shared across modules (the wire + in-memory contract).

These Pydantic v2 models are the single definition of the pipeline's output. The
``serving`` layer emits them directly; ``parsing``/``nomenclature``/``pipeline`` all
build and pass them. Money is carried as a **Decimal-string** (e.g. ``"250.00"``),
never a float — see ``MoneyStr``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FieldStatus = Literal["ok", "abstain", "corrected", "conflict", "missing"]
FieldSource = Literal["ocr", "nomenclature", "ocr+nomenclature", "checksum", "none"]
BandColor = Literal["green", "red", "orange", "unknown"]
ChecksumVerdict = Literal["ok", "repaired", "mismatch", "incomplete"]
Flow = Literal["selling", "receiving"]


def money_str(value: Decimal | str | None) -> str | None:
    """Normalize a money value to a centime-quantized Decimal string, or None."""
    if value is None:
        return None
    d = value if isinstance(value, Decimal) else Decimal(str(value))
    return str(d.quantize(Decimal("0.01")))


class BBox(BaseModel):
    """COCO-style box in pixels of the input image: [x, y, w, h]."""

    x: float
    y: float
    w: float
    h: float


class FieldRead(BaseModel):
    name: str
    value: str | None = None
    raw: str | None = None
    confidence: float = 0.0
    status: FieldStatus = "missing"
    source: FieldSource = "none"
    bbox: BBox | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class Reimbursability(BaseModel):
    color: BandColor = "unknown"
    eligible: bool | None = None
    confidence: float = 0.0
    label: str = "À vérifier"


class ChecksumReport(BaseModel):
    verdict: ChecksumVerdict = "incomplete"
    prix: str | None = None
    shp: str | None = None
    ppa: str | None = None
    repaired_field: str | None = None


class NomenclatureConflict(BaseModel):
    field: str
    ocr: str | None
    nomenclature: str | None
    action: Literal["flagged", "kept_ocr", "kept_nomenclature"]


class NomenclatureReport(BaseModel):
    matched: bool = False
    num_enregistrement_normalized: str | None = None
    match_confidence: float = 0.0
    conflicts: list[NomenclatureConflict] = Field(default_factory=list)


class ExtractionRecord(BaseModel):
    """The structured result of running one vignette through the pipeline."""

    # `model_versions` legitimately starts with "model_"; opt out of pydantic's
    # protected-namespace check so it doesn't warn on every construction.
    model_config = ConfigDict(protected_namespaces=())

    image_id: str
    fields: dict[str, FieldRead] = Field(default_factory=dict)
    reimbursability: Reimbursability = Field(default_factory=Reimbursability)
    checksum: ChecksumReport = Field(default_factory=ChecksumReport)
    nomenclature: NomenclatureReport = Field(default_factory=NomenclatureReport)
    abstentions: list[str] = Field(default_factory=list)
    flow: Flow = "selling"
    model_versions: dict[str, str] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
