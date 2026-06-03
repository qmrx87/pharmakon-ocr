"""HTTP request/response envelopes for the serving layer.

These models are *thin wrappers* around the canonical pipeline types defined in
:mod:`vignocr.common.schemas`. The structured result of an extraction is the
shared :class:`~vignocr.common.schemas.ExtractionRecord` — it is **not**
redefined here; the ``POST /extract`` handler returns it directly. This module
only adds the API-surface types: the request body, and the liveness/readiness
payloads.

Re-exported here for one-stop importing by the app and by API consumers:
``ExtractionRecord``, ``FieldRead`` (and the supporting report models).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Reuse — never redefine — the canonical pipeline contract.
from vignocr.common.schemas import (
    BBox,
    ChecksumReport,
    ExtractionRecord,
    FieldRead,
    Flow,
    NomenclatureReport,
    Reimbursability,
)

__all__ = [
    "ExtractRequest",
    "HealthResponse",
    "ReadyResponse",
    "ErrorResponse",
    # re-exported canonical types (so consumers import them from one place)
    "ExtractionRecord",
    "FieldRead",
    "BBox",
    "Reimbursability",
    "ChecksumReport",
    "NomenclatureReport",
    "Flow",
]


class ExtractRequest(BaseModel):
    """Non-file parameters for ``POST /extract``.

    The vignette image itself arrives as a multipart ``UploadFile`` (handled by
    the route), so this model carries only the side-channel parameters. FastAPI
    binds these from multipart form fields.

    ``flow`` selects the abstention profile (``selling`` is stricter than
    ``receiving`` — see ``configs/parsing/fields.yaml: abstention``). When
    omitted, the server falls back to its configured default flow.

    ``idempotency_key`` lets the caller safely retry: it is echoed back in the
    response (and structured logs) so a consumer can de-dupe. The serving layer
    is stateless and does not itself persist keys — de-duplication is the
    integration layer's responsibility (see ``docs/INTEGRATION.md``).
    """

    flow: Flow | None = Field(
        default=None,
        description="Abstention profile: 'selling' (stricter) or 'receiving'. "
        "Defaults to the server's configured flow when omitted.",
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=255,
        description="Optional client-supplied key, echoed back for safe retries.",
    )


class HealthResponse(BaseModel):
    """Liveness probe payload — ``GET /health``. Always ``{"status": "ok"}``."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Readiness probe payload — ``GET /ready``.

    ``ready`` is ``True`` once the pipeline singleton has been constructed (or
    stub mode is active). ``models`` reports the resolved model versions /
    sources so an operator can confirm *which* weights are live; in stub mode
    the values indicate the stub (e.g. ``{"detector": "stub"}``).
    """

    ready: bool = False
    models: dict[str, str] = Field(default_factory=dict)
    flow_default: Flow = "selling"
    stub: bool = False


class ErrorResponse(BaseModel):
    """Uniform error envelope for 4xx/5xx responses."""

    detail: str
    code: str | None = None
