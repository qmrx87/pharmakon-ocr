"""Serving configuration + lazily-constructed pipeline singleton.

Two concerns live here:

1. :class:`Settings` — a ``pydantic-settings`` model. Every operational knob
   (model paths, default flow, upload limit, log format) is read from the
   environment with the ``VIGNOCR_`` prefix, so the container is 12-factor and
   needs zero code changes between dev / staging / prod.

2. :func:`get_pipeline` — a cached getter that builds the
   :class:`vignocr.pipeline.orchestrator.VignocrPipeline` **once** per process.
   The orchestrator (and its torch/onnxruntime backends) are imported *inside*
   the function, never at module top level, so importing this module — and
   booting the FastAPI app — never drags in heavy ML libs. If the pipeline
   module is absent or its ML deps are missing, the getter falls back to a
   deterministic :class:`_StubPipeline` so ``/extract`` still answers on a
   CPU-only box today.

Nothing here is request-scoped: the singleton is shared across requests and the
workers stay stateless.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from vignocr.common import get_logger
from vignocr.common.schemas import (
    ChecksumReport,
    ExtractionRecord,
    Flow,
    NomenclatureReport,
    Reimbursability,
)

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Settings (12-factor; env prefix VIGNOCR_)
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Process configuration, overridable via ``VIGNOCR_*`` environment vars.

    Examples::

        VIGNOCR_DEFAULT_FLOW=receiving
        VIGNOCR_MAX_UPLOAD_MB=20
        VIGNOCR_DETECTOR_PATH=/models/detector.onnx
        VIGNOCR_ALLOW_STUB=0     # fail readiness instead of stubbing in prod
    """

    model_config = SettingsConfigDict(
        env_prefix="VIGNOCR_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API behaviour ---
    default_flow: Flow = Field(
        default="selling",
        description="Abstention profile used when a request omits 'flow'.",
    )
    max_upload_mb: float = Field(
        default=10.0,
        gt=0,
        description="Max accepted upload size in megabytes (413 above this).",
    )
    allowed_upload_prefix: str = Field(
        default="image/",
        description="Required Content-Type prefix for uploads (415 otherwise).",
    )

    # --- model artifacts (paths/URIs resolved by the pipeline, not here) ---
    detector_path: str | None = Field(
        default=None, description="Path/URI to the detector weights (ONNX or ckpt)."
    )
    recognizer_path: str | None = Field(
        default=None, description="Path/URI to the OCR recognizer weights."
    )
    nomenclature_csv: str | None = Field(
        default=None,
        description="Override for the nomenclature CSV path (else from config).",
    )

    # --- ops ---
    allow_stub: bool = Field(
        default=True,
        description="If the real pipeline can't load, serve a deterministic stub "
        "(True) or report not-ready / 503 (False). Set False in prod.",
    )
    title: str = Field(default="VignOCR", description="OpenAPI/title shown in docs.")

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached process settings (read from the environment once)."""
    return Settings()


# --------------------------------------------------------------------------- #
# Pipeline protocol + lazy singleton
# --------------------------------------------------------------------------- #


@runtime_checkable
class PipelineLike(Protocol):
    """Structural type the app depends on (keeps serving decoupled from pipeline)."""

    def extract(self, image: Any, *, flow: Flow = "selling") -> ExtractionRecord: ...

    def model_versions(self) -> dict[str, str]: ...


class _StubPipeline:
    """Deterministic stand-in used when the real pipeline can't be constructed.

    Returns a well-formed :class:`ExtractionRecord` (empty fields, everything
    marked ``incomplete``/``unknown``) so the HTTP contract is exercisable on a
    CPU-only box without ML deps or trained weights. It is *honest*: callers can
    tell it is a stub from ``model_versions`` (``{"detector": "stub", ...}``)
    and from ``/ready`` reporting ``stub: true``.
    """

    _VERSIONS = {
        "detector": "stub",
        "recognizer": "stub",
        "nomenclature_version": "stub",
    }

    def extract(self, image: Any, *, flow: Flow = "selling") -> ExtractionRecord:  # noqa: ARG002
        return ExtractionRecord(
            image_id="stub",
            fields={},
            reimbursability=Reimbursability(),
            checksum=ChecksumReport(verdict="incomplete"),
            nomenclature=NomenclatureReport(),
            abstentions=[],
            flow=flow,
            model_versions=dict(self._VERSIONS),
            timings_ms={},
        )

    def model_versions(self) -> dict[str, str]:
        return dict(self._VERSIONS)


def _build_real_pipeline(settings: Settings) -> PipelineLike:
    """Import and construct the orchestrator lazily.

    Raises whatever the pipeline raises (``ImportError`` if the module/ML deps
    are missing); the caller decides whether to fall back to the stub.
    """
    # Lazy import: the pipeline module may not exist yet, and when it does it may
    # pull in torch/onnxruntime. Keeping this here means app import stays light.
    from vignocr.pipeline.orchestrator import VignocrPipeline  # noqa: PLC0415

    cfg: dict[str, Any] = {
        "detector_path": settings.detector_path,
        "recognizer_path": settings.recognizer_path,
        "nomenclature_csv": settings.nomenclature_csv,
        "default_flow": settings.default_flow,
    }
    pipeline = VignocrPipeline(cfg)
    if not isinstance(pipeline, PipelineLike):  # defensive: contract drift guard
        log.warning("pipeline_missing_protocol_methods", type=type(pipeline).__name__)
    return pipeline  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_pipeline() -> PipelineLike:
    """Return the process-wide pipeline singleton, building it on first call.

    Tries the real orchestrator first; on ``ImportError`` (module/ML deps absent)
    falls back to :class:`_StubPipeline` when ``allow_stub`` is set, otherwise
    re-raises so ``/ready`` can report not-ready.
    """
    settings = get_settings()
    t0 = time.perf_counter()
    try:
        pipeline = _build_real_pipeline(settings)
        log.info(
            "pipeline_loaded",
            stub=False,
            load_ms=round((time.perf_counter() - t0) * 1000, 1),
            versions=pipeline.model_versions(),
        )
        return pipeline
    except ImportError as exc:
        if not settings.allow_stub:
            log.error("pipeline_unavailable", error=str(exc), allow_stub=False)
            raise
        log.warning("pipeline_stub_fallback", reason=str(exc))
        return _StubPipeline()


def is_stub(pipeline: PipelineLike) -> bool:
    """True if ``pipeline`` is the deterministic stub (not a real model)."""
    return isinstance(pipeline, _StubPipeline)


def reset_pipeline_cache() -> None:
    """Clear the cached singleton (used by tests / hot config reloads)."""
    get_pipeline.cache_clear()
    get_settings.cache_clear()
