"""FastAPI application — the stateless HTTP front door to the VignOCR pipeline.

Endpoints
---------
* ``GET  /health`` — liveness. Always ``{"status": "ok"}`` (no model load).
* ``GET  /ready``  — readiness. Builds the pipeline singleton lazily and reports
  ``{ready, models, flow_default, stub}``.
* ``POST /extract`` — multipart image + ``flow`` → :class:`ExtractionRecord`. The
  pipeline (and its ML backends) is imported **lazily** inside the handler, so
  the app boots on CPU even before ``vignocr.pipeline.orchestrator`` exists.

Design
------
* **Stateless / 12-factor.** Behaviour comes entirely from :class:`Settings`
  (env ``VIGNOCR_*``). No request state is retained between calls.
* **Prefill-and-confirm.** ``/extract`` returns the full record — abstentions,
  per-field confidence/status, and the checksum verdict — so the consumer can
  pre-fill a form and ask a human to confirm the low-confidence parts.
* **Defensive uploads.** Content-Type must match ``VIGNOCR_ALLOWED_UPLOAD_PREFIX``
  (default ``image/``) → 415 otherwise; body over ``VIGNOCR_MAX_UPLOAD_MB`` → 413.
* **Structured logging** via :mod:`vignocr.common` with the idempotency key bound
  to every line for a request.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import JSONResponse

from vignocr.common import configure_logging, get_logger
from vignocr.common.schemas import ExtractionRecord, Flow
from vignocr.serving.deps import (
    PipelineLike,
    Settings,
    get_pipeline,
    get_settings,
    is_stub,
)
from vignocr.serving.schemas import (
    ErrorResponse,
    ExtractRequest,
    HealthResponse,
    ReadyResponse,
)

log = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Importing/constructing it loads **no** ML libs."""
    configure_logging()
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.title,
        version="0.1.0",
        summary="OCR extraction for Algerian pharmaceutical vignettes.",
        description=(
            "Stateless extraction service. POST a vignette image to /extract to "
            "receive a structured ExtractionRecord (fields, reimbursability, "
            "checksum, nomenclature, abstentions) for prefill-and-confirm."
        ),
    )

    # ----------------------------------------------------------------- health
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["ops"],
        summary="Liveness probe",
    )
    def health() -> HealthResponse:
        """Cheap liveness check — does not touch the pipeline or any model."""
        return HealthResponse(status="ok")

    # ------------------------------------------------------------------ ready
    @app.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["ops"],
        responses={503: {"model": ReadyResponse}},
        summary="Readiness probe",
    )
    def ready() -> Response:
        """Readiness check — builds the pipeline singleton lazily on first call.

        Returns 200 with model versions when ready (real or stub). If the real
        pipeline cannot load and stubbing is disabled, returns 503.
        """
        try:
            pipeline = get_pipeline()
        except ImportError as exc:
            body = ReadyResponse(
                ready=False,
                models={},
                flow_default=settings.default_flow,
                stub=False,
            )
            log.warning("ready_not_ready", error=str(exc))
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=body.model_dump(),
            )
        body = ReadyResponse(
            ready=True,
            models=pipeline.model_versions(),
            flow_default=settings.default_flow,
            stub=is_stub(pipeline),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump())

    # ---------------------------------------------------------------- extract
    @app.post(
        "/extract",
        response_model=ExtractionRecord,
        tags=["extraction"],
        responses={
            413: {"model": ErrorResponse, "description": "Upload too large"},
            415: {"model": ErrorResponse, "description": "Unsupported media type"},
            422: {"model": ErrorResponse, "description": "Unreadable / invalid image"},
            503: {"model": ErrorResponse, "description": "Pipeline unavailable"},
        },
        summary="Extract structured fields from a vignette image",
    )
    async def extract(
        # FastAPI's dependency markers are *required* in arg defaults — B008 here
        # is the framework's idiom, not a mistake.
        file: UploadFile = File(..., description="Vignette image (image/*)."),  # noqa: B008
        flow: Flow | None = Form(  # noqa: B008
            default=None,
            description="Abstention profile: 'selling' (stricter) or 'receiving'.",
        ),
        idempotency_key: str | None = Form(  # noqa: B008
            default=None, description="Optional client key, echoed in logs."
        ),
    ) -> Response:
        """Run one vignette through the pipeline and return its ExtractionRecord.

        Validates the upload (MIME prefix + size) before doing any work, decodes
        the image (lazy Pillow import), then calls the lazily-built pipeline
        singleton. The chosen ``flow`` selects the abstention profile.
        """
        req = ExtractRequest(flow=flow, idempotency_key=idempotency_key)
        chosen_flow: Flow = req.flow or settings.default_flow
        req_id = req.idempotency_key or uuid.uuid4().hex
        rlog = log.bind(
            request_id=req_id,
            idempotency_key=req.idempotency_key,
            flow=chosen_flow,
            filename=file.filename,
        )

        data = await _read_validated_upload(file, settings, rlog)
        image = _decode_image(data, rlog)

        # Lazy pipeline build — keeps app import/boot free of ML libs.
        try:
            pipeline: PipelineLike = get_pipeline()
        except ImportError as exc:
            rlog.error("extract_pipeline_unavailable", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Extraction pipeline is not available.",
            ) from exc

        t0 = time.perf_counter()
        try:
            record: ExtractionRecord = pipeline.extract(image, flow=chosen_flow)
        except Exception as exc:  # noqa: BLE001 — surface as 500 w/ structured log
            rlog.error("extract_failed", error=str(exc), error_type=type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Extraction failed while processing the image.",
            ) from exc
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        rlog.info(
            "extract_ok",
            image_id=record.image_id,
            stub=is_stub(pipeline),
            n_fields=len(record.fields),
            n_abstentions=len(record.abstentions),
            checksum_verdict=record.checksum.verdict,
            reimbursability=record.reimbursability.color,
            elapsed_ms=elapsed_ms,
        )

        # Echo the idempotency key so the caller can correlate retries.
        headers = {"X-Request-ID": req_id}
        if req.idempotency_key:
            headers["Idempotency-Key"] = req.idempotency_key
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=record.model_dump(mode="json"),
            headers=headers,
        )

    return app


# --------------------------------------------------------------------------- #
# Upload helpers (validation + decode). Kept module-level for unit testing.
# --------------------------------------------------------------------------- #


async def _read_validated_upload(file: UploadFile, settings: Settings, rlog: Any) -> bytes:
    """Validate MIME prefix and size, returning the raw bytes.

    Raises ``HTTPException`` 415 (bad media type) or 413 (too large). The body is
    read in bounded chunks so an oversized upload is rejected without buffering
    the whole thing in memory.
    """
    content_type = (file.content_type or "").lower()
    if not content_type.startswith(settings.allowed_upload_prefix):
        rlog.warning("upload_rejected_mime", content_type=content_type or "<none>")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported Content-Type {content_type or '<none>'!r}; "
                f"expected {settings.allowed_upload_prefix}*."
            ),
        )

    max_bytes = settings.max_upload_bytes
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1 << 20  # 1 MiB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            rlog.warning("upload_rejected_size", bytes_read=total, limit=max_bytes)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Upload exceeds limit of {settings.max_upload_mb} MB.",
            )
        chunks.append(chunk)

    if total == 0:
        rlog.warning("upload_rejected_empty")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty upload.",
        )
    return b"".join(chunks)


def _decode_image(data: bytes, rlog: Any) -> Any:
    """Decode raw bytes into a PIL image (lazy Pillow import).

    Returns an ``RGB`` ``PIL.Image.Image``. Raises ``HTTPException`` 422 if the
    bytes are not a decodable image, or 503 if Pillow is somehow unavailable
    (a core dependency, so this should not happen in a correct install).
    """
    try:
        import io  # noqa: PLC0415

        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover — Pillow is a core dep
        rlog.error("pillow_unavailable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image decoding backend unavailable; run: pip install -e .",
        ) from exc

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        rlog.warning("image_decode_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode the uploaded image.",
        ) from exc


# Module-level ASGI app for ``uvicorn vignocr.serving.app:app``.
app = create_app()
