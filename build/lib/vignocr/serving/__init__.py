"""HTTP serving layer — stateless FastAPI in front of the VignOCR pipeline.

The app is 12-factor: every knob comes from the environment (see
:mod:`vignocr.serving.deps`), it holds no per-request state, and it emits the
canonical :class:`vignocr.common.schemas.ExtractionRecord` straight from the
pipeline so the consumer can *prefill-and-confirm* (abstentions, per-field
confidence, and the checksum verdict ride along in the response).

Heavy ML libs and the pipeline itself are imported **lazily** inside request
handlers / the cached singleton, so ``uvicorn vignocr.serving.app:app`` boots on
a CPU-only box — even before the pipeline module exists (stub mode).

Public API::

    from vignocr.serving import create_app
    app = create_app()
"""

from vignocr.serving.app import app, create_app

__all__ = ["app", "create_app"]
