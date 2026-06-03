"""Serving layer — FastAPI TestClient (httpx) against the stub-backed pipeline.

Covers the HTTP contract on a CPU-only box:
* ``GET /health`` — liveness, always 200 ``{"status": "ok"}``;
* ``GET /ready``  — readiness, 200 with model versions (stub here);
* ``POST /extract`` — a synthetic image returns a valid ``ExtractionRecord``;
* an oversized upload (413) and a non-image upload (415) are rejected with 4xx.

The serving path resolves an in-memory upload to its fixture identity by content
hash, so we upload the *actual bytes* of a generated fixture image to get a fully
populated record.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vignocr.common.config import get_active_dataset
from vignocr.data.coco import load_split
from vignocr.serving import deps
from vignocr.serving.app import create_app
from vignocr.serving.deps import Settings


@pytest.fixture(autouse=True)
def _reset_pipeline_singleton():
    """Each test starts with a fresh pipeline/settings singleton."""
    deps.reset_pipeline_cache()
    yield
    deps.reset_pipeline_cache()


@pytest.fixture
def client() -> TestClient:
    """A TestClient over the default app (stub pipeline, default settings)."""
    return TestClient(create_app())


@pytest.fixture
def fixture_image(synthetic_root: Path) -> tuple[str, bytes]:
    """The (file_name, bytes) of the first generated fixture image."""
    split = next(iter(get_active_dataset().get("splits", {}).values()))
    sp = load_split(synthetic_root, split)
    file_name = sp.images[0]["file_name"]
    data = (synthetic_root / split / file_name).read_bytes()
    return file_name, data


# --------------------------------------------------------------------------- #
# Liveness / readiness
# --------------------------------------------------------------------------- #


def test_health_is_ok(client: TestClient) -> None:
    """``GET /health`` returns 200 with the liveness payload."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_reports_stub_models(client: TestClient) -> None:
    """``GET /ready`` returns 200 and reports the stub-backend model versions.

    The orchestrator imports fine on CPU, so the real ``VignocrPipeline`` is built
    (``stub`` flag is False — that flag means the serving-layer ``_StubPipeline``
    fallback, used only when the orchestrator/ML deps are absent). With no torch
    and no weights, its *backend* resolves to the deterministic stub, which the
    reported model versions make explicit (``detector == "stub"``).
    """
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert set(body["models"]) == {"detector", "recognizer", "nomenclature_version"}
    # Real pipeline object, but stub backend (no ML libs / weights here).
    assert body["stub"] is False
    assert body["models"]["detector"] == "stub"
    assert body["models"]["recognizer"] == "stub"
    assert body["flow_default"] == "selling"


# --------------------------------------------------------------------------- #
# Extract: a valid ExtractionRecord
# --------------------------------------------------------------------------- #


def test_extract_returns_valid_extraction_record(
    client: TestClient, fixture_image: tuple[str, bytes], ground_truth: dict
) -> None:
    """POST /extract with a synthetic image returns a well-formed ExtractionRecord."""
    file_name, data = fixture_image
    resp = client.post(
        "/extract",
        files={"file": (file_name, data, "image/jpeg")},
        data={"flow": "selling"},
    )
    assert resp.status_code == 200, resp.text

    # The body validates against the canonical contract.
    from vignocr.common.schemas import ExtractionRecord

    record = ExtractionRecord.model_validate(resp.json())
    body = resp.json()

    # Resolved to the right fixture by content hash -> fields match ground truth.
    assert record.image_id == file_name
    assert record.flow == "selling"
    gt = ground_truth[file_name]
    for name, expected in gt["fields"].items():
        assert record.fields[name].value == expected

    # Reports are present and coherent.
    assert record.checksum.verdict == "ok"
    assert record.reimbursability.color == gt["reimbursability"]
    assert record.nomenclature.matched is True

    # Money is serialized as Decimal-strings, not floats, in the JSON body.
    assert isinstance(body["fields"]["ppa"]["value"], str)
    assert isinstance(body["checksum"]["ppa"], str)

    # The request id is echoed for correlation.
    assert "X-Request-ID" in resp.headers


def test_extract_defaults_to_configured_flow(
    client: TestClient, fixture_image: tuple[str, bytes]
) -> None:
    """Omitting ``flow`` falls back to the server's default (selling)."""
    file_name, data = fixture_image
    resp = client.post("/extract", files={"file": (file_name, data, "image/jpeg")})
    assert resp.status_code == 200
    assert resp.json()["flow"] == "selling"


def test_extract_echoes_idempotency_key(
    client: TestClient, fixture_image: tuple[str, bytes]
) -> None:
    """A supplied idempotency key is echoed back in the response headers."""
    file_name, data = fixture_image
    resp = client.post(
        "/extract",
        files={"file": (file_name, data, "image/jpeg")},
        data={"idempotency_key": "abc-123"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("Idempotency-Key") == "abc-123"


# --------------------------------------------------------------------------- #
# Upload rejection: 4xx for bad media type / oversized / unreadable
# --------------------------------------------------------------------------- #


def test_non_image_upload_is_rejected_4xx(client: TestClient) -> None:
    """A non-image Content-Type is rejected with 415 (a 4xx)."""
    resp = client.post(
        "/extract",
        files={"file": ("note.txt", b"this is not an image", "text/plain")},
    )
    assert resp.status_code == 415
    assert 400 <= resp.status_code < 500


def test_unreadable_image_is_rejected_4xx(client: TestClient) -> None:
    """Image Content-Type but undecodable bytes -> 422 (a 4xx), never a crash."""
    resp = client.post(
        "/extract",
        files={"file": ("broken.jpg", b"not-a-real-jpeg-payload", "image/jpeg")},
    )
    assert resp.status_code == 422
    assert 400 <= resp.status_code < 500


def test_oversized_upload_is_rejected_4xx(fixture_image: tuple[str, bytes]) -> None:
    """An upload larger than the configured limit is rejected with 413 (a 4xx)."""
    # Tiny limit so the real fixture image (a few KB) exceeds it.
    tiny_settings = Settings(max_upload_mb=0.0001)  # ~104 bytes
    client = TestClient(create_app(tiny_settings))

    file_name, data = fixture_image
    assert len(data) > tiny_settings.max_upload_bytes, "fixture too small to exceed limit"

    resp = client.post(
        "/extract",
        files={"file": (file_name, data, "image/jpeg")},
    )
    assert resp.status_code == 413
    assert 400 <= resp.status_code < 500


def test_empty_upload_is_rejected_4xx(client: TestClient) -> None:
    """An empty body is rejected (422), never silently processed."""
    resp = client.post(
        "/extract",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert 400 <= resp.status_code < 500
