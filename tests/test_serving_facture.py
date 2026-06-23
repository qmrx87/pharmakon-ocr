"""Serving layer — the FactureOCR endpoint + the /extract/vignette alias.

CPU-only, no network: the facture backend resolves to the honest stub when no
``ANTHROPIC_API_KEY`` is set, and the "real" path is exercised by monkeypatching
the extractor getter. Mirrors ``test_serving_api.py``'s TestClient setup.
"""

from __future__ import annotations

import importlib
import io

import pytest
from fastapi.testclient import TestClient

from vignocr.serving import deps
from vignocr.serving.app import create_app

# The package re-exports the FastAPI instance as ``vignocr.serving.app``, which
# shadows the submodule under attribute/``import ... as`` access — fetch the real
# module object so monkeypatching its ``get_facture_extractor`` reaches the handler.
app_module = importlib.import_module("vignocr.serving.app")


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    deps.reset_pipeline_cache()
    yield
    deps.reset_pipeline_cache()


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (96, 128), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _post_facture(client: TestClient, *, mime: str = "image/png", data: bytes | None = None):
    return client.post(
        "/extract/facture",
        files={"file": ("invoice.png", data if data is not None else _png_bytes(), mime)},
    )


# --------------------------------------------------------------------------- #
# /extract/facture
# --------------------------------------------------------------------------- #


def test_facture_stub_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ANTHROPIC_API_KEY + allow_stub → honest empty record (stub: true)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VIGNOCR_FACTURE_ENABLED", raising=False)
    deps.reset_pipeline_cache()

    resp = _post_facture(TestClient(create_app()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["lines"] == []
    assert body["verification"]["stub"] is True
    assert body["verification"]["needs_review"] is False


def test_facture_disabled_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIGNOCR_FACTURE_ENABLED", "false")
    deps.reset_pipeline_cache()

    resp = _post_facture(TestClient(create_app()))
    assert resp.status_code == 503


def test_facture_returns_verified_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real path: extractor output is passed through verbatim as JSON."""

    class _Fake:
        def extract_and_verify(self, image):  # noqa: ANN001, ARG002
            return {
                "header": {"supplier": "VECOPHARM", "invoice_number": "26/FA241955"},
                "lines": [
                    {
                        "designation": "APROVASC 150MG/5MG COMP.PELLI B/30",
                        "quantity": "20",
                        "unit_price_ht": "1 247.92",
                        "line_total": "24 958.40",
                        "lot": "GNN0194",
                        "confidence": 0.96,
                        "_verify": {"ok": True},
                    }
                ],
                "totals": {"total_ht": "24 958.40"},
                "verification": {"n_lines": 1, "n_line_mismatches": 0, "needs_review": False},
            }

    monkeypatch.setattr(app_module, "get_facture_extractor", lambda: _Fake())

    resp = _post_facture(TestClient(create_app()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["header"]["supplier"] == "VECOPHARM"
    assert body["lines"][0]["lot"] == "GNN0194"
    assert body["verification"]["needs_review"] is False


def test_facture_rejects_non_image() -> None:
    resp = TestClient(create_app()).post(
        "/extract/facture", files={"file": ("x.txt", b"not an image", "text/plain")}
    )
    assert resp.status_code == 415


def test_facture_backend_failure_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def extract_and_verify(self, image):  # noqa: ANN001, ARG002
            raise RuntimeError("api down")

    monkeypatch.setattr(app_module, "get_facture_extractor", lambda: _Boom())
    resp = _post_facture(TestClient(create_app()))
    assert resp.status_code == 502


# --------------------------------------------------------------------------- #
# /extract/vignette alias
# --------------------------------------------------------------------------- #


def test_vignette_alias_matches_extract() -> None:
    """``POST /extract/vignette`` returns a valid ExtractionRecord (stub pipeline)."""
    resp = TestClient(create_app()).post(
        "/extract/vignette", files={"file": ("v.png", _png_bytes(), "image/png")}
    )
    assert resp.status_code == 200
    assert "fields" in resp.json()
