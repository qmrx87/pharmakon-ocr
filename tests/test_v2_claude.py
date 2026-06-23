"""CPU-only tests for the v2c Claude API variant.

The anthropic SDK and network are mocked — `ClaudeExtractor` imports anthropic
lazily and short-circuits when `_client` is already set, so these run with no
SDK installed and no ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from vignocr.v2.claude_extract import ClaudeExtractor


def _resp(text: str, stop_reason: str = "end_turn") -> Any:
    block = types.SimpleNamespace(type="text", text=text)
    return types.SimpleNamespace(content=[block], stop_reason=stop_reason, stop_details=None)


class _FakeMessages:
    def __init__(self, resp: Any) -> None:
        self._resp = resp
        self.calls: list[dict[str, Any]] = []

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        return self._resp


class _FakeClient:
    def __init__(self, resp: Any) -> None:
        self.messages = _FakeMessages(resp)


def _img() -> Any:
    from PIL import Image

    return Image.new("RGB", (64, 32), (255, 255, 255))


def test_claude_extract_maps_fields_and_drops_abstentions() -> None:
    ex = ClaudeExtractor()
    payload = {
        "num_lot": {"value": "B1234", "confidence": 0.9},
        "date_exp": {"value": "05/2027", "confidence": 0.8},
        "tr": {"value": "", "confidence": 0.1},  # empty -> abstained -> dropped
    }
    ex._client = _FakeClient(_resp(json.dumps(payload)))
    out = ex.extract(_img())
    assert out["num_lot"] == ("B1234", 0.9)
    assert out["date_exp"][0] == "05/2027"
    assert "tr" not in out

    kw = ex._client.messages.calls[0]
    assert kw["model"].startswith("claude")
    assert kw["output_config"]["format"]["type"] == "json_schema"
    content = kw["messages"][0]["content"]
    assert any(b.get("type") == "image" for b in content)
    # system prompt is cached
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_claude_refusal_returns_empty() -> None:
    ex = ClaudeExtractor()
    ex._client = _FakeClient(_resp("", stop_reason="refusal"))
    assert ex.extract(_img()) == {}


def test_claude_bad_json_returns_empty() -> None:
    ex = ClaudeExtractor()
    ex._client = _FakeClient(_resp("not json at all"))
    assert ex.extract(_img()) == {}


def test_claude_confidence_clamped_to_unit_interval() -> None:
    ex = ClaudeExtractor()
    ex._client = _FakeClient(_resp(json.dumps({"num_lot": {"value": "X", "confidence": 5.0}})))
    assert ex.extract(_img())["num_lot"][1] == 1.0


def test_pipeline_variant_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    from vignocr.pipeline.orchestrator import VignocrPipeline

    monkeypatch.setenv("VIGNOCR_PIPELINE_VARIANT", "claude")
    p = VignocrPipeline()
    assert p._variant == "claude"
    versions = p.model_versions()
    assert versions["variant"] == "claude"
    assert versions["recognizer"].startswith("claude")  # the configured model id
    assert versions["detector"] == "none"
