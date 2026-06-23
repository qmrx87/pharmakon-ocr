"""v2c: Claude API vision extractor — vignette image -> field JSON (zero-training).

A frontier multimodal model reads the vignette directly into structured fields:
no detector, no fine-tune. ``ClaudeExtractor.extract(pil) -> {field: (value,
confidence)}`` — the same contract as the Donut/full-page extractors, so it drops
into the v1/v2a/v2b comparison as the ``claude`` variant and shares the
deterministic core (checksum / nomenclature / abstention).

Runs wherever there's network + ``ANTHROPIC_API_KEY`` (Narval LOGIN node or the
cloud backend) — NOT on offline compute nodes. The ``anthropic`` SDK is imported
lazily, so importing this module is CPU/offline-safe.

PRIVACY: ``extract`` sends the vignette IMAGE to Anthropic's API. The API does
not train on your data by default and ZDR is available, but images leave your
infrastructure — clear this for production pharmacy data before enabling.
"""

from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING, Any

from vignocr.common import get_logger, load_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

log = get_logger(__name__)

_HINT = "The claude variant needs the anthropic SDK. Run: pip install -e .[claude]"


class ClaudeExtractor:
    """Read a vignette via the Claude API into ``{field: (value, confidence)}``."""

    def __init__(self, cfg_path: str = "v2/claude") -> None:
        self._cfg = load_config(cfg_path)
        self._mcfg = self._cfg.get("model", {}) or {}
        self._fields: list[str] = list(self._cfg.get("fields", []))
        self._client: Any | None = None

    # ------------------------------------------------------------------ #
    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic  # noqa: PLC0415 - lazy: keeps module offline-importable
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_HINT) from exc
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from the env.
        self._client = anthropic.Anthropic()
        log.info("claude.client_init", model=self._mcfg.get("id", "claude-opus-4-8"))

    def _output_schema(self) -> dict[str, Any]:
        """JSON schema: each field -> {value: str, confidence: number}."""
        field_obj = {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["value", "confidence"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": dict.fromkeys(self._fields, field_obj),
            "required": list(self._fields),
            "additionalProperties": False,
        }

    @staticmethod
    def _encode_png(pil: Image) -> str:
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------ #
    def extract(self, pil: Image) -> dict[str, tuple[str, float]]:
        """One vignette image -> ``{field: (raw_value, confidence)}`` (abstentions dropped)."""
        self._ensure_client()
        assert self._client is not None
        pcfg = self._cfg.get("prompt", {}) or {}
        b64 = self._encode_png(pil)

        req: dict[str, Any] = {
            "model": str(self._mcfg.get("id", "claude-opus-4-8")),
            "max_tokens": int(self._mcfg.get("max_tokens", 8192)),
            # Stable system prompt is cached; the per-scan image goes after it.
            "system": [
                {
                    "type": "text",
                    "text": str(pcfg.get("system", "Extract the vignette fields as JSON.")),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": str(pcfg.get("user", "Extract the fields as JSON."))},
                    ],
                }
            ],
            # Structured outputs: guarantees the text block is schema-valid JSON.
            "output_config": {"format": {"type": "json_schema", "schema": self._output_schema()}},
        }
        think = self._mcfg.get("thinking", {}) or {}
        if think.get("enabled", True):
            req["thinking"] = {"type": "adaptive"}
            req["output_config"]["effort"] = str(think.get("effort", "low"))

        resp = self._client.messages.create(**req)

        # Safety classifiers / model can decline — abstain on everything, never guess.
        if getattr(resp, "stop_reason", None) == "refusal":
            log.warning("claude.refusal", details=str(getattr(resp, "stop_details", None)))
            return {}
        if getattr(resp, "stop_reason", None) == "max_tokens":
            log.warning("claude.truncated", note="raise model.max_tokens — JSON may be incomplete")

        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            log.warning("claude.bad_json", head=text[:200])
            return {}

        out: dict[str, tuple[str, float]] = {}
        for field, rec in (data or {}).items():
            if not isinstance(rec, dict):
                continue
            value = str(rec.get("value", "")).strip()
            if not value:  # empty == model abstained on this field
                continue
            try:
                conf = max(0.0, min(1.0, float(rec.get("confidence", 0.5))))
            except (TypeError, ValueError):
                conf = 0.5
            out[field] = (value, conf)
        log.info("claude.extract", n_fields=len(out), model=req["model"])
        return out
