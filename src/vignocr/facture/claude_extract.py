"""FactureOCR — read a supplier invoice image via the Claude API.

``FactureExtractor.extract(pil) -> {supplier, invoice_number, invoice_date,
client, lines:[...], totals:{...}}`` — one VLM call returns the whole invoice as
structured JSON (robust to the per-distributor layout variation a template parser
can't handle). ``extract_and_verify(pil)`` additionally runs the deterministic
arithmetic check (:mod:`vignocr.facture.verify`).

Needs network + ``ANTHROPIC_API_KEY``; the ``anthropic`` SDK is imported lazily
so importing this module stays CPU/offline-safe. Output is always
PREFILL-AND-CONFIRM input for stock intake — never auto-commit.

PRIVACY: ``extract`` sends the invoice IMAGE to Anthropic's API.
"""

from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING, Any

from vignocr.common import get_logger, load_config
from vignocr.facture.verify import verify_facture

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

log = get_logger(__name__)

_HINT = "FactureOCR needs the anthropic SDK. Run: pip install -e .[claude]"
_MAX_EDGE = 2200  # cap the long edge before encoding (legible + payload-safe)


class FactureExtractor:
    """Read a supplier-invoice image into structured ``{header, lines, totals}``."""

    def __init__(self, cfg_path: str = "facture/claude") -> None:
        self._cfg = load_config(cfg_path)
        self._mcfg = self._cfg.get("model", {}) or {}
        self._line_fields: list[str] = list(self._cfg.get("line_fields", []))
        self._client: Any | None = None

    # ------------------------------------------------------------------ #
    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic  # noqa: PLC0415 - lazy: keeps module offline-importable
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_HINT) from exc
        self._client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from env
        log.info("facture.client_init", model=self._mcfg.get("id", "claude-opus-4-8"))

    def _output_schema(self) -> dict[str, Any]:
        line_props: dict[str, Any] = {f: {"type": "string"} for f in self._line_fields}
        line_props["confidence"] = {"type": "number"}
        line_item = {
            "type": "object",
            "properties": line_props,
            "required": [*self._line_fields, "confidence"],
            "additionalProperties": False,
        }
        totals = {
            "type": "object",
            "properties": {
                "net_a_payer": {"type": "string"},
                "total_ht": {"type": "string"},
                "net_ht": {"type": "string"},
            },
            "required": ["net_a_payer", "total_ht", "net_ht"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "supplier": {"type": "string"},
                "invoice_number": {"type": "string"},
                "invoice_date": {"type": "string"},
                "client": {"type": "string"},
                "lines": {"type": "array", "items": line_item},
                "totals": totals,
            },
            "required": ["supplier", "invoice_number", "invoice_date", "client", "lines", "totals"],
            "additionalProperties": False,
        }

    @staticmethod
    def _encode(pil: Image) -> str:
        img = pil.convert("RGB")
        w, h = img.size
        scale = _MAX_EDGE / max(w, h)
        if scale < 1.0:
            from PIL import Image as PILImage  # noqa: PLC0415

            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------ #
    def extract(self, pil: Image) -> dict[str, Any]:
        """One invoice image -> raw structured dict (header, lines, totals)."""
        self._ensure_client()
        assert self._client is not None
        pcfg = self._cfg.get("prompt", {}) or {}
        b64 = self._encode(pil)

        req: dict[str, Any] = {
            "model": str(self._mcfg.get("id", "claude-opus-4-8")),
            "max_tokens": int(self._mcfg.get("max_tokens", 16000)),
            "system": [
                {
                    "type": "text",
                    "text": str(pcfg.get("system", "Extract the supplier invoice as JSON.")),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        },
                        {"type": "text", "text": str(pcfg.get("user", "Extract this invoice as JSON."))},
                    ],
                }
            ],
            "output_config": {"format": {"type": "json_schema", "schema": self._output_schema()}},
        }
        think = self._mcfg.get("thinking", {}) or {}
        if think.get("enabled", True):
            req["thinking"] = {"type": "adaptive"}
            req["output_config"]["effort"] = str(think.get("effort", "low"))

        resp = self._client.messages.create(**req)

        if getattr(resp, "stop_reason", None) == "refusal":
            log.warning("facture.refusal", details=str(getattr(resp, "stop_details", None)))
            return {"supplier": "", "invoice_number": "", "invoice_date": "", "client": "",
                    "lines": [], "totals": {}}
        if getattr(resp, "stop_reason", None) == "max_tokens":
            log.warning("facture.truncated", note="raise model.max_tokens — long invoice JSON cut off")

        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            log.warning("facture.bad_json", head=text[:300])
            return {"supplier": "", "invoice_number": "", "invoice_date": "", "client": "",
                    "lines": [], "totals": {}}
        log.info("facture.extract", n_lines=len(data.get("lines", []) or []), model=req["model"])
        return data

    def extract_and_verify(self, pil: Image) -> dict[str, Any]:
        """Extract then run the deterministic arithmetic check (the shippable path)."""
        raw = self.extract(pil)
        vcfg = self._cfg.get("verify", {}) or {}
        return verify_facture(
            raw,
            line_tolerance=float(vcfg.get("line_tolerance", 0.02)),
            total_tolerance=float(vcfg.get("total_tolerance", 0.02)),
        )
