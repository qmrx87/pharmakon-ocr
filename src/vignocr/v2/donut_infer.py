"""v2a inference: a fine-tuned Donut reads a vignette image into field values.

``DonutExtractor.extract(pil) -> {field: (value, confidence)}``. Confidence is
the exponentiated mean token log-probability of the generated sequence — one
sequence-level score shared by its fields (Donut decodes all fields jointly;
a per-field decomposition would be pseudo-precision). The downstream abstention
gate treats it exactly like an OCR line score.

Heavy libs are imported lazily; constructing the extractor is cheap until the
first ``extract``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vignocr.common import get_logger
from vignocr.v2.donut_format import token2json

log = get_logger(__name__)

_V2_HINT = "Donut inference needs the ml+v2 extras. Run: pip install -e .[ml,v2]"


class DonutExtractor:
    """Load a fine-tuned Donut checkpoint dir and read vignettes."""

    def __init__(self, model_dir: str | Path, *, device: str | None = None) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Donut checkpoint dir not found: {self.model_dir}")
        meta_path = self.model_dir / "vignocr_donut.json"
        self._meta: dict[str, Any] = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        )
        self._device = device
        self._model: Any | None = None
        self._processor: Any | None = None

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_V2_HINT) from exc
        self._processor = DonutProcessor.from_pretrained(self.model_dir)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_dir)
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()
        log.info("donut.loaded", dir=str(self.model_dir), device=self._device)

    # ------------------------------------------------------------------ #
    def extract(self, pil: Any) -> dict[str, tuple[str, float]]:
        """Read one vignette image; returns ``{field: (raw_value, confidence)}``."""
        self._load()
        import torch

        assert self._model is not None and self._processor is not None
        tok = self._processor.tokenizer
        task_token = self._meta.get("task_start_token", "<s_vignocr>")
        max_length = int(self._meta.get("max_length", 192))

        pixel = self._processor(pil.convert("RGB"), return_tensors="pt").pixel_values.to(
            self._device
        )
        prompt = torch.full((1, 1), tok.convert_tokens_to_ids(task_token), device=self._device)
        with torch.no_grad():
            out = self._model.generate(
                pixel,
                decoder_input_ids=prompt,
                max_length=max_length,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        seq = out.sequences[0]
        # Sequence confidence = exp(mean log-prob of the sampled tokens).
        conf = 1.0
        if out.scores:
            logps = []
            for tok_idx, score in zip(seq[1:], out.scores, strict=False):
                lp = torch.log_softmax(score[0], dim=-1)[tok_idx]
                logps.append(float(lp))
            if logps:
                conf = float(torch.tensor(logps).mean().exp())
        values = token2json(tok.decode(seq, skip_special_tokens=False))
        return {f: (v, conf) for f, v in values.items()}
