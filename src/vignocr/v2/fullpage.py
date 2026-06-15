"""v2b: full-page docTR OCR + deterministic layout parsing — zero training.

Flow:  vignette crop -> docTR (det + PARSeq rec) on the crop AND on the crop
rotated -90 deg (the lot/dates strip is vertical) -> ``assign_fields`` maps the
recognized words to vignette fields with config-driven patterns:

* dates  : tokens matching ``date_patterns``; chronology disambiguates —
           manufacture is ALWAYS before expiry.
* num_lot: the closest non-date alphanum token to the date pair (same pass),
           shaped like ``lot_pattern``.
* num_enregistrement: best ``code_pattern`` match (longest wins).
* ppa    : the largest ``money_pattern`` amount (vignette prices ppa >= tr).

``assign_fields`` is a pure function over ``Word`` tuples so the layout logic is
unit-tested on CPU without docTR installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vignocr.common import get_logger, load_config

log = get_logger(__name__)

_V2_HINT = "Full-page OCR needs the v2 extra (python-doctr). Run: pip install -e .[ml,v2]"


@dataclass(frozen=True)
class Word:
    """One recognized word: text + confidence + center (relative 0-1) + pass id."""

    text: str
    conf: float
    cx: float
    cy: float
    rotated: bool  # True if read on the -90deg pass (vertical strip text)


# --------------------------------------------------------------------------- #
# Pure layout parser (CPU-testable)
# --------------------------------------------------------------------------- #
def _date_key(tok: str) -> tuple[int, int, int]:
    """Sortable (year, month, day) from a date-ish token; lenient on format."""
    parts = [int(p) for p in re.split(r"[/\-.\s]+", tok) if p.isdigit()]
    if not parts:
        return (9999, 12, 31)
    if len(parts) == 2:  # MM/YYYY or MM/YY
        m, y = parts
        if m > 12 and y <= 12:  # actually YYYY/MM
            m, y = y, m
        return (y if y > 99 else 2000 + y, m, 1)
    d, m, y = parts[0], parts[1], parts[-1]
    if d > 31:  # YYYY first
        y, m, d = parts[0], parts[1], parts[-1]
    return (y if y > 99 else 2000 + y, min(m, 12), min(d, 31))


def assign_fields(words: list[Word], pcfg: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """Map recognized words to vignette fields; returns ``{field: (value, conf)}``."""
    date_res = [re.compile(p) for p in pcfg.get("date_patterns", [])]
    lot_re = re.compile(str(pcfg.get("lot_pattern", r"\b[A-Z0-9\-]{3,}\b")))
    code_re = re.compile(str(pcfg.get("code_pattern", r"\b\d{2}[/ ]\w+[/ ]\d{2,4}\b")))
    money_re = re.compile(str(pcfg.get("money_pattern", r"\b\d{1,6}[.,]\d{2}\b")))
    discount = float(pcfg.get("confidence_discount", 0.9))

    out: dict[str, tuple[str, float]] = {}

    # --- dates: collect across passes, prefer the pass that found >= 2 -------
    dated: list[tuple[Word, str]] = []
    for w in words:
        for rx in date_res:
            m = rx.search(w.text.upper())
            if m:
                dated.append((w, m.group(0)))
                break
    by_pass: dict[bool, list[tuple[Word, str]]] = {True: [], False: []}
    for w, tok in dated:
        by_pass[w.rotated].append((w, tok))
    strip_pass = max(by_pass, key=lambda k: len(by_pass[k])) if dated else None
    pair = sorted(by_pass.get(strip_pass, []), key=lambda t: _date_key(t[1]))[:2] \
        if strip_pass is not None else []
    if len(pair) >= 2:
        (w_fab, fab), (w_exp, exp) = pair[0], pair[1]
        out["date_fab"] = (fab, w_fab.conf * discount)
        out["date_exp"] = (exp, w_exp.conf * discount)
    elif len(pair) == 1:
        # A single date on a vignette is the expiry far more often than the
        # manufacture date; emit only date_exp and let abstention flag the rest.
        out["date_exp"] = (pair[0][1], pair[0][0].conf * discount)

    # --- num_lot: nearest non-date lot-shaped token to the dates (same pass) --
    date_words = {id(w) for w, _ in dated}
    candidates = [
        w for w in words
        if id(w) not in date_words and lot_re.search(w.text.upper())
        and not money_re.search(w.text)  # an amount is never the lot
        and (strip_pass is None or w.rotated == strip_pass)
    ]
    if candidates:
        if pair:
            cyx = (sum(w.cx for w, _ in pair) / len(pair), sum(w.cy for w, _ in pair) / len(pair))
            best = min(candidates, key=lambda w: (w.cx - cyx[0]) ** 2 + (w.cy - cyx[1]) ** 2)
        else:
            best = max(candidates, key=lambda w: w.conf)
        out["num_lot"] = (lot_re.search(best.text.upper()).group(0), best.conf * discount)

    # --- num_enregistrement: best code-pattern match anywhere -----------------
    codes = [(w, m.group(0)) for w in words if (m := code_re.search(w.text.upper()))]
    if codes:
        w, tok = max(codes, key=lambda t: (len(t[1]), t[0].conf))
        out["num_enregistrement"] = (tok, w.conf * discount)

    # --- ppa: the LARGEST amount on the vignette ------------------------------
    monies = [(w, m.group(0)) for w in words if (m := money_re.search(w.text))]
    if monies:
        def _amt(tok: str) -> float:
            return float(tok.replace(",", "."))
        w, tok = max(monies, key=lambda t: _amt(t[1]))
        out["ppa"] = (tok, w.conf * discount)

    return out


# --------------------------------------------------------------------------- #
# docTR-backed extractor
# --------------------------------------------------------------------------- #
class FullPageExtractor:
    """Pretrained docTR det+rec over the vignette, two passes, layout-parsed."""

    def __init__(self, cfg_path: str = "v2/fullpage_doctr") -> None:
        self._cfg = load_config(cfg_path)
        self._predictor: Any | None = None

    def _load(self) -> None:
        if self._predictor is not None:
            return
        ocfg = self._cfg.get("ocr", {})
        try:
            from doctr.models import ocr_predictor
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_V2_HINT) from exc
        self._predictor = ocr_predictor(
            det_arch=str(ocfg.get("det_arch", "db_mobilenet_v3_large")),
            reco_arch=str(ocfg.get("reco_arch", "parseq")),
            pretrained=bool(ocfg.get("pretrained", True)),
            assume_straight_pages=bool(ocfg.get("assume_straight_pages", True)),
        )
        # Optional fine-tuned PARSeq weights (slurm/15_finetune_parseq output).
        reco_weights = ocfg.get("reco_weights")
        if reco_weights:
            import torch

            sd = torch.load(str(Path(reco_weights)), map_location="cpu")
            self._predictor.reco_predictor.model.load_state_dict(sd)
            log.info("fullpage.reco_weights", path=str(reco_weights))
        log.info("fullpage.loaded", det=ocfg.get("det_arch"), rec=ocfg.get("reco_arch"))

    def _words_one_pass(self, pil: Any, *, rotated: bool) -> list[Word]:
        import numpy as np

        assert self._predictor is not None
        min_conf = float(self._cfg.get("ocr", {}).get("min_word_conf", 0.20))
        res = self._predictor([np.asarray(pil.convert("RGB"))])
        words: list[Word] = []
        for page in res.pages:
            for block in page.blocks:
                for line in block.lines:
                    for w in line.words:
                        if w.confidence < min_conf or not w.value.strip():
                            continue
                        (x0, y0), (x1, y1) = w.geometry  # relative coords
                        words.append(Word(
                            text=w.value, conf=float(w.confidence),
                            cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, rotated=rotated,
                        ))
        return words

    def extract(self, pil: Any) -> dict[str, tuple[str, float]]:
        """OCR the vignette (2 passes) and parse fields; ``{field: (value, conf)}``."""
        self._load()
        words = self._words_one_pass(pil, rotated=False)
        if bool(self._cfg.get("ocr", {}).get("rotated_pass", True)):
            words += self._words_one_pass(pil.rotate(-90, expand=True), rotated=True)
        fields = assign_fields(words, self._cfg.get("parser", {}))
        log.info("fullpage.extract", n_words=len(words), fields=sorted(fields))
        return fields
