"""Fuzzy-match a read ``num_enregistrement`` against the nomenclature index.

Strategy (``match.strategy = structural_edit_distance``): the code has a known
shape — ``AA/BB/CC<LETTER>DDD/EEE`` (see ``parsing/fields.yaml``) — so we don't run
a flat string distance. We split both the query and each candidate into the six
blocks ``a,b,c,letter,d,e`` and sum a **per-block weighted** Levenshtein distance.
The letter block carries the most identity (``block_weights.letter = 2.0``), so a
change there costs more than a change in a digit block.

Two gates from ``configs/nomenclature/correction.yaml`` decide a match:
    * ``max_edit_distance``    — cap on the *raw* (unweighted) char edits.
    * ``min_match_confidence`` — floor on the derived confidence in ``[0, 1]``.

Below either gate -> ``(None, confidence)`` so the caller keeps OCR values and
flags the anchor. ``rapidfuzz`` is a **core** dependency (CPU, no ML), so it is
imported at module top-level.

Normalization (``normalize_code``) is exposed here because both the loader (to
build keys) and the matcher (to canonicalize the query) must use the *same*
canonical space.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from rapidfuzz.distance import Levenshtein

from vignocr.common import get_logger, load_config

log = get_logger(__name__)

# Structural block order of AA/BB/CC<LETTER>DDD/EEE. ``letter`` is the only
# non-digit slot; the confusion map applies to the digit slots only.
_BLOCK_ORDER: tuple[str, ...] = ("a", "b", "c", "letter", "d", "e")
_DIGIT_BLOCKS: frozenset[str] = frozenset({"a", "b", "c", "d", "e"})


# --------------------------------------------------------------------------- #
# Config access (cached); structural regex comes from parsing/fields.yaml so the
# code structure is defined once.
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _fields_cfg() -> dict[str, Any]:
    return load_config("parsing/fields").get("fields", {}).get("num_enregistrement", {})


@lru_cache(maxsize=1)
def _structural_regex() -> re.Pattern[str] | None:
    pattern = _fields_cfg().get("regex")
    return re.compile(pattern) if pattern else None


@lru_cache(maxsize=1)
def _confusion_map() -> dict[str, str]:
    return dict(_fields_cfg().get("confusion_map", {}))


def _normalized_format() -> str:
    return _fields_cfg().get("normalized_format", "{a}/{b}/{c}{letter}{d}/{e}")


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def _apply_confusion_to_digits(value: str) -> str:
    """Map OCR letter→digit confusions (O->0, I->1, ...) over a digit slot."""
    cmap = _confusion_map()
    if not cmap:
        return value
    return "".join(cmap.get(ch, ch) for ch in value)


def split_blocks(code: str) -> dict[str, str] | None:
    """Parse a code into its structural blocks, or ``None`` if it doesn't match.

    Uses the ``num_enregistrement`` regex from ``parsing/fields.yaml`` — keys are
    ``a, b, c, letter, d, e``.
    """
    rx = _structural_regex()
    if rx is None:
        return None
    m = rx.match(code)
    if not m:
        return None
    return {b: m.group(b) for b in _BLOCK_ORDER}


def normalize_code(text: str | None, cfg: dict[str, Any] | None = None) -> str:
    """Canonicalize a read code per ``correction.yaml.normalize``.

    Steps (each gated by config): uppercase, strip internal spaces, glue the
    letter block (the ``AA/BB/CC<LETTER>DDD/EEE`` shape), and apply the
    ``confusion_map`` (from ``parsing/fields.yaml``) to **digit** slots only.

    The function is idempotent: feeding it an already-normalized code returns the
    same string. Returns ``""`` for empty input.
    """
    if not text:
        return ""
    cfg = cfg if cfg is not None else load_config("nomenclature/correction")
    norm_cfg = cfg.get("normalize", {})

    out = text.strip()
    if norm_cfg.get("uppercase", True):
        out = out.upper()
    if norm_cfg.get("strip_internal_spaces", True):
        out = re.sub(r"\s+", "", out)

    # Confusion remap on digit slots (O->0, I->1, ...). Two cases:
    #   * code already parses -> remap ONLY the digit blocks, leaving the
    #     identity-bearing letter block untouched.
    #   * code does NOT parse -> the OCR confusion may be what broke the structure
    #     (e.g. "OO/O7/3OP126/317"). Try remapping every char and re-parsing; adopt
    #     the result only if it now parses (so we never corrupt a valid code, and
    #     only ever repair *toward* the canonical shape).
    if norm_cfg.get("apply_confusion_map", True):
        blocks = split_blocks(out)
        if blocks is not None:
            fixed = {
                b: (_apply_confusion_to_digits(v) if b in _DIGIT_BLOCKS else v)
                for b, v in blocks.items()
            }
            out = _normalized_format().format(**fixed)
        else:
            remapped = _apply_confusion_to_digits(out)  # map every char
            reparsed = split_blocks(remapped)
            if reparsed is not None:
                out = _normalized_format().format(**reparsed)
    # glue_letter_block: after stripping internal spaces the letter block is
    # already contiguous; the structural regex + normalized_format above enforce
    # the canonical AA/BB/CC<LETTER>DDD/EEE shape when parseable.
    return out


# --------------------------------------------------------------------------- #
# Structural distance + confidence
# --------------------------------------------------------------------------- #


def _block_weights(cfg: dict[str, Any]) -> dict[str, float]:
    weights = cfg.get("match", {}).get("block_weights", {})
    return {b: float(weights.get(b, 1.0)) for b in _BLOCK_ORDER}


def _weighted_distance(
    query_blocks: dict[str, str],
    cand_blocks: dict[str, str],
    weights: dict[str, float],
) -> tuple[float, int]:
    """Return ``(weighted_distance, raw_edits)`` across the structural blocks."""
    weighted = 0.0
    raw = 0
    for b in _BLOCK_ORDER:
        d = Levenshtein.distance(query_blocks[b], cand_blocks[b])
        raw += d
        weighted += weights[b] * d
    return weighted, raw


def _max_weighted(blocks: dict[str, str], weights: dict[str, float]) -> float:
    """Upper bound on weighted distance: editing every char of every block."""
    return sum(weights[b] * max(len(blocks[b]), 1) for b in _BLOCK_ORDER)


def _confidence_from_distance(weighted: float, max_weighted: float) -> float:
    if max_weighted <= 0:
        return 0.0
    return max(0.0, 1.0 - weighted / max_weighted)


def _flat_score(query: str, candidate: str) -> tuple[float, int]:
    """Fallback when a code isn't structurally parseable: flat Levenshtein.

    Returns ``(confidence, raw_edits)``.
    """
    raw = Levenshtein.distance(query, candidate)
    denom = max(len(query), len(candidate), 1)
    return max(0.0, 1.0 - raw / denom), raw


def score(query_norm: str, candidate_norm: str, cfg: dict[str, Any]) -> tuple[float, int]:
    """Score a single (normalized) query/candidate pair.

    Returns ``(confidence_in_[0,1], raw_edit_distance)``. Uses the structural
    block distance when both parse; otherwise a flat normalized Levenshtein.
    """
    if query_norm == candidate_norm:
        return 1.0, 0
    qb = split_blocks(query_norm)
    cb = split_blocks(candidate_norm)
    if qb is not None and cb is not None:
        weights = _block_weights(cfg)
        weighted, raw = _weighted_distance(qb, cb, weights)
        max_w = _max_weighted(qb, weights)
        return _confidence_from_distance(weighted, max_w), raw
    return _flat_score(query_norm, candidate_norm)


# --------------------------------------------------------------------------- #
# Public matcher
# --------------------------------------------------------------------------- #


def find(
    norm_code: str | None,
    index: "Any",
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, str] | None, float]:
    """Find the best nomenclature row for a read ``num_enregistrement``.

    Args:
        norm_code: the read code (normalized or raw — it is re-normalized here so
            the call is robust either way).
        index: a :class:`vignocr.nomenclature.loader.NomenclatureIndex`.
        cfg: ``correction.yaml`` dict; loaded if omitted.

    Returns:
        ``(row, confidence)`` where ``row`` is the matched ``{column: value}`` dict
        and ``confidence`` is in ``[0, 1]``. If nothing clears both
        ``max_edit_distance`` and ``min_match_confidence``, returns
        ``(None, best_confidence_seen)`` (so callers can log how close it was).
    """
    cfg = cfg if cfg is not None else load_config("nomenclature/correction")
    match_cfg = cfg.get("match", {})
    max_edit = int(match_cfg.get("max_edit_distance", 2))
    min_conf = float(match_cfg.get("min_match_confidence", 0.75))

    query = normalize_code(norm_code, cfg)
    if not query or index is None or getattr(index, "is_empty", True):
        return None, 0.0

    # Exact hit short-circuit (confidence 1.0).
    exact = index.get(query)
    if exact is not None:
        return exact, 1.0

    best_row: dict[str, str] | None = None
    best_conf = 0.0
    best_raw = 0
    for key in index.keys():
        conf, raw = score(query, key, cfg)
        if conf > best_conf:
            best_conf, best_raw = conf, raw
            best_row = index.get(key)

    if best_row is not None and best_raw <= max_edit and best_conf >= min_conf:
        log.info(
            "nomenclature.match",
            query=query,
            confidence=round(best_conf, 4),
            raw_edits=best_raw,
        )
        return best_row, best_conf

    log.info(
        "nomenclature.no_match",
        query=query,
        best_confidence=round(best_conf, 4),
        best_raw_edits=best_raw,
        max_edit_distance=max_edit,
        min_match_confidence=min_conf,
    )
    return None, best_conf
