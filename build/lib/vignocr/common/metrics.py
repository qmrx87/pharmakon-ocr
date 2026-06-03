"""Light, dependency-free metric helpers shared across modules.

Heavy detection mAP (pycocotools) lives in ``vignocr.detection.eval``; this
module holds the small, CPU-only helpers that parsing/eval/serving reuse —
crucially the *business-critical localization recall* the roadmap gates on.
"""

from __future__ import annotations

from collections.abc import Iterable


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def localization_recall(
    detected_class_names: Iterable[str],
    required_class_names: Iterable[str],
) -> dict[str, object]:
    """Fraction of *required* (business-critical) field classes that were localized.

    The detector can have great mAP yet still miss the one box the business rule
    needs (e.g. ``ppa``). This reports per-class hit/miss + overall recall so the
    roadmap can gate on "did we localize the fields the rule depends on".
    """
    detected = set(detected_class_names)
    required = list(required_class_names)
    hits = {name: (name in detected) for name in required}
    n_hit = sum(hits.values())
    return {
        "recall": (n_hit / len(required)) if required else 1.0,
        "per_class": hits,
        "missing": [n for n, ok in hits.items() if not ok],
    }


def iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two COCO-style ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
