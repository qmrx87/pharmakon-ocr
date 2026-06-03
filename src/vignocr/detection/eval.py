"""Evaluate a trained RF-DETR detector on a COCO split.

Reports the standard COCO detection metrics **and** the business-critical
localization recall the roadmap gates on (Phase 4 exit: ``loc_recall == 1.0`` on
``classes.yaml: business_critical_fields``). Heavy libs (pycocotools / torch via
the ``Detector``) are imported lazily; nothing heavy is imported on module load.

Ground truth is read straight from the split's ``_annotations.coco.json`` with the
stdlib ``json`` module and categories are mapped to schema classes **by name** (so
this composes with whatever category ids a Roboflow export assigns, and does not
depend on the ``data/`` package existing).

Public API (per docs/INTERFACES.md):
    run(ckpt, root, split="valid") -> dict
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vignocr.common import (
    get_classes,
    get_logger,
    load_config,
)
from vignocr.common.metrics import iou_xywh, localization_recall
from vignocr.detection._resolve import resolve_class_schema, resolve_dataset

log = get_logger(__name__)

_ML_HINT = "Detection eval needs the ML extra (pycocotools/torch). Run: pip install -e .[ml]"


def _split_dir(root: Path, split: str, cfg: dict[str, Any] | None = None) -> Path:
    """Resolve the split directory, honouring data.yaml's split-name mapping.

    ``split`` may be a logical key (``train``/``val``/``test``) or a literal dir
    name (``valid``). We consult ``data.yaml: splits`` to translate keys, then
    fall back to the name as-is.
    """
    # Use the eval config's bound dataset (not the global active) so Stage A
    # eval resolves data2's splits, Stage B resolves data/'s splits.
    if cfg is None:
        cfg = load_config("detection/rfdetr_medium")
    ds = resolve_dataset(cfg)
    splits = ds.get("splits", {})
    name = splits.get(split, split)
    candidate = root / name
    if candidate.exists():
        return candidate
    if (root / split).exists():
        return root / split
    raise FileNotFoundError(f"split dir not found under {root}: tried {name!r} and {split!r}")


def _coco_filename(cfg: dict[str, Any] | None = None) -> str:
    if cfg is None:
        cfg = load_config("detection/rfdetr_medium")
    return resolve_dataset(cfg).get("coco_filename", "_annotations.coco.json")


def _load_coco(ann_path: Path) -> dict[str, Any]:
    with open(ann_path, encoding="utf-8") as fh:
        return json.load(fh)


def run(
    ckpt: Path | str,
    root: Path | str,
    split: str = "valid",
    cfg_path: str = "detection/rfdetr_medium",
) -> dict[str, Any]:
    """Evaluate ``ckpt`` on ``root/<split>`` and return a metrics dict.

    Args:
        ckpt: detector weights — an RF-DETR ``.pth``/``.pt`` checkpoint or an
            exported ``.onnx`` graph (the ``Detector`` picks the backend).
        root: dataset root containing the split dirs (Roboflow COCO layout).
        split: split key (``valid``/``test``/``train``) or literal dir name.

    Returns:
        ``{
            "map": float,                 # COCO mAP@[.5:.95]
            "map_50": float,              # mAP@.5
            "map_75": float,
            "per_class_ap": {name: AP},   # per-class AP@[.5:.95]
            "localization_recall": {      # business-critical (classes.yaml)
                "recall": float, "per_class": {name: bool}, "missing": [...]
            },
            "loc_recall_per_image": [...],
            "n_images": int, "split": str, "checkpoint": str,
        }``
    """
    ckpt = Path(ckpt)
    root = Path(root)
    cfg = load_config(cfg_path)
    ds = resolve_dataset(cfg)
    num_classes, class_names = resolve_class_schema(cfg, ds)
    iou_thr = float(cfg.get("eval", {}).get("iou_threshold", 0.5))
    score_thr = float(cfg.get("eval", {}).get("score_threshold", 0.30))

    # Business-critical localization recall is a Stage B concept (classes.yaml).
    # Stage A (dataset-bound class list) has no business-critical fields.
    business_critical: list[str] = []
    if not cfg.get("dataset") or cfg.get("dataset") in {"real", "synthetic"}:
        try:
            business_critical = [
                n for n in get_classes().business_critical_fields if n in class_names
            ]
        except Exception:  # noqa: BLE001
            business_critical = []

    split_dir = _split_dir(root, split, cfg)
    ann_path = split_dir / _coco_filename(cfg)
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {ann_path}")
    coco = _load_coco(ann_path)

    # name <-> coco category id maps for THIS file (robust to Roboflow ids).
    catid2name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    images = {img["id"]: img for img in coco.get("images", [])}
    gt_by_image: dict[int, list[dict[str, Any]]] = {img_id: [] for img_id in images}
    for ann in coco.get("annotations", []):
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    # Lazy: construct the detector (this is where torch/onnxruntime loads).
    from vignocr.detection.infer import Detector

    detector = Detector(ckpt, cfg_path=cfg_path, score_threshold=score_thr)

    predictions: list[dict[str, Any]] = []  # COCO-result format for pycocotools
    loc_recall_per_image: list[dict[str, Any]] = []
    all_detected_names: set[str] = set()

    for img_id, img in images.items():
        img_path = split_dir / img["file_name"]
        if not img_path.exists():
            log.warning("eval.missing_image", path=str(img_path))
            continue
        dets = detector.detect(img_path)

        # Accumulate COCO predictions (category id mapped back BY NAME).
        name2catid = {v: k for k, v in catid2name.items()}
        for d in dets:
            all_detected_names.add(d.name)
            cat_id = name2catid.get(d.name)
            if cat_id is None:
                # Detected a class the GT file doesn't list — skip for mAP scoring.
                continue
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
                    "score": d.score,
                }
            )

        # Per-image localization recall over business-critical classes:
        # a class is "localized" iff a detection of that class overlaps a GT box
        # of the same class at IoU >= threshold.
        localized = _localized_business_critical(
            dets, gt_by_image.get(img_id, []), catid2name, business_critical, iou_thr
        )
        rec = localization_recall(
            localized,
            _present_business_critical(gt_by_image.get(img_id, []), catid2name, business_critical),
        )
        loc_recall_per_image.append({"image_id": img_id, **rec})

    # COCO mAP via pycocotools (lazy import).
    coco_metrics = _coco_eval(coco, predictions, catid2name)

    # Dataset-level localization recall: which business-critical classes were
    # localized in *every* image where they are present (the strict gate view).
    overall_loc = _aggregate_localization_recall(loc_recall_per_image, business_critical)

    result = {
        "split": split,
        "checkpoint": str(ckpt),
        "n_images": len(loc_recall_per_image),
        **coco_metrics,
        "localization_recall": overall_loc,
        "loc_recall_per_image": loc_recall_per_image,
    }
    log.info(
        "eval.done",
        split=split,
        map=result.get("map"),
        map_50=result.get("map_50"),
        loc_recall=overall_loc["recall"],
        missing=overall_loc["missing"],
    )
    return result


# --------------------------------------------------------------------------- #
# Localization recall helpers (pure-Python; reuse common.metrics)
# --------------------------------------------------------------------------- #
def _present_business_critical(
    gts: list[dict[str, Any]],
    catid2name: dict[int, str],
    business_critical: list[str],
) -> list[str]:
    """Business-critical class names that actually appear in this image's GT."""
    present = {catid2name.get(g["category_id"]) for g in gts}
    return [n for n in business_critical if n in present]


def _localized_business_critical(
    dets: list[Any],
    gts: list[dict[str, Any]],
    catid2name: dict[int, str],
    business_critical: list[str],
    iou_thr: float,
) -> list[str]:
    """Names of business-critical classes whose GT box was localized by a detection."""
    bc = set(business_critical)
    localized: set[str] = set()
    for g in gts:
        gname = catid2name.get(g["category_id"])
        if gname not in bc:
            continue
        gbox = tuple(g["bbox"])  # COCO xywh
        for d in dets:
            if d.name != gname:
                continue
            if iou_xywh((d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h), gbox) >= iou_thr:
                localized.add(gname)
                break
    return list(localized)


def _aggregate_localization_recall(
    per_image: list[dict[str, Any]],
    business_critical: list[str],
) -> dict[str, Any]:
    """Strict aggregation: a class passes only if localized in EVERY image it appears in.

    This matches the Phase-4 gate ("localization recall = 1.0 on
    business_critical_fields") — one missed mandatory box on one image fails it.
    """
    # appears[name] = images where the class is present; hit[name] = images where localized.
    appears = dict.fromkeys(business_critical, 0)
    hit = dict.fromkeys(business_critical, 0)
    for rec in per_image:
        per_class: dict[str, bool] = rec.get("per_class", {})  # type: ignore[assignment]
        for name, ok in per_class.items():
            appears[name] += 1
            if ok:
                hit[name] += 1
    per_class_pass = {
        name: (appears[name] > 0 and hit[name] == appears[name]) for name in business_critical
    }
    # Classes never present in the split can't be gated -> excluded from recall.
    gated = [n for n in business_critical if appears[n] > 0]
    n_pass = sum(per_class_pass[n] for n in gated)
    return {
        "recall": (n_pass / len(gated)) if gated else 1.0,
        "per_class": per_class_pass,
        "missing": [n for n in gated if not per_class_pass[n]],
        "not_present_in_split": [n for n in business_critical if appears[n] == 0],
        "appears": appears,
        "hit": hit,
    }


# --------------------------------------------------------------------------- #
# COCO mAP (lazy pycocotools)
# --------------------------------------------------------------------------- #
def _coco_eval(
    coco: dict[str, Any],
    predictions: list[dict[str, Any]],
    catid2name: dict[int, str],
) -> dict[str, Any]:
    """Compute COCO mAP / per-class AP. Returns zeros (with a warning) if no preds."""
    if not predictions:
        log.warning("eval.no_predictions", note="detector returned no boxes; mAP=0")
        return {
            "map": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "per_class_ap": {catid2name[c]: 0.0 for c in catid2name},
        }
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_ML_HINT) from exc

    import tempfile

    # pycocotools loads GT from a file; write a temp copy of this split's COCO.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as gt_fh:
        json.dump(coco, gt_fh)
        gt_path = gt_fh.name

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = ev.stats  # [mAP, mAP50, mAP75, ...]
    per_class_ap = _per_class_ap(ev, coco_gt, catid2name)
    return {
        "map": float(stats[0]),
        "map_50": float(stats[1]),
        "map_75": float(stats[2]),
        "per_class_ap": per_class_ap,
    }


def _per_class_ap(ev: Any, coco_gt: Any, catid2name: dict[int, str]) -> dict[str, float]:
    """Extract per-category AP@[.5:.95] from a COCOeval ``precision`` tensor.

    precision shape: [T(iou), R(recall), K(cat), A(area), M(maxDet)].
    AP per class = mean over IoU & recall at area=all (idx 0), maxDet=last.
    """
    import numpy as np

    precision = ev.eval["precision"]  # type: ignore[index]
    cat_ids = list(coco_gt.getCatIds())
    out: dict[str, float] = {}
    for k, cat_id in enumerate(cat_ids):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        ap = float(np.mean(p)) if p.size else 0.0
        out[catid2name.get(cat_id, str(cat_id))] = ap
    return out
