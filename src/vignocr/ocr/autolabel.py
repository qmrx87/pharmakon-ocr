"""Auto-label every detected field crop with a SOTA OCR (PaddleOCR).

Bootstrap path for **Stage C** (text recognition): we have field-level bounding
boxes from Stage B, but NO labelled (crop, text) pairs to train a recognizer.
This script crops every annotated field, runs a pretrained multilingual OCR on
each crop, and writes a labelled dataset in two formats:

* **Roboflow-importable** — per-split ``labels.csv`` (``crop_path, field, text,
  confidence, src_image, src_ann_id``). Confidence-sorted ascending so a
  reviewer corrects the WORST predictions first.
* **PaddleOCR fine-tune format** — per-split ``paddle.txt`` (``crop_path\\ttext``)
  and a shared ``char_dict.txt`` (the union of characters), consumed directly by
  ``vignocr.ocr.finetune.run`` once the labels are corrected.

The OCR backend is **lazy** (no ``paddleocr`` import at module top); a
deterministic ``stub`` backend lets the unit tests exercise the full pipeline on
CPU without the ``[ml]`` extra.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vignocr.common import (
    get_classes,
    get_dataset,
    get_logger,
    load_config,
    seed_everything,
)
from vignocr.data.coco import crops_for_image, load_split

log = get_logger(__name__)

_ML_HINT = "Auto-labelling needs the ML extra (paddleocr). Run: pip install -e .[ml]"

# Class types that aren't OCR-text (regions / combo) — skipped by default.
_NON_OCR_TYPES = {"region", "money_combo"}


# --------------------------------------------------------------------------- #
# Backends (PaddleOCR + a deterministic stub for CPU/tests)
# --------------------------------------------------------------------------- #


class OcrBackend(Protocol):
    """Per-crop OCR backend interface."""

    name: str

    def read(self, crop_path: Path) -> tuple[str, float]:  # (text, confidence)
        ...


@dataclass
class StubBackend:
    """Deterministic backend used for tests + CPU smoke runs.

    Returns a predictable ``"AUTO_<field>"`` string at confidence 0.50 so the
    full auto-label pipeline (cropping, file layout, CSV writing, manifest) is
    exercised without needing the ``[ml]`` extra. The field name is encoded in
    the crop's filename prefix (``<n>_<field>.jpg``).
    """

    name: str = "stub"

    def read(self, crop_path: Path) -> tuple[str, float]:
        # Crops are named "<index>_<field>.jpg"; recover the field for a useful label.
        stem = crop_path.stem
        field = stem.split("_", 1)[1] if "_" in stem else stem
        return f"AUTO_{field.upper()}", 0.50


class PaddleBackend:
    """Real backend (PaddleOCR ``ocr.OCR``). Lazily constructed."""

    name = "paddleocr"

    def __init__(self, lang: str = "fr", use_gpu: bool | None = None) -> None:
        try:
            from paddleocr import PaddleOCR  # noqa: WPS433 — lazy by design
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_ML_HINT) from exc
        kwargs: dict[str, Any] = {"lang": lang, "show_log": False}
        if use_gpu is not None:
            kwargs["use_gpu"] = use_gpu
        # det=False because the crop IS the field box; we only need recognition.
        self._ocr = PaddleOCR(use_angle_cls=True, det=False, **kwargs)

    def read(self, crop_path: Path) -> tuple[str, float]:
        # PaddleOCR returns [[ (text, score) ]] for the per-image recognition call.
        result = self._ocr.ocr(str(crop_path), cls=True, det=False, rec=True)
        if not result or not result[0]:
            return "", 0.0
        # When det=False each call returns one (text, score) per crop.
        first = result[0]
        if isinstance(first, list) and first and isinstance(first[0], tuple):
            text, score = first[0]
        else:
            text, score = first
        return str(text), float(score)


def _build_backend(name: str) -> OcrBackend:
    if name == "stub":
        return StubBackend()
    if name == "paddleocr":
        return PaddleBackend()
    raise ValueError(f"unknown OCR backend: {name!r} (use 'paddleocr' or 'stub')")


# --------------------------------------------------------------------------- #
# Orientation-aware cropping
# --------------------------------------------------------------------------- #


def _orient_crop(image: Any, *, is_vertical: bool) -> Any:
    """Rotate vertical-field crops 90° upright before OCR (PIL only — no ML)."""
    if not is_vertical:
        return image
    return image.rotate(-90, expand=True)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def autolabel(
    *,
    dataset_name: str = "real",
    output_dir: Path | str = "ocr_dataset",
    backend: str = "paddleocr",
    splits: Iterable[str] | None = None,
    skip_classes: set[str] | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    """Run OCR over every annotated field crop in ``dataset_name`` and emit a
    Roboflow + PaddleOCR-importable labelled dataset under ``output_dir``.

    Args:
        dataset_name: which ``data.yaml`` block to crop from (typically the
            Stage B field dataset — ``real``).
        output_dir: target directory (created if missing).
        backend: ``"paddleocr"`` (real) or ``"stub"`` (deterministic test backend).
        splits: which splits to process (default: every split declared by the
            dataset block).
        skip_classes: extra class names to skip on top of the default region /
            money-combo skip list.
        seed: deterministic seed for any RNG inside backends.

    Returns:
        A summary dict ``{splits: {<split>: {images, crops, mean_conf}},
        total_crops, low_conf, output_dir}``.
    """
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = get_classes()
    ocr_cfg = load_config("ocr/recognition")
    rotated = set(schema.role("rotated_fields") or [])
    skip = set(skip_classes or set())
    # Default skip set: structural regions + combo boxes that aren't a single text run.
    for c in schema.classes:
        if c.get("type") in _NON_OCR_TYPES:
            skip.add(c["name"])

    ds = get_dataset(dataset_name)
    split_dirs = list(ds.get("splits", {}).values()) or list(splits or [])
    if splits:
        split_dirs = list(splits)

    log.info(
        "autolabel.start",
        dataset=dataset_name,
        backend=backend,
        output=str(output_dir),
        splits=split_dirs,
        skip_classes=sorted(skip),
        rotated_fields=sorted(rotated),
    )

    be = _build_backend(backend)
    char_set: set[str] = set()
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "backend": be.name,
        "ocr_cfg_lang": ocr_cfg.get("backend", "paddleocr"),
        "splits": {},
        "total_crops": 0,
        "low_conf_threshold": 0.75,
        "low_conf": 0,
        "output_dir": str(output_dir.resolve()),
    }

    review_rows: list[dict[str, Any]] = []

    for split in split_dirs:
        sp_root = output_dir / split
        crops_root = sp_root / "crops"
        crops_root.mkdir(parents=True, exist_ok=True)
        labels_csv = sp_root / "labels.csv"
        paddle_txt = sp_root / "paddle.txt"

        n_imgs = n_crops = 0
        conf_sum = 0.0
        with (
            labels_csv.open("w", encoding="utf-8", newline="") as cfh,
            paddle_txt.open("w", encoding="utf-8") as pfh,
        ):
            writer = csv.writer(cfh)
            writer.writerow(["crop_path", "field", "text", "confidence", "src_image", "src_ann_id"])

            try:
                coco = load_split(ds["root"], split, dataset=ds)
            except FileNotFoundError as exc:
                log.warning("autolabel.split_missing", split=split, err=str(exc))
                summary["splits"][split] = {"images": 0, "crops": 0, "mean_conf": 0.0}
                continue

            for img in coco.images:
                anns = coco.annotations_for(int(img["id"]))
                if not anns:
                    continue
                img_path = coco.image_path(img)
                if not img_path.exists():
                    log.warning("autolabel.image_missing", path=str(img_path))
                    continue
                by_name = crops_for_image(img_path, anns, coco.cat_id_to_name)
                n_imgs += 1

                for field_name, crops in by_name.items():
                    if field_name in skip:
                        continue
                    is_vertical = field_name in rotated
                    for ci, crop in enumerate(crops):
                        # Find the source annotation id — the FIRST annotation
                        # with this category that hasn't been used yet is good
                        # enough (per-image groupings are small).
                        ann_id = _match_ann_id(anns, coco.cat_id_to_name, field_name, ci)
                        crop_name = f"{img['id']:06d}_{ann_id}_{field_name}.jpg"
                        out_path = crops_root / crop_name
                        oriented = _orient_crop(crop.image, is_vertical=is_vertical)
                        oriented.save(out_path, "JPEG", quality=92)

                        text, conf = be.read(out_path)
                        char_set.update(text)
                        rel = f"crops/{crop_name}"
                        writer.writerow(
                            [rel, field_name, text, f"{conf:.4f}", img["file_name"], ann_id]
                        )
                        pfh.write(f"{rel}\t{text}\n")
                        n_crops += 1
                        conf_sum += conf
                        if conf < summary["low_conf_threshold"]:
                            summary["low_conf"] += 1
                            review_rows.append(
                                {
                                    "split": split,
                                    "crop_path": str((sp_root / rel).as_posix()),
                                    "field": field_name,
                                    "text": text,
                                    "confidence": round(conf, 4),
                                    "src_image": img["file_name"],
                                    "src_ann_id": ann_id,
                                }
                            )

        mean_conf = (conf_sum / n_crops) if n_crops else 0.0
        summary["splits"][split] = {"images": n_imgs, "crops": n_crops, "mean_conf": round(mean_conf, 4)}
        summary["total_crops"] += n_crops
        log.info(
            "autolabel.split_done",
            split=split,
            images=n_imgs,
            crops=n_crops,
            mean_conf=round(mean_conf, 4),
        )

    # Char dictionary (PaddleOCR rec vocab).
    char_dict = output_dir / "char_dict.txt"
    char_dict.write_text("\n".join(sorted(c for c in char_set if c != "\n")), encoding="utf-8")

    # Confidence-sorted review queue.
    review_rows.sort(key=lambda r: r["confidence"])
    review_csv = output_dir / "review_low_conf.csv"
    with review_csv.open("w", encoding="utf-8", newline="") as fh:
        if review_rows:
            writer = csv.DictWriter(fh, fieldnames=list(review_rows[0].keys()))
            writer.writeheader()
            writer.writerows(review_rows)
        else:
            fh.write("# no low-confidence predictions\n")

    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info(
        "autolabel.done",
        total_crops=summary["total_crops"],
        low_conf=summary["low_conf"],
        output=str(output_dir),
    )
    return summary


def _match_ann_id(
    anns: list[dict[str, Any]],
    cat_id_to_name: dict[int, str],
    field_name: str,
    nth: int,
) -> int:
    """The id of the ``nth`` annotation whose category resolves to ``field_name``."""
    seen = 0
    for ann in anns:
        if cat_id_to_name.get(int(ann["category_id"])) == field_name:
            if seen == nth:
                return int(ann.get("id", -1))
            seen += 1
    return -1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        default="real",
        help="data.yaml block to crop from (default: real — the Stage B fields).",
    )
    p.add_argument(
        "--output",
        default="ocr_dataset",
        help="Output directory for the labelled dataset.",
    )
    p.add_argument(
        "--backend",
        default="paddleocr",
        choices=("paddleocr", "stub"),
        help="OCR backend. 'stub' is deterministic and CPU-only (no [ml]).",
    )
    p.add_argument("--splits", nargs="*", default=None, help="Subset of splits to process.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = p.parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    summary = autolabel(
        dataset_name=args.dataset,
        output_dir=args.output,
        backend=args.backend,
        splits=args.splits,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
