#!/usr/bin/env python
"""Map COCO field annotations -> the JSON dataset the v2a VLM (Donut) trains on.

THE PROBLEM THIS SOLVES
    The VLM needs (vignette image, target JSON) pairs, but the Stage B export
    (``data/``) carries field BOXES, not transcriptions. This script bridges the
    gap: for every annotated image it crops each field box, obtains the field's
    TEXT VALUE, and assembles the flat target JSON.

VALUE SOURCES (per crop, first hit wins)
    1. A REVIEWED autolabel CSV (``<autolabel_dir>/<split>/labels.csv``, the
       stage-04 output after human correction in Roboflow) — keyed by
       (src_image, src_ann_id). Human-reviewed text always beats fresh OCR.
    2. Fresh OCR via the configured backend:
         * ``doctr`` — pretrained docTR recognition (PARSeq) on the crop
           (vertical fields are rotated upright first, same convention as
           ocr.autolabel). Requires the [v2] extra + prefetched weights.
         * ``stub``  — deterministic ``AUTO_<FIELD>`` values (CPU tests).
    Values below ``dataset.min_value_conf`` are OMITTED from the target (the
    VLM must not be taught noise) and queued in ``review_values.csv``.

OUTPUT LAYOUT (<out>/, HF-imagefolder + Donut conventions)
    <out>/<split>/<image>.jpg          the FULL vignette image (copied)
    <out>/<split>/metadata.jsonl       {"file_name", "ground_truth":
                                        {"gt_parse": {field: value, ...}}}
    <out>/review_values.csv            low-confidence values, ascending conf
    <out>/manifest.json                builder provenance + per-split stats

USAGE
    python scripts/build_vlm_dataset.py --config v2/vlm_donut --output ocr_dataset_vlm
    python scripts/build_vlm_dataset.py --backend stub          # CPU smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from vignocr.common import get_classes, get_dataset, get_logger, load_config, seed_everything
from vignocr.data.coco import crops_for_image, load_split
from vignocr.v2.donut_format import json2token  # noqa: F401  (re-export convenience)

log = get_logger(__name__)

_V2_HINT = "The doctr backend needs the v2 extra. Run: pip install -e .[ml,v2]"


# --------------------------------------------------------------------------- #
# Crop-value backends
# --------------------------------------------------------------------------- #
class StubValueBackend:
    """Deterministic values for CPU tests: ``AUTO_<FIELD>`` at conf 0.50."""

    name = "stub"

    def read(self, crop: Any, field: str) -> tuple[str, float]:
        return f"AUTO_{field.upper()}", 0.50


class DoctrValueBackend:
    """Pretrained docTR recognition (PARSeq) on a single field crop."""

    name = "doctr"

    def __init__(self) -> None:
        try:
            import numpy as np  # noqa: F401
            from doctr.models import recognition_predictor
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_V2_HINT) from exc
        # Recognition-only: the COCO box already localized the field.
        self._rec = recognition_predictor("parseq", pretrained=True)

    def read(self, crop: Any, field: str) -> tuple[str, float]:
        import numpy as np

        out = self._rec([np.asarray(crop.convert("RGB"))])
        if not out:
            return "", 0.0
        value, conf = out[0]
        return str(value), float(conf)


def _build_backend(name: str) -> Any:
    if name == "stub":
        return StubValueBackend()
    if name == "doctr":
        return DoctrValueBackend()
    raise ValueError(f"unknown value backend {name!r} (use 'doctr' or 'stub')")


# --------------------------------------------------------------------------- #
# Reviewed-autolabel index: (src_image, src_ann_id) -> (text, conf)
# --------------------------------------------------------------------------- #
def _load_reviewed(autolabel_dir: Path, split: str) -> dict[tuple[str, int], tuple[str, float]]:
    csv_path = autolabel_dir / split / "labels.csv"
    if not csv_path.is_file():
        return {}
    out: dict[tuple[str, int], tuple[str, float]] = {}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                key = (row["src_image"], int(row["src_ann_id"]))
                out[key] = (row["text"], float(row.get("confidence", 1.0)))
            except (KeyError, ValueError):
                continue
    log.info("vlm_dataset.reviewed_loaded", split=split, entries=len(out))
    return out


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build(
    *,
    cfg_path: str = "v2/vlm_donut",
    output_dir: Path | str | None = None,
    backend: str | None = None,
    splits: list[str] | None = None,
    source_dataset: str | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    """Build the VLM dataset; returns the manifest dict (also written to disk).

    ``source_dataset`` overrides the config's ``dataset.source_dataset`` (used
    by the CPU test-suite to build from the synthetic fixture).
    """
    seed_everything(seed)
    cfg = load_config(cfg_path)
    dcfg = cfg.get("dataset", {})
    fields: list[str] = list(cfg.get("fields", []))
    min_conf = float(dcfg.get("min_value_conf", 0.30))

    ds = get_dataset(str(source_dataset or dcfg.get("source_dataset", "real")))
    out_root = Path(output_dir or dcfg.get("dir", "ocr_dataset_vlm"))
    out_root.mkdir(parents=True, exist_ok=True)
    autolabel_dir = Path(dcfg.get("autolabel_dir", "ocr_dataset"))
    be = _build_backend(backend or str(dcfg.get("ocr_backend", "doctr")))

    schema = get_classes()
    rotated = set(schema.role("rotated_fields") or [])

    split_dirs = splits or list(ds.get("splits", {}).values())
    manifest: dict[str, Any] = {
        "cfg_path": cfg_path,
        "source_dataset": ds.get("name"),
        "value_backend": be.name,
        "fields": fields,
        "min_value_conf": min_conf,
        "splits": {},
        "output_dir": str(out_root.resolve()),
    }
    review_rows: list[dict[str, Any]] = []

    for split in split_dirs:
        sp_out = out_root / split
        sp_out.mkdir(parents=True, exist_ok=True)
        meta_path = sp_out / "metadata.jsonl"
        reviewed = _load_reviewed(autolabel_dir, split)

        try:
            coco = load_split(ds["root"], split, dataset=ds)
        except FileNotFoundError as exc:
            log.warning("vlm_dataset.split_missing", split=split, err=str(exc))
            manifest["splits"][split] = {"images": 0, "skipped": 0}
            continue

        n_imgs = n_skipped = n_fields = 0
        with meta_path.open("w", encoding="utf-8") as mfh:
            for img in coco.images:
                anns = coco.annotations_for(int(img["id"]))
                if not anns:
                    continue
                img_path = coco.image_path(img)
                if not img_path.exists():
                    log.warning("vlm_dataset.image_missing", path=str(img_path))
                    continue

                by_name = crops_for_image(img_path, anns, coco.cat_id_to_name)
                values: dict[str, str] = {}
                for field in fields:
                    crops = by_name.get(field) or []
                    if not crops:
                        continue
                    # One value per field: the FIRST annotation (vignettes carry
                    # each field at most once; duplicates are annotation noise).
                    crop = crops[0]
                    ann_id = _ann_id_for(anns, coco.cat_id_to_name, field, 0)
                    src = reviewed.get((img["file_name"], ann_id))
                    if src is not None:
                        text, conf, origin = src[0], src[1], "reviewed"
                    else:
                        pil = crop.image
                        if field in rotated:
                            pil = pil.rotate(-90, expand=True)
                        text, conf = be.read(pil, field)
                        origin = be.name
                    if not text.strip():
                        continue
                    if conf < min_conf:
                        review_rows.append({
                            "split": split, "image": img["file_name"], "field": field,
                            "text": text, "confidence": round(conf, 4), "origin": origin,
                        })
                        continue
                    values[field] = text.strip()

                if not values:
                    n_skipped += 1
                    continue
                # Copy the FULL vignette image next to its metadata row.
                dest = sp_out / img_path.name
                if not dest.exists():
                    shutil.copy2(img_path, dest)
                mfh.write(json.dumps(
                    {"file_name": img_path.name,
                     "ground_truth": {"gt_parse": values}},
                    ensure_ascii=False) + "\n")
                n_imgs += 1
                n_fields += len(values)

        manifest["splits"][split] = {
            "images": n_imgs,
            "skipped": n_skipped,
            "mean_fields_per_image": round(n_fields / n_imgs, 2) if n_imgs else 0.0,
        }
        log.info("vlm_dataset.split_done", split=split, **manifest["splits"][split])

    review_rows.sort(key=lambda r: r["confidence"])
    with (out_root / "review_values.csv").open("w", encoding="utf-8", newline="") as fh:
        if review_rows:
            w = csv.DictWriter(fh, fieldnames=list(review_rows[0].keys()))
            w.writeheader()
            w.writerows(review_rows)
        else:
            fh.write("# no low-confidence values\n")

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("vlm_dataset.done", output=str(out_root), low_conf=len(review_rows))
    return manifest


def _ann_id_for(
    anns: list[dict[str, Any]], cat_id_to_name: dict[int, str], field: str, nth: int
) -> int:
    seen = 0
    for ann in anns:
        if cat_id_to_name.get(int(ann["category_id"])) == field:
            if seen == nth:
                return int(ann.get("id", -1))
            seen += 1
    return -1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="v2/vlm_donut", help="config name under configs/")
    p.add_argument("--output", default=None, help="output dir (default: cfg dataset.dir)")
    p.add_argument("--backend", default=None, choices=("doctr", "stub"),
                   help="value backend override (default: cfg dataset.ocr_backend)")
    p.add_argument("--splits", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)
    manifest = build(
        cfg_path=args.config, output_dir=args.output,
        backend=args.backend, splits=args.splits, seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
