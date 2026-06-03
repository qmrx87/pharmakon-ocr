"""Dataset integrity validation, driven by ``configs/data.yaml: integrity``.

Enforces (each gated by its flag):
  * ``assert_no_image_leakage_across_splits``  — no file_name stem in >1 split.
  * ``assert_every_annotation_valid_bbox``     — x>=0, y>=0, w>0, h>0, in image.
  * ``assert_class_names_subset_of_schema``    — every COCO category name exists
    in ``configs/classes.yaml``.
  * ``assert_all_business_critical_present``   — each split covers the
    ``business_critical_fields``. Hard for synthetic; **warn-only** for the real
    dataset (not yet annotated to the schema).

Returns a structured :class:`IntegrityReport`. Raises :class:`IntegrityError`
on hard failures so callers (train/eval) fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vignocr.common import get_active_dataset, get_classes, get_logger
from vignocr.data.coco import load_split

log = get_logger(__name__)


class IntegrityError(RuntimeError):
    """Raised when a dataset fails a hard integrity check."""


@dataclass
class IntegrityReport:
    """Structured outcome of :func:`check_integrity`.

    ``ok`` is False iff any hard error was recorded. Warnings never flip ``ok``.
    """

    root: str
    dataset_name: str
    splits_checked: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_split: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        log.warning("integrity_error", msg=msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        log.info("integrity_warning", msg=msg)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "root": self.root,
            "dataset_name": self.dataset_name,
            "splits_checked": list(self.splits_checked),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "per_split": self.per_split,
        }


def _stem(file_name: str) -> str:
    """File_name stem used for leakage detection (drops the extension only)."""
    return Path(file_name).stem


def check_integrity(root: Path | str) -> IntegrityReport:
    """Validate the dataset at ``root`` against ``data.yaml`` integrity flags.

    Args:
        root: dataset root containing per-split dirs with ``_annotations.coco.json``.

    Returns:
        An :class:`IntegrityReport` (always returned, even on warnings).

    Raises:
        IntegrityError: if any *hard* check fails (after collecting all of them,
            so the message lists every problem found).
    """
    root = Path(root)
    ds = get_active_dataset()
    integrity: dict[str, bool] = ds.get("_integrity", {})
    # Logical-split -> directory-name map for the active dataset (train/valid/test).
    split_dirs: list[str] = list(ds.get("splits", {}).values()) or ["train", "valid", "test"]
    is_real = ds.get("name") == "real"
    bbox_tol = float(ds.get("bbox_clamp_tolerance", 0.0))
    bc_severity = str(ds.get("business_critical_present_severity", "warn" if is_real else "error"))

    # Schema source:
    #   - Stage-bound datasets (vignette has its own class_names) -> use those.
    #   - Default (synthetic / real for fields) -> classes.yaml.
    ds_class_names = ds.get("class_names")
    if ds_class_names:
        schema_names = set(ds_class_names)
        business_critical: set[str] = set()  # not a Stage A concept
    else:
        schema = get_classes()
        schema_names = set(schema.names)
        business_critical = set(schema.business_critical_fields)

    report = IntegrityReport(root=str(root), dataset_name=str(ds.get("name", "")))

    # Load whichever splits actually exist on disk; a missing split is a warning
    # (the real export may not carry all three until annotated).
    loaded = {}
    for split in split_dirs:
        try:
            loaded[split] = load_split(root, split, dataset=ds)
            report.splits_checked.append(split)
        except FileNotFoundError as exc:
            report.add_warning(f"split {split!r} not found: {exc}")

    if not loaded:
        report.add_error(f"no COCO splits found under {root}")
        _maybe_raise(report)
        return report

    # ---- per-split checks ---------------------------------------------------
    stems_by_split: dict[str, set[str]] = {}
    for split, sp in loaded.items():
        stems = {_stem(img["file_name"]) for img in sp.images}
        stems_by_split[split] = stems

        present_classes = {
            sp.cat_id_to_name[int(a["category_id"])]
            for a in sp.annotations
            if int(a["category_id"]) in sp.cat_id_to_name
        }
        report.per_split[split] = {
            "images": len(sp.images),
            "annotations": len(sp.annotations),
            "categories": [c["name"] for c in sp.categories],
            "present_field_classes": sorted(present_classes),
        }

        # (a) class names subset of schema
        if integrity.get("assert_class_names_subset_of_schema", True):
            unknown = {c["name"] for c in sp.categories} - schema_names
            if unknown:
                report.add_error(
                    f"[{split}] COCO category names not in classes.yaml: {sorted(unknown)}"
                )

        # (b) every annotation has a valid bbox within its image (sub-pixel
        # tolerance from data.yaml clamps Roboflow's float-precision artifacts).
        if integrity.get("assert_every_annotation_valid_bbox", True):
            dims = {
                int(img["id"]): (float(img["width"]), float(img["height"])) for img in sp.images
            }
            bad = _invalid_boxes(sp.annotations, dims, tolerance=bbox_tol)
            if bad:
                sample = bad[:5]
                report.add_error(f"[{split}] {len(bad)} invalid bbox(es); first few: {sample}")

        # (d) business-critical coverage. Severity is configurable per dataset
        # (data.yaml: business_critical_present_severity = error|warn|skip) so
        # the real annotation can flag prix/shp as warnings (they are combined
        # into ppa_shp) while synthetic still hard-fails on a missing schema class.
        if integrity.get("assert_all_business_critical_present", True) and bc_severity != "skip":
            missing = sorted(business_critical - present_classes)
            if missing:
                msg = f"[{split}] missing business-critical field classes: {missing}"
                if bc_severity == "warn":
                    suffix = " (combined/aliased in this annotation — see data.yaml)"
                    report.add_warning(msg + suffix)
                else:
                    report.add_error(msg)

    # ---- cross-split leakage ------------------------------------------------
    if integrity.get("assert_no_image_leakage_across_splits", True):
        splits = list(stems_by_split)
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                a, b = splits[i], splits[j]
                shared = stems_by_split[a] & stems_by_split[b]
                if shared:
                    sample = sorted(shared)[:5]
                    report.add_error(
                        f"image leakage between {a!r} and {b!r}: "
                        f"{len(shared)} shared stem(s); first few: {sample}"
                    )

    _maybe_raise(report)
    return report


def _invalid_boxes(
    annotations: list[dict[str, Any]],
    dims_by_image: dict[int, tuple[float, float]],
    *,
    tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Return annotations whose bbox is degenerate or out of image bounds.

    ``tolerance`` (pixels) clamps Roboflow sub-pixel artifacts (e.g. ``y=-0.01``)
    silently — only boxes still invalid AFTER the clamp are reported.
    """
    bad: list[dict[str, Any]] = []
    for ann in annotations:
        try:
            x, y, w, h = (float(v) for v in ann["bbox"])
        except (KeyError, TypeError, ValueError):
            bad.append({"id": ann.get("id"), "reason": "missing/invalid bbox"})
            continue
        iw, ih = dims_by_image.get(int(ann["image_id"]), (None, None))
        # Clamp slightly-OOB origin/extent within tolerance (sub-pixel Roboflow noise).
        if tolerance > 0:
            if -tolerance <= x < 0:
                w += x
                x = 0.0
            if -tolerance <= y < 0:
                h += y
                y = 0.0
            if iw is not None and 0 < (x + w) - iw <= tolerance:
                w = iw - x
            if ih is not None and 0 < (y + h) - ih <= tolerance:
                h = ih - y
        reason = None
        if not (x >= 0 and y >= 0 and w > 0 and h > 0):
            reason = "non-positive size or negative origin"
        elif iw is not None and (x + w > iw + 1e-6 or y + h > ih + 1e-6):
            reason = "exceeds image bounds"
        if reason:
            bad.append(
                {
                    "id": ann.get("id"),
                    "image_id": ann.get("image_id"),
                    "bbox": ann.get("bbox"),
                    "reason": reason,
                }
            )
    return bad


def _maybe_raise(report: IntegrityReport) -> None:
    if report.errors:
        joined = "\n  - ".join(report.errors)
        raise IntegrityError(f"dataset integrity check failed for {report.root!r}:\n  - {joined}")


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    import json as _json

    _ds = get_active_dataset()
    try:
        _report = check_integrity(_ds["root"])
        print(_json.dumps(_report.as_dict(), ensure_ascii=False, indent=2))
    except IntegrityError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
