"""Dataset integrity — the synthetic fixture must satisfy ``data.yaml: integrity``.

These checks mirror the invariants the real Roboflow export will be held to once
annotated to the 15-class schema:

* no image leakage across splits (by ``file_name`` stem),
* every annotation carries a valid, in-bounds bbox,
* every COCO category name exists in ``configs/classes.yaml``,
* each split covers the ``business_critical_fields``.

We exercise both the high-level :func:`vignocr.data.validate.check_integrity`
(which raises on a hard failure) and the raw COCO splits, so a regression in
either the generator or the validator is caught.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vignocr.common import get_active_dataset, get_classes
from vignocr.data import stats
from vignocr.data.coco import load_split
from vignocr.data.validate import IntegrityReport, check_integrity


def _split_dirs() -> list[str]:
    """On-disk split directory names for the active dataset (train/valid/test)."""
    return list(get_active_dataset().get("splits", {}).values())


def test_check_integrity_passes_clean(synthetic_root: Path) -> None:
    """The generated dataset passes every configured integrity check."""
    report = check_integrity(synthetic_root)
    assert isinstance(report, IntegrityReport)
    assert report.ok, f"integrity errors: {report.errors}"
    assert report.errors == []
    # All three splits should have been found and checked.
    assert set(report.splits_checked) == set(_split_dirs())


def test_no_image_leakage_across_splits(synthetic_root: Path) -> None:
    """No image (by file_name stem) appears in more than one split."""
    stems_by_split: dict[str, set[str]] = {}
    for split in _split_dirs():
        sp = load_split(synthetic_root, split)
        stems_by_split[split] = {Path(img["file_name"]).stem for img in sp.images}

    splits = list(stems_by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            shared = stems_by_split[a] & stems_by_split[b]
            assert not shared, f"image leakage between {a!r} and {b!r}: {sorted(shared)}"


def test_all_annotations_have_valid_bboxes(synthetic_root: Path) -> None:
    """Every annotation has x>=0, y>=0, w>0, h>0 and stays within its image."""
    for split in _split_dirs():
        sp = load_split(synthetic_root, split)
        dims = {int(img["id"]): (float(img["width"]), float(img["height"])) for img in sp.images}
        assert sp.annotations, f"split {split!r} has no annotations"
        for ann in sp.annotations:
            x, y, w, h = (float(v) for v in ann["bbox"])
            assert x >= 0 and y >= 0, f"[{split}] negative origin in {ann!r}"
            assert w > 0 and h > 0, f"[{split}] non-positive size in {ann!r}"
            iw, ih = dims[int(ann["image_id"])]
            assert x + w <= iw + 1e-6, f"[{split}] bbox exceeds width in {ann!r}"
            assert y + h <= ih + 1e-6, f"[{split}] bbox exceeds height in {ann!r}"


def test_category_names_subset_of_schema(synthetic_root: Path) -> None:
    """Every COCO category name is one of the classes.yaml class names."""
    schema_names = set(get_classes().names)
    # Guard against accidental truncation of classes.yaml: the 12 semantic + 3
    # structural classes are non-negotiable; real-data extensions (ppa_shp, tr)
    # may push the total higher.
    assert len(schema_names) >= 15, "schema must define at least the 15 core classes"
    for split in _split_dirs():
        sp = load_split(synthetic_root, split)
        coco_names = {c["name"] for c in sp.categories}
        unknown = coco_names - schema_names
        assert not unknown, f"[{split}] categories not in classes.yaml: {sorted(unknown)}"


def test_business_critical_fields_present_each_split(synthetic_root: Path) -> None:
    """Each split covers every business-critical field class (ppa/prix/shp/...)."""
    business_critical = set(get_classes().business_critical_fields)
    assert business_critical, "classes.yaml must declare business_critical_fields"
    for split in _split_dirs():
        sp = load_split(synthetic_root, split)
        present = {
            sp.cat_id_to_name[int(a["category_id"])]
            for a in sp.annotations
            if int(a["category_id"]) in sp.cat_id_to_name
        }
        missing = business_critical - present
        assert not missing, f"[{split}] missing business-critical classes: {sorted(missing)}"


def test_category_ids_match_schema_ids(synthetic_root: Path) -> None:
    """The COCO ``categories`` array agrees with the schema's id<->name mapping.

    Loading is by *name* (robust to the real export's ids), but the synthetic
    generator emits the schema ids — verify they line up so the fixture is a
    faithful stand-in for a correctly-annotated export.
    """
    schema = get_classes()
    for split in _split_dirs():
        sp = load_split(synthetic_root, split)
        for cat in sp.categories:
            assert schema.id_of(cat["name"]) == int(cat["id"]), (
                f"[{split}] category id mismatch for {cat['name']!r}: "
                f"COCO={cat['id']} schema={schema.id_of(cat['name'])}"
            )


def test_stats_summary_counts_are_coherent(synthetic_root: Path, data_config: dict) -> None:
    """``stats.summarize`` reports the expected image totals and per-class coverage."""
    summary = stats.summarize(synthetic_root)

    expected_total = sum(int(v) for v in data_config["num_images"].values())
    assert summary["totals"]["images"] == expected_total
    assert summary["missing_splits"] == []

    # Every business-critical field is counted across the dataset.
    per_class = summary["per_class"]
    for name in get_classes().business_critical_fields:
        assert per_class.get(name, 0) >= 1, f"{name!r} not represented in stats"


def test_ground_truth_keys_cover_all_images(synthetic_root: Path, ground_truth: dict) -> None:
    """Ground truth is keyed by every image file_name across the splits."""
    image_files: set[str] = set()
    for split in _split_dirs():
        sp = load_split(synthetic_root, split)
        image_files.update(img["file_name"] for img in sp.images)
    assert image_files, "no images found in the fixture"
    assert image_files <= set(ground_truth), (
        "ground_truth.json is missing entries for: "
        f"{sorted(image_files - set(ground_truth))}"
    )


def test_check_integrity_raises_on_corrupted_copy(synthetic_root: Path, tmp_path: Path) -> None:
    """A deliberately corrupted COCO (out-of-bounds bbox) trips a hard failure.

    Proves the integrity gate actually fails closed rather than rubber-stamping —
    we never weaken it. We copy the fixture, inject a bad bbox, and assert the
    validator raises :class:`IntegrityError`.
    """
    import json
    import shutil

    from vignocr.data.validate import IntegrityError

    dst = tmp_path / "corrupt"
    shutil.copytree(synthetic_root, dst)

    # Corrupt the first annotation of the train split: blow the bbox past bounds.
    train_dir = dst / get_active_dataset()["splits"]["train"]
    coco_path = train_dir / "_annotations.coco.json"
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    assert data["annotations"], "fixture train split unexpectedly empty"
    data["annotations"][0]["bbox"] = [10.0, 10.0, 999999.0, 999999.0]
    coco_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(IntegrityError):
        check_integrity(dst)
