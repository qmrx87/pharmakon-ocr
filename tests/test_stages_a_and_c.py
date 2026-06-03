"""Regression tests for the Stage A wiring + the Stage C bootstrap (autolabel).

Locks the contract that future refactors must not break:

* Stage A binding — ``configs/detection/rfdetr_vignette.yaml`` resolves to
  ``data2/`` and a 3-class head ``[date_info, entete, vin]``; ``rfdetr_medium``
  still resolves to ``data/`` and the classes.yaml field schema.
* The COCO loader honours per-dataset aliases when ``dataset=`` is passed
  explicitly (the auto-labeler must NOT depend on the global VIGNOCR_DATA_ACTIVE).
* The auto-labeler produces a Roboflow-importable layout (per-split
  ``labels.csv`` + ``paddle.txt``), shared ``char_dict.txt``, and a
  confidence-sorted ``review_low_conf.csv``. Aliases + ignore are applied so
  ``num_lot`` / ``tr`` appear and ``text`` / ``drug-labels`` do not.
* Pipeline orchestrator: Stage A pre-crop is OFF by default (back-compat).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from vignocr.common import get_dataset, load_config
from vignocr.detection._resolve import resolve_class_schema, resolve_dataset


# --------------------------------------------------------------------------- #
# Stage A / Stage B detector-config binding
# --------------------------------------------------------------------------- #


def test_stage_a_config_binds_data2_and_three_classes() -> None:
    cfg = load_config("detection/rfdetr_vignette")
    ds = resolve_dataset(cfg)
    n, names = resolve_class_schema(cfg, ds)
    assert ds["name"] == "vignette"
    assert ds["root"].endswith("data2") or "data2" in ds["root"]
    assert n == 3
    assert names == ["date_info", "entete", "vin"]


def test_stage_b_config_still_binds_real_and_classes_yaml() -> None:
    cfg = load_config("detection/rfdetr_medium")
    ds = resolve_dataset(cfg)
    n, names = resolve_class_schema(cfg, ds)
    assert ds["name"] == "real"
    # classes.yaml carries the 12 semantic + 3 structural + 2 real-data classes.
    assert n >= 15
    assert {"ppa", "num_enregistrement", "num_lot"}.issubset(set(names))


# --------------------------------------------------------------------------- #
# Auto-labeler — Stage C bootstrap (stub backend; no [ml] required)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def autolabel_output(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the autolabeler on the synthetic fixture once and reuse the artefacts."""
    from vignocr.ocr.autolabel import autolabel

    out = tmp_path_factory.mktemp("ocr_autolabel")
    summary = autolabel(
        dataset_name="synthetic",
        output_dir=out,
        backend="stub",
        splits=["train"],
    )
    return {"summary": summary, "out": out}


def test_autolabel_emits_roboflow_compatible_layout(autolabel_output: dict[str, Any]) -> None:
    out: Path = autolabel_output["out"]
    summary = autolabel_output["summary"]
    assert summary["total_crops"] > 0
    assert (out / "manifest.json").is_file()
    assert (out / "char_dict.txt").is_file()
    assert (out / "review_low_conf.csv").is_file()
    for split in ["train"]:
        sp = out / split
        assert (sp / "labels.csv").is_file()
        assert (sp / "paddle.txt").is_file()
        # Crops directory has at least one JPEG.
        assert any((sp / "crops").iterdir()), f"no crops written under {sp / 'crops'}"


def test_autolabel_labels_csv_has_expected_columns(autolabel_output: dict[str, Any]) -> None:
    csv_path = autolabel_output["out"] / "train" / "labels.csv"
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        first = next(reader, None)
    assert header == ["crop_path", "field", "text", "confidence", "src_image", "src_ann_id"]
    assert first is not None and len(first) == 6


def test_autolabel_paddle_txt_is_tab_separated_path_text(
    autolabel_output: dict[str, Any],
) -> None:
    paddle = autolabel_output["out"] / "train" / "paddle.txt"
    for line in paddle.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        path, _, text = line.partition("\t")
        assert path.startswith("crops/") and path.endswith(".jpg")
        assert text  # non-empty


def test_autolabel_skips_structural_regions(autolabel_output: dict[str, Any]) -> None:
    """Region classes (entete/vin/color_band) are never crop-labeled."""
    csv_path = autolabel_output["out"] / "train" / "labels.csv"
    with csv_path.open(encoding="utf-8", newline="") as fh:
        fields = {row[1] for row in csv.reader(fh) if row and row[0] != "crop_path"}
    assert fields, "labels.csv has no rows"
    assert fields.isdisjoint({"entete", "vin", "color_band", "ppa_shp"}), (
        f"region/combo classes leaked into auto-labels: {fields}"
    )


def test_autolabel_review_low_conf_sorted_ascending(autolabel_output: dict[str, Any]) -> None:
    review = autolabel_output["out"] / "review_low_conf.csv"
    with review.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    confs = [float(r["confidence"]) for r in rows]
    assert confs == sorted(confs), "review queue must be ascending-by-confidence"


def test_autolabel_manifest_summary_shape(autolabel_output: dict[str, Any]) -> None:
    manifest = json.loads(
        (autolabel_output["out"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset"] == "synthetic"
    assert manifest["backend"] == "stub"
    assert manifest["total_crops"] >= 1
    assert "splits" in manifest and "train" in manifest["splits"]


# --------------------------------------------------------------------------- #
# coco loader: aliases / ignore travel with the explicit `dataset` kwarg
# --------------------------------------------------------------------------- #


def test_coco_load_split_applies_explicit_dataset_aliases(tmp_path: Path) -> None:
    """When ``dataset=`` is passed, aliases + ignore must come from THAT block —
    not from the global active dataset (the bug that caused `lot`/`text` to
    leak into the auto-label output before the fix).
    """
    from vignocr.data.coco import load_split

    real = get_dataset("real")
    # Pick the smallest split for speed.
    sp = load_split(real["root"], "test", dataset=real)
    coco_names = {c["name"] for c in sp.categories}
    # `lot` must have been aliased to `num_lot`; `text` must be filtered out.
    assert "lot" not in coco_names, "alias for `lot` -> `num_lot` not applied"
    assert "num_lot" in coco_names, "expected aliased class `num_lot` missing"
    assert "text" not in coco_names, "`text` should be in coco_ignore"
    assert "tarif_ref" not in coco_names, "alias for `tarif_ref` -> `tr` not applied"
    assert "tr" in coco_names, "expected aliased class `tr` missing"


# --------------------------------------------------------------------------- #
# Pipeline orchestrator: Stage A pre-crop is off-by-default
# --------------------------------------------------------------------------- #


def test_pipeline_pre_crop_off_by_default() -> None:
    from vignocr.pipeline.orchestrator import VignocrPipeline

    p = VignocrPipeline()
    assert p.vignette_detector_path is None
    # The defaults still point at the Stage A config so a serving deployment
    # only needs to set vignette_detector_path to enable the stage.
    assert p.vignette_cfg_path == "detection/rfdetr_vignette"
    assert p.vignette_class == "entete"
