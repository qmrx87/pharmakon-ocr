"""End-to-end pipeline on the synthetic fixture (stub backend, CPU-only).

For every generated image we run :meth:`VignocrPipeline.extract` with the
deterministic stub backend and assert the assembled :class:`ExtractionRecord`
matches the ground truth exactly:

* money/code/date/identity field values match the ground truth to the
  centime / character (money as Decimal-strings, e.g. ``"262.31"``);
* the ``prix + shp == ppa`` checksum verdict is ``ok``;
* reimbursability colour matches the drawn band;
* the nomenclature anchor matches (identity fields end up trusted).

A final test demonstrates the selling flow's abstention is *stricter* than
receiving on a borderline-confidence read (driven through ``parsing.record.build``,
where the threshold lives, since the stub recognizer always reads at 0.99).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vignocr.common import FieldRead, get_classes, load_config
from vignocr.common.config import get_active_dataset
from vignocr.data.coco import load_split
from vignocr.parsing import record as record_mod
from vignocr.pipeline.orchestrator import VignocrPipeline


@pytest.fixture(scope="module")
def pipeline() -> VignocrPipeline:
    """A stub-backed pipeline (no ML libs, replays fixture ground truth)."""
    return VignocrPipeline({"backend": "stub"})


def _all_image_files(root: Path) -> list[str]:
    files: list[str] = []
    for split in get_active_dataset().get("splits", {}).values():
        sp = load_split(root, split)
        files.extend(img["file_name"] for img in sp.images)
    return files


def _image_path(root: Path, file_name: str) -> Path:
    for split in get_active_dataset().get("splits", {}).values():
        candidate = root / split / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(file_name)


# --------------------------------------------------------------------------- #
# Per-image e2e: every field matches ground truth exactly
# --------------------------------------------------------------------------- #


def test_extract_matches_ground_truth_on_every_image(
    pipeline: VignocrPipeline, synthetic_root: Path, ground_truth: dict
) -> None:
    """Run extract on each synthetic image; all fields match GT to the centime/char."""
    files = _all_image_files(synthetic_root)
    assert files, "no fixture images to test"

    # Iterate over what ground_truth actually carries — that's the test's real
    # intent ("the pipeline extracted everything the synthetic put there"). It
    # adapts naturally when the schema grows with real-data classes (ppa_shp,
    # tr) that synthetic doesn't draw.
    schema = get_classes()
    money_fields = set(schema.role("money_fields") or [])

    for file_name in files:
        gt = ground_truth[file_name]
        record = pipeline.extract(_image_path(synthetic_root, file_name), flow="selling")

        assert record.image_id == file_name

        # Every field the ground_truth lists IS extracted and matches verbatim.
        for name, expected in gt["fields"].items():
            assert name in record.fields, f"{file_name}: missing field {name!r}"
            got = record.fields[name].value
            assert got == expected, (
                f"{file_name}: field {name!r} = {got!r}, expected {expected!r}"
            )

        # Money fields present in this record are centime-quantized Decimal-strings.
        for money_name in money_fields & set(record.fields):
            value = record.fields[money_name].value
            if value is None:  # abstained — separate gate
                continue
            assert "." in value and len(value.split(".")[1]) == 2, (
                f"{file_name}: {money_name} = {value!r} is not a centime Decimal-string"
            )


def test_checksum_ok_on_every_image(
    pipeline: VignocrPipeline, synthetic_root: Path, ground_truth: dict
) -> None:
    """The generator guarantees prix+shp==ppa; the pipeline verdict is ``ok``."""
    from decimal import Decimal

    for file_name in _all_image_files(synthetic_root):
        record = pipeline.extract(_image_path(synthetic_root, file_name), flow="selling")
        assert record.checksum.verdict == "ok", (
            f"{file_name}: checksum verdict {record.checksum.verdict!r}"
        )
        # And the identity actually holds on the reported Decimal-strings.
        prix = Decimal(record.checksum.prix)
        shp = Decimal(record.checksum.shp)
        ppa = Decimal(record.checksum.ppa)
        assert prix + shp == ppa


def test_reimbursability_matches_drawn_band(
    pipeline: VignocrPipeline, synthetic_root: Path, ground_truth: dict
) -> None:
    """The classified band colour equals the colour the generator drew."""
    palette = get_classes().reimbursability["colors"]
    for file_name in _all_image_files(synthetic_root):
        gt_color = ground_truth[file_name]["reimbursability"]
        record = pipeline.extract(_image_path(synthetic_root, file_name), flow="selling")
        assert record.reimbursability.color == gt_color, (
            f"{file_name}: band {record.reimbursability.color!r}, drew {gt_color!r}"
        )
        # Eligibility agrees with the palette mapping (green=True, red=False, orange=None).
        assert record.reimbursability.eligible == palette[gt_color]["eligible"]


def test_nomenclature_anchor_matches(
    pipeline: VignocrPipeline, synthetic_root: Path
) -> None:
    """Every fixture code is a register key, so the anchor always matches."""
    for file_name in _all_image_files(synthetic_root):
        record = pipeline.extract(_image_path(synthetic_root, file_name), flow="selling")
        assert record.nomenclature.matched is True, f"{file_name}: anchor did not match"
        assert record.nomenclature.match_confidence == pytest.approx(1.0)


def test_record_is_serializable_extraction_record(
    pipeline: VignocrPipeline, synthetic_root: Path
) -> None:
    """The pipeline returns a JSON-serializable ExtractionRecord (money as strings)."""
    file_name = _all_image_files(synthetic_root)[0]
    record = pipeline.extract(_image_path(synthetic_root, file_name), flow="selling")
    dumped = record.model_dump(mode="json")
    assert dumped["image_id"] == file_name
    # Money serialized as a string, not a float.
    assert isinstance(dumped["fields"]["ppa"]["value"], str)
    assert isinstance(dumped["checksum"]["ppa"], str)
    assert dumped["model_versions"]["detector"] == "stub"


# --------------------------------------------------------------------------- #
# Abstention: selling is stricter than receiving
# --------------------------------------------------------------------------- #


def test_selling_abstention_is_stricter_than_receiving() -> None:
    """A borderline-confidence read abstains under selling but not under receiving.

    The stub recognizer always reads at 0.99, so we exercise the abstention policy
    where it actually lives — ``parsing.record.build`` — with a confidence between
    the two flows' thresholds (receiving default 0.75 < c < selling default 0.90).
    """
    abst = load_config("parsing/fields")["abstention"]
    selling_tau = float(abst["selling"]["default"])
    receiving_tau = float(abst["receiving"]["default"])
    assert selling_tau > receiving_tau, "config: selling must be stricter than receiving"

    borderline = (selling_tau + receiving_tau) / 2.0  # below selling, above receiving
    fields = {
        "dci": FieldRead(
            name="dci", value="PARACETAMOL", raw="PARACETAMOL",
            confidence=borderline, status="ok", source="ocr",
        )
    }

    selling = record_mod.build(dict(fields), "selling")
    receiving = record_mod.build(dict(fields), "receiving")

    assert "dci" in selling.abstentions, "selling should abstain on a borderline read"
    assert selling.fields["dci"].status == "abstain"

    assert "dci" not in receiving.abstentions, "receiving should accept a borderline read"
    assert receiving.fields["dci"].status == "ok"


def test_high_confidence_never_abstains_in_either_flow() -> None:
    """A confident read (0.99) clears both thresholds — no abstention."""
    fields = {
        "dci": FieldRead(
            name="dci", value="PARACETAMOL", raw="PARACETAMOL",
            confidence=0.99, status="ok", source="ocr",
        )
    }
    for flow in ("selling", "receiving"):
        rec = record_mod.build(dict(fields), flow)  # type: ignore[arg-type]
        assert rec.abstentions == []
        assert rec.fields["dci"].status == "ok"
