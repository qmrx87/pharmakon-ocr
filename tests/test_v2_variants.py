"""CPU-only regression tests for the v2 challenger plumbing.

Covers the pure-Python pieces (no torch / transformers / doctr needed):
  * Donut token format roundtrip + tolerance to malformed generations.
  * Type-aware normalizers (date / code / money / money-pair).
  * Full-page layout parser (synthetic words -> field assignment, including
    the date_fab-before-date_exp chronology rule and the rotated-strip pass).
  * VLM dataset builder (stub value backend, synthetic fixture source).
  * Orchestrator variant resolution (env > cfg > v1) + version reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vignocr.v2.donut_format import json2token, special_tokens_for, token2json
from vignocr.v2.fullpage import Word, assign_fields
from vignocr.v2.normalize import normalize_value

# --------------------------------------------------------------------------- #
# Donut token format
# --------------------------------------------------------------------------- #


def test_json2token_roundtrip_with_order() -> None:
    values = {"num_lot": "B1234", "date_exp": "05/2027", "date_fab": "05/2025"}
    seq = json2token(values, ["num_lot", "date_fab", "date_exp"])
    assert seq == (
        "<s_num_lot>B1234</s_num_lot>"
        "<s_date_fab>05/2025</s_date_fab>"
        "<s_date_exp>05/2027</s_date_exp>"
    )
    assert token2json(seq) == values


def test_json2token_skips_empty_and_none() -> None:
    seq = json2token({"a": "", "b": None, "c": "x"})  # type: ignore[dict-item]
    assert seq == "<s_c>x</s_c>"


def test_token2json_tolerates_debris_and_partial_generation() -> None:
    seq = "<s_vignocr><s_num_lot>B1</s_num_lot><s_date_exp>05/27"  # truncated
    assert token2json(seq) == {"num_lot": "B1"}
    assert token2json("") == {}
    assert token2json("</s><pad>") == {}


def test_special_tokens_cover_every_field() -> None:
    toks = special_tokens_for(["num_lot", "ppa"], "<s_vignocr>")
    assert toks == ["<s_vignocr>", "<s_num_lot>", "</s_num_lot>", "<s_ppa>", "</s_ppa>"]


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "a", "b"),
    [
        ("date_exp", "05/2027", "5-2027"),
        ("date_fab", " 01/05/2025", "1.5.2025"),
        ("num_lot", "B-1234 ", "b1234"),
        ("num_enregistrement", "20/05B/123", "20 05B 123"),
        ("ppa", "132,21 DA", "132.21"),
        ("ppa_shp", "248.50 + 1,50", "248,50+1.50"),
        ("product_name", "  DOLIPRANE  1000 ", "doliprane 1000"),
    ],
)
def test_normalize_equates_harmless_variants(field: str, a: str, b: str) -> None:
    assert normalize_value(field, a) == normalize_value(field, b)


def test_normalize_keeps_real_differences() -> None:
    assert normalize_value("num_lot", "B1234") != normalize_value("num_lot", "B1235")
    assert normalize_value("ppa", "132.21") != normalize_value("ppa", "132.12")


# --------------------------------------------------------------------------- #
# Full-page layout parser
# --------------------------------------------------------------------------- #


def _strip_words() -> list[Word]:
    """A synthetic vignette: rotated strip carries lot + 2 dates; horizontal
    pass carries the registration code and two amounts."""
    return [
        # rotated (vertical strip) pass — dates deliberately in REVERSE order:
        Word("11/2027", 0.95, 0.90, 0.30, True),   # expiry (later)
        Word("11/2025", 0.94, 0.90, 0.55, True),   # manufacture (earlier)
        Word("B4521X", 0.91, 0.90, 0.80, True),    # lot, near the dates
        # horizontal pass:
        Word("DOLIPRANE", 0.99, 0.30, 0.20, False),
        Word("20/05B/123/2024", 0.88, 0.40, 0.85, False),  # num_enregistrement
        Word("248,50", 0.92, 0.30, 0.60, False),
        Word("1,50", 0.90, 0.55, 0.60, False),
    ]


def _pcfg() -> dict[str, Any]:
    from vignocr.common import load_config

    return load_config("v2/fullpage_doctr").get("parser", {})


def test_assign_fields_dates_follow_chronology_not_reading_order() -> None:
    fields = assign_fields(_strip_words(), _pcfg())
    assert fields["date_fab"][0] == "11/2025"
    assert fields["date_exp"][0] == "11/2027"


def test_assign_fields_lot_code_and_ppa() -> None:
    fields = assign_fields(_strip_words(), _pcfg())
    assert fields["num_lot"][0] == "B4521X"
    assert "20/05B/123" in fields["num_enregistrement"][0].replace(" ", "/")
    assert fields["ppa"][0] == "248,50"  # the LARGEST amount wins


def test_assign_fields_single_date_is_expiry() -> None:
    words = [Word("12/2026", 0.9, 0.5, 0.5, True)]
    fields = assign_fields(words, _pcfg())
    assert "date_exp" in fields and "date_fab" not in fields


def test_assign_fields_empty_input() -> None:
    assert assign_fields([], _pcfg()) == {}


# --------------------------------------------------------------------------- #
# VLM dataset builder (stub backend, synthetic source)
# --------------------------------------------------------------------------- #


def test_build_vlm_dataset_stub_on_synthetic(tmp_path: Path) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "build_vlm_dataset",
        Path(__file__).resolve().parents[1] / "scripts" / "build_vlm_dataset.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_vlm_dataset"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    manifest = mod.build(
        output_dir=tmp_path, backend="stub",
        splits=["train"], source_dataset="synthetic",
    )
    assert manifest["splits"]["train"]["images"] > 0
    meta = tmp_path / "train" / "metadata.jsonl"
    assert meta.is_file()
    rows = [json.loads(line) for line in meta.read_text(encoding="utf-8").splitlines() if line]
    assert rows, "no metadata rows written"
    first = rows[0]
    assert "ground_truth" in first and "gt_parse" in first["ground_truth"]
    gt = first["ground_truth"]["gt_parse"]
    # Stub values are AUTO_<FIELD>; the 4 business-critical fields must be there.
    assert gt.get("num_lot") == "AUTO_NUM_LOT"
    assert gt.get("num_enregistrement") == "AUTO_NUM_ENREGISTREMENT"
    # And the referenced image was copied next to the metadata.
    assert (tmp_path / "train" / first["file_name"]).is_file()
    assert (tmp_path / "manifest.json").is_file()


# --------------------------------------------------------------------------- #
# Orchestrator variant plumbing
# --------------------------------------------------------------------------- #


def test_pipeline_variant_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from vignocr.pipeline.orchestrator import VignocrPipeline

    assert VignocrPipeline()._variant == "v1"
    monkeypatch.setenv("VIGNOCR_PIPELINE_VARIANT", "fullpage")
    assert VignocrPipeline()._variant == "fullpage"
    monkeypatch.setenv("VIGNOCR_PIPELINE_VARIANT", "nonsense")
    assert VignocrPipeline()._variant == "v1"  # invalid -> safe fallback


def test_pipeline_variant_reported_in_model_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vignocr.pipeline.orchestrator import VignocrPipeline

    monkeypatch.setenv("VIGNOCR_PIPELINE_VARIANT", "fullpage")
    versions = VignocrPipeline().model_versions()
    assert versions["variant"] == "fullpage"
    assert versions["recognizer"] == "parseq"
    monkeypatch.delenv("VIGNOCR_PIPELINE_VARIANT")
    assert VignocrPipeline().model_versions()["variant"] == "v1"
