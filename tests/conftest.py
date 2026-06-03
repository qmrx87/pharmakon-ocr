"""Shared pytest fixtures for the VignOCR test suite.

The whole suite runs on a CPU-only box (``pytest -m "not ml"``) against a
**deterministically generated** synthetic fixture. This module owns that
generation: a session-scoped, autouse fixture (re)builds the synthetic dataset —
per-split images + ``_annotations.coco.json``, ``ground_truth.json``, and the
matching ``fixtures/nomenclature.csv`` — via :func:`vignocr.data.synthetic.generate`
using the seed/sizes declared in ``configs/data.yaml`` (the single source of
truth; nothing is hardcoded here).

Everything downstream — the pipeline stubs, ``nomenclature.loader.load_csv`` and
``data.validate.check_integrity`` — resolves its paths from config against the
repo root, so we generate into the *configured* root (``fixtures/synthetic``) to
keep the fixture, the nomenclature CSV, and the ground truth mutually consistent.

Fixtures exposed:
    * ``synthetic_root``  — ``pathlib.Path`` to the generated dataset root.
    * ``ground_truth``    — the loaded ``ground_truth.json`` dict (keyed by file_name).
    * ``data_config``     — the resolved active-dataset config dict.
    * ``nomenclature_csv``— ``pathlib.Path`` to the generated nomenclature CSV.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

# Pydantic warns that ``ExtractionRecord.model_versions`` shadows its protected
# ``model_`` namespace. It is harmless (the field is intentional) and out of this
# suite's ownership (schemas.py is owned elsewhere); silence it so it does not
# clutter collection output.
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_versions" .* has conflict with protected namespace "model_"\.',
    category=UserWarning,
)

from vignocr.common import get_active_dataset  # noqa: E402  (after warning filter)
from vignocr.data import synthetic  # noqa: E402


def _generate_synthetic_dataset() -> Path:
    """Deterministically (re)generate the active synthetic dataset from config.

    Mirrors ``vignocr.data.synthetic._generate_from_config`` but is inlined here
    so the test suite controls generation explicitly (and never depends on a
    private helper). Returns the dataset root.
    """
    ds = get_active_dataset()
    assert ds.get("name") == "synthetic", (
        f"test suite expects the synthetic dataset to be active, got {ds.get('name')!r}; "
        f"unset VIGNOCR_DATA_ACTIVE or set it to 'synthetic'."
    )

    seed = int(ds.get("seed", 1337))
    root = Path(ds["root"])
    splits: dict[str, str] = ds["splits"]  # logical -> directory name, e.g. {val: valid}
    num_images: dict[str, int] = ds["num_images"]  # keyed by logical split (train/val/test)
    # ``generate`` keys ``num`` by the on-disk split *directory* name.
    num = {splits[logical]: int(count) for logical, count in num_images.items()}
    size = tuple(int(v) for v in ds["image_size"])

    synthetic.generate(root, num=num, seed=seed, image_size=(size[0], size[1]))
    return root


@pytest.fixture(scope="session", autouse=True)
def _synthetic_dataset() -> Path:
    """Session-scoped, autouse: ensure the synthetic fixture exists before tests.

    Autouse so any test (and the FastAPI / pipeline paths that read the fixture
    from config-resolved paths) sees a freshly, deterministically generated
    dataset — without each test having to request it explicitly.
    """
    return _generate_synthetic_dataset()


@pytest.fixture(scope="session")
def data_config() -> dict[str, Any]:
    """The resolved active-dataset config (absolute ``root``, splits, sizes)."""
    return get_active_dataset()


@pytest.fixture(scope="session")
def synthetic_root(_synthetic_dataset: Path) -> Path:
    """Absolute path to the generated synthetic dataset root."""
    return Path(_synthetic_dataset)


@pytest.fixture(scope="session")
def ground_truth(synthetic_root: Path) -> dict[str, dict[str, Any]]:
    """The generated ground truth, keyed by image ``file_name``."""
    return synthetic.load_ground_truth(synthetic_root)


@pytest.fixture(scope="session")
def nomenclature_csv(synthetic_root: Path) -> Path:
    """Path to the nomenclature CSV the generator wrote (one level above root)."""
    return synthetic_root.parent / "nomenclature.csv"
