"""Latency / throughput harness for the stub-backed pipeline (CPU-only).

Marked :mod:`@pytest.mark.slow` so the default fast run can skip it
(``pytest -m "not slow"``). It measures per-extraction wall-time over N synthetic
images and reports ms/extraction and throughput. It is a *smoke* benchmark — it
asserts the harness runs and produces sane, positive numbers, not a hard latency
SLA (which is meaningless under the deterministic stub and varies by machine).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from vignocr.common.config import get_active_dataset
from vignocr.data.coco import load_split
from vignocr.pipeline.orchestrator import VignocrPipeline

pytestmark = pytest.mark.slow


def _image_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for split in get_active_dataset().get("splits", {}).values():
        sp = load_split(root, split)
        paths.extend(root / split / img["file_name"] for img in sp.images)
    return paths


def test_extraction_latency_throughput(synthetic_root: Path) -> None:
    """Run extract over every fixture image; report ms/extraction + throughput."""
    pipeline = VignocrPipeline({"backend": "stub"})
    paths = _image_paths(synthetic_root)
    assert paths, "no fixture images to benchmark"

    # Warm up caches (ground-truth load, config, nomenclature index) so the timed
    # loop measures steady-state extraction, not one-off initialization.
    pipeline.extract(paths[0], flow="selling")

    per_image_ms: list[float] = []
    t0 = time.perf_counter()
    for path in paths:
        s = time.perf_counter()
        record = pipeline.extract(path, flow="selling")
        per_image_ms.append((time.perf_counter() - s) * 1000.0)
        # Every extraction is internally timed by the orchestrator too.
        assert record.timings_ms, "orchestrator should report stage timings"
        assert sum(record.timings_ms.values()) > 0.0
    total_s = time.perf_counter() - t0

    n = len(paths)
    mean_ms = sum(per_image_ms) / n
    p95_ms = sorted(per_image_ms)[min(n - 1, int(round(0.95 * (n - 1))))]
    throughput = n / total_s if total_s > 0 else float("inf")

    report = (
        f"\n[benchmark] n={n}  mean={mean_ms:.2f} ms/extraction  "
        f"p95={p95_ms:.2f} ms  throughput={throughput:.1f} extractions/s"
    )
    print(report)

    # Sanity (not an SLA): positive, finite timings and throughput.
    assert mean_ms > 0.0
    assert p95_ms >= 0.0
    assert throughput > 0.0


def test_throughput_harness_handles_repeated_runs(synthetic_root: Path) -> None:
    """Repeated extraction of the same image is stable and cached (no growth in error)."""
    pipeline = VignocrPipeline({"backend": "stub"})
    path = _image_paths(synthetic_root)[0]

    records = [pipeline.extract(path, flow="selling") for _ in range(5)]
    # Deterministic: identical image_id and checksum verdict across runs.
    assert len({r.image_id for r in records}) == 1
    assert {r.checksum.verdict for r in records} == {"ok"}
