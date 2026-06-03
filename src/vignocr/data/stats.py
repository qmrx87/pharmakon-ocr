"""Dataset statistics: per-class counts, per-split sizes, box-size quartiles.

Pure-CPU, dependency-light. ``python -m vignocr.data.stats`` prints a readable
summary for the active dataset (``configs/data.yaml``); :func:`summarize` returns
the same data as a plain dict for tests / dashboards.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vignocr.common import get_active_dataset, get_logger
from vignocr.data.coco import load_split

log = get_logger(__name__)


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted list (empty -> 0.0)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _box_quartiles(values: list[float]) -> dict[str, float]:
    """min / q1 / median / q3 / max / mean of a list of box measures."""
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    s = sorted(values)
    return {
        "count": len(s),
        "min": float(s[0]),
        "q1": round(_quantile(s, 0.25), 3),
        "median": round(_quantile(s, 0.5), 3),
        "q3": round(_quantile(s, 0.75), 3),
        "max": float(s[-1]),
        "mean": round(sum(s) / len(s), 3),
    }


def summarize(root: Path | str) -> dict[str, Any]:
    """Summarize the dataset at ``root``.

    Returns a dict with:
      * ``splits``        — per-split ``{images, annotations}`` sizes.
      * ``per_class``     — class name -> annotation count (summed across splits).
      * ``per_class_by_split`` — class name -> {split: count}.
      * ``box_size``      — width/height/area quartiles (pixels), overall.
      * ``box_size_by_class`` — area quartiles per class.
      * ``totals``        — overall image + annotation counts.

    Splits that are missing on disk are skipped (recorded under ``missing_splits``).
    """
    root = Path(root)
    ds = get_active_dataset()
    split_dirs: list[str] = list(ds.get("splits", {}).values()) or ["train", "valid", "test"]

    splits_info: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    per_class_by_split: dict[str, dict[str, int]] = {}
    class_counter: Counter[str] = Counter()

    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    areas_by_class: dict[str, list[float]] = {}

    total_images = 0
    total_anns = 0

    for split in split_dirs:
        try:
            sp = load_split(root, split)
        except FileNotFoundError:
            missing.append(split)
            continue

        splits_info[split] = {"images": len(sp.images), "annotations": len(sp.annotations)}
        total_images += len(sp.images)
        total_anns += len(sp.annotations)

        for ann in sp.annotations:
            cid = int(ann["category_id"])
            name = sp.cat_id_to_name.get(cid, f"<unknown:{cid}>")
            class_counter[name] += 1
            split_counts = per_class_by_split.setdefault(name, {})
            split_counts[split] = split_counts.get(split, 0) + 1
            try:
                _x, _y, w, h = (float(v) for v in ann["bbox"])
            except (KeyError, TypeError, ValueError):
                continue
            widths.append(w)
            heights.append(h)
            area = float(ann.get("area", w * h))
            areas.append(area)
            areas_by_class.setdefault(name, []).append(area)

    result: dict[str, Any] = {
        "root": str(root),
        "dataset_name": ds.get("name"),
        "splits": splits_info,
        "missing_splits": missing,
        "totals": {
            "images": total_images,
            "annotations": total_anns,
            "classes_present": len(class_counter),
        },
        "per_class": dict(sorted(class_counter.items())),
        "per_class_by_split": {k: per_class_by_split[k] for k in sorted(per_class_by_split)},
        "box_size": {
            "width": _box_quartiles(widths),
            "height": _box_quartiles(heights),
            "area": _box_quartiles(areas),
        },
        "box_size_by_class": {
            name: _box_quartiles(areas_by_class[name]) for name in sorted(areas_by_class)
        },
    }
    return result


def _print_summary(summary: dict[str, Any]) -> None:
    """Render :func:`summarize` output as an aligned text table."""
    lines: list[str] = []
    lines.append(f"Dataset: {summary['dataset_name']}  ({summary['root']})")

    totals = summary["totals"]
    lines.append(
        f"Totals: {totals['images']} images, {totals['annotations']} annotations, "
        f"{totals['classes_present']} classes present"
    )
    if summary["missing_splits"]:
        lines.append(f"Missing splits: {', '.join(summary['missing_splits'])}")

    lines.append("")
    lines.append("Per-split sizes:")
    lines.append(f"  {'split':<10}{'images':>10}{'annotations':>14}")
    for split, info in summary["splits"].items():
        lines.append(f"  {split:<10}{info['images']:>10}{info['annotations']:>14}")

    lines.append("")
    lines.append("Per-class annotation counts:")
    lines.append(f"  {'class':<22}{'count':>8}")
    for name, count in summary["per_class"].items():
        lines.append(f"  {name:<22}{count:>8}")

    lines.append("")
    lines.append("Box-size quartiles (pixels):")
    for measure in ("width", "height", "area"):
        q = summary["box_size"][measure]
        lines.append(
            f"  {measure:<7} n={q['count']:<6} min={q['min']:.1f}  q1={q['q1']:.1f}  "
            f"median={q['median']:.1f}  q3={q['q3']:.1f}  max={q['max']:.1f}  mean={q['mean']:.1f}"
        )

    print("\n".join(lines))


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    _ds = get_active_dataset()
    _root = Path(_ds["root"])
    if not _root.exists() and _ds.get("name") == "synthetic":
        # Convenience: generate the fixture first so the command always works.
        from vignocr.data.synthetic import _generate_from_config

        _generate_from_config()
    _print_summary(summarize(_root))
