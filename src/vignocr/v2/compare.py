"""Head-to-head harness: v1 cascade vs v2a (Donut) vs v2b (full-page docTR).

Evaluates every available variant on the SAME held-out reference — the test
split of the VLM dataset (``metadata.jsonl``: image + per-field values, human-
reviewed where the Roboflow pass happened). Reports, per variant x field:

* exact-match rate (after type-aware normalization — see ``v2.normalize``)
* CER (Levenshtein / reference length)
* coverage (how often the variant emitted the field at all)
* latency p50 / p95 per image

A variant whose weights are missing is SKIPPED with a notice (so the harness
runs incrementally as training jobs land). Until the review pass completes,
the reference is OCR-bootstrapped — treat absolute numbers with care; the
RANKING between variants is still informative because all share the reference.

CLI:  python -m vignocr.v2.compare --dataset ocr_dataset_vlm --split test \\
          --variants vlm,fullpage --vlm-dir <ckpt>/best --out report_dir
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from vignocr.common import get_logger
from vignocr.v2.normalize import normalize_value

log = get_logger(__name__)

REPORT_FIELDS = ["num_lot", "date_fab", "date_exp", "num_enregistrement"]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _read_rows(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    meta = dataset_dir / split / "metadata.jsonl"
    rows = []
    with meta.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_extractors(args: argparse.Namespace) -> dict[str, Any]:
    """Construct the requested extractors; skip (with a notice) what can't load."""
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    out: dict[str, Any] = {}
    for name in want:
        try:
            if name == "vlm":
                from vignocr.v2.donut_infer import DonutExtractor

                out[name] = DonutExtractor(args.vlm_dir)
            elif name == "fullpage":
                from vignocr.v2.fullpage import FullPageExtractor

                out[name] = FullPageExtractor()
            elif name == "claude":
                from vignocr.v2.claude_extract import ClaudeExtractor

                out[name] = ClaudeExtractor()  # needs network + ANTHROPIC_API_KEY
            elif name == "v1":
                out[name] = _V1Adapter(args.v1_detector, args.v1_cfg)
            else:
                log.warning("compare.unknown_variant", variant=name)
        except (FileNotFoundError, ImportError) as exc:
            log.warning("compare.variant_skipped", variant=name, reason=str(exc))
    return out


class _V1Adapter:
    """Adapt the v1 cascade (Stage B detector + per-field OCR) to extract()."""

    def __init__(self, detector_path: str | None, cfg_path: str) -> None:
        if not detector_path:
            raise FileNotFoundError("v1 requested but --v1-detector not provided")
        from vignocr.pipeline.orchestrator import VignocrPipeline

        self._pipe = VignocrPipeline({"detector_path": detector_path, "backend": "real"})
        self._cfg_path = cfg_path

    def extract(self, pil: Any) -> dict[str, tuple[str, float]]:
        record = self._pipe.extract(pil)
        return {
            name: (fr.value or fr.raw or "", float(fr.confidence))
            for name, fr in record.fields.items()
            if (fr.value or fr.raw)
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image

    dataset_dir = Path(args.dataset)
    rows = _read_rows(dataset_dir, args.split)
    if args.limit:
        rows = rows[: args.limit]
    extractors = _build_extractors(args)
    if not extractors:
        raise SystemExit("no variant could be constructed — nothing to compare")
    log.info("compare.start", n_images=len(rows), variants=sorted(extractors))

    report: dict[str, Any] = {"split": args.split, "n_images": len(rows), "variants": {}}
    for name, ex in extractors.items():
        per_field = {f: {"hits": 0, "total": 0, "cer_sum": 0.0, "emitted": 0}
                     for f in REPORT_FIELDS}
        lat_ms: list[float] = []
        for row in rows:
            img = Image.open(dataset_dir / args.split / row["file_name"]).convert("RGB")
            t0 = time.perf_counter()
            try:
                pred = ex.extract(img)
            except Exception as exc:  # noqa: BLE001 — a variant crash != harness crash
                log.warning("compare.extract_failed", variant=name, err=str(exc))
                continue
            lat_ms.append((time.perf_counter() - t0) * 1000)
            gt = row["ground_truth"]["gt_parse"]
            for f in REPORT_FIELDS:
                if f not in gt:
                    continue
                stats = per_field[f]
                stats["total"] += 1
                ref = normalize_value(f, gt[f])
                hyp_raw = pred.get(f)
                if hyp_raw is None:
                    stats["cer_sum"] += 1.0  # missing == 100% error
                    continue
                stats["emitted"] += 1
                hyp = normalize_value(f, hyp_raw[0])
                stats["hits"] += int(hyp == ref)
                stats["cer_sum"] += _levenshtein(hyp, ref) / max(1, len(ref))

        fields_out = {}
        for f, s in per_field.items():
            if not s["total"]:
                continue
            fields_out[f] = {
                "exact_match": round(s["hits"] / s["total"], 4),
                "cer": round(s["cer_sum"] / s["total"], 4),
                "coverage": round(s["emitted"] / s["total"], 4),
                "n": s["total"],
            }
        report["variants"][name] = {
            "fields": fields_out,
            "mean_exact_match": round(
                statistics.mean(v["exact_match"] for v in fields_out.values()), 4
            ) if fields_out else 0.0,
            "latency_ms": {
                "p50": round(statistics.median(lat_ms), 1) if lat_ms else None,
                "p95": round(sorted(lat_ms)[int(0.95 * (len(lat_ms) - 1))], 1)
                if lat_ms else None,
            },
        }
        log.info("compare.variant_done", variant=name, **report["variants"][name]["latency_ms"])

    # Fold in a prior report's variants (e.g. the on-cluster v1/vlm/fullpage run)
    # so the off-cluster `claude` run produces ONE combined table. New variants
    # win on key collisions.
    merge_path = getattr(args, "merge", None)
    if merge_path and Path(merge_path).is_file():
        try:
            prior = json.loads(Path(merge_path).read_text(encoding="utf-8"))
            merged = dict(prior.get("variants", {}))
            merged.update(report["variants"])
            report["variants"] = merged
            log.info("compare.merged", source=str(merge_path), variants=sorted(merged))
        except (OSError, ValueError) as exc:
            log.warning("compare.merge_failed", source=str(merge_path), err=str(exc))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown(report, out_dir / "compare_report.md")
    log.info("compare.done", out=str(out_dir))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# VignOCR variant comparison",
        f"split: {report['split']} | images: {report['n_images']}",
        "",
        "| variant | " + " | ".join(REPORT_FIELDS) + " | mean EM | p50 ms | p95 ms |",
        "|---|" + "---|" * (len(REPORT_FIELDS) + 3),
    ]
    for name, v in report["variants"].items():
        cells = []
        for f in REPORT_FIELDS:
            s = v["fields"].get(f)
            cells.append(f"{s['exact_match']:.2%} (cov {s['coverage']:.0%})" if s else "—")
        lat = v["latency_ms"]
        lines.append(
            f"| {name} | " + " | ".join(cells)
            + f" | {v['mean_exact_match']:.2%} | {lat['p50']} | {lat['p95']} |"
        )
    lines += ["", "EM = normalized exact match; cov = field emitted at all.",
              "Reference = VLM dataset values (OCR-bootstrapped until human review)."]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="ocr_dataset_vlm")
    p.add_argument("--split", default="test")
    p.add_argument("--variants", default="vlm,fullpage",
                   help="comma list: v1,vlm,fullpage,claude (claude needs network + ANTHROPIC_API_KEY)")
    p.add_argument("--vlm-dir", default=None, help="fine-tuned Donut checkpoint dir")
    p.add_argument("--v1-detector", default=None, help="v1 Stage B checkpoint/onnx")
    p.add_argument("--v1-cfg", default="detection/rfdetr_medium")
    p.add_argument("--out", default="experiments/v2_compare")
    p.add_argument("--merge", default=None,
                   help="prior compare_report.json to fold in (e.g. the on-cluster run "
                        "when adding `claude` off-cluster) — one combined table")
    p.add_argument("--limit", type=int, default=0, help="cap images (0 = all)")
    args = p.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
