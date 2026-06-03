#!/usr/bin/env bash
# =============================================================================
# slurm/06_pipeline_benchmark.sbatch  —  STAGE 6: end-to-end pipeline benchmark.
#
# Runs the full orchestrator (preprocess -> detect -> orient+crop -> recognize
# -> parse+checksum+PPA -> nomenclature correct -> reimbursability -> assemble
# ExtractionRecord) over the eval split and records throughput + per-stage
# latency (timings_ms) + abstention/checksum/conflict rates.
#
# Single A100 (the detector + recognizer run on GPU; parsing/nomenclature are
# CPU). Consumes the artifacts from stages 03 (detector ONNX) and 04 (OCR ckpt).
#
# Entrypoint (documented orchestrator contract, docs/INTERFACES.md `pipeline/`
# + serving/deps.py construction): we drive the public API directly rather than
# depend on a benchmark module that may not exist yet:
#   VignocrPipeline({"detector_path","recognizer_path","nomenclature_csv",
#                    "default_flow"}).extract(image, flow=...) -> ExtractionRecord
# The inline harness times extract() over the split and aggregates latency
# percentiles, throughput, and abstention/checksum/conflict rates. ExtractionRecord
# carries per-stage timings_ms, which we summarize too.
#
# Artifact discovery: explicit env overrides win, else the latest stage-03/04 run
# dirs. Flow profile via $VIGNOCR_FLOW. If vignocr.pipeline isn't built yet, the
# import raises a clear message (this stage is gated behind Phase 6).
# =============================================================================
#SBATCH --job-name=vignocr-06-bench
#SBATCH --account=def-<PI>
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -Eeuo pipefail
SBATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=lib.sh
source "$SBATCH_DIR/lib.sh"

vignocr_preamble "06_pipeline_benchmark"

BENCH_SPLIT="${VIGNOCR_BENCH_SPLIT:-test}"
FLOW="${VIGNOCR_FLOW:-selling}"

# --- detector ONNX ---
DETECTOR="${VIGNOCR_DETECTOR_ONNX:-}"
if [[ -z "$DETECTOR" ]]; then
  EVAL_RUN="$(vignocr_latest_run 03_eval_export_detection)"
  [[ -n "$EVAL_RUN" && -f "$EVAL_RUN/DETECTOR_ONNX" ]] && DETECTOR="$(cat "$EVAL_RUN/DETECTOR_ONNX")"
fi
[[ -n "$DETECTOR" && -f "$DETECTOR" ]] || vdie "detector ONNX not found — run stage 03 or set VIGNOCR_DETECTOR_ONNX"

# --- OCR recognizer checkpoint (a directory for the baseline backend) ---
RECOGNIZER="${VIGNOCR_OCR_CKPT:-}"
if [[ -z "$RECOGNIZER" ]]; then
  OCR_RUN="$(vignocr_latest_run 04_train_ocr)"
  [[ -n "$OCR_RUN" && -f "$OCR_RUN/BEST_CKPT" ]] && RECOGNIZER="$(cat "$OCR_RUN/BEST_CKPT")"
fi
[[ -n "$RECOGNIZER" && -e "$RECOGNIZER" ]] || vdie "OCR recognizer not found — run stage 04 or set VIGNOCR_OCR_CKPT"

# Nomenclature CSV: explicit override, else the project-space copy, else the
# package default (configs/nomenclature/correction.yaml: csv.path).
NOMENCLATURE_CSV="${VIGNOCR_NOMENCLATURE_CSV:-}"
[[ -z "$NOMENCLATURE_CSV" && -f "$VIGNOCR_PROJECT_DIR/nomenclature.csv" ]] && NOMENCLATURE_CSV="$VIGNOCR_PROJECT_DIR/nomenclature.csv"

vlog "benchmark: detector=$DETECTOR recognizer=$RECOGNIZER split=$BENCH_SPLIT flow=$FLOW nomenclature=${NOMENCLATURE_CSV:-<config default>}"
{ echo "detector=$DETECTOR"; echo "recognizer=$RECOGNIZER"; echo "nomenclature=${NOMENCLATURE_CSV:-<config default>}"; } > "$VIGNOCR_RUN_DIR/ARTIFACTS"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /' || vwarn "nvidia-smi unavailable"

BENCH_JSON="$VIGNOCR_RUN_DIR/pipeline_benchmark.json"
vlog "running end-to-end pipeline benchmark -> $BENCH_JSON"
# Inline harness over the documented VignocrPipeline contract. Args:
#   detector  recognizer  nomenclature_csv  split  flow  out  warmup
# `set +e` around the tee'd pipeline to capture PIPESTATUS for our own error.
set +e
python - "$DETECTOR" "$RECOGNIZER" "$NOMENCLATURE_CSV" "$BENCH_SPLIT" "$FLOW" "$BENCH_JSON" "${VIGNOCR_BENCH_WARMUP:-1}" 2>&1 \
  <<'PY' | tee "$VIGNOCR_RUN_DIR/pipeline_benchmark.log"
import json, sys, time
from pathlib import Path
from statistics import mean

detector, recognizer, nomenclature, split, flow, out, warmup = sys.argv[1:8]
warmup = int(warmup)

from vignocr.common import get_active_dataset
# Construct via the exact cfg dict serving/deps.py uses (the documented contract).
cfg = {
    "detector_path": detector,
    "recognizer_path": recognizer,
    "nomenclature_csv": (nomenclature or None),
    "default_flow": flow,
}
try:
    from vignocr.pipeline.orchestrator import VignocrPipeline
except ImportError as exc:
    raise SystemExit(
        "vignocr.pipeline.orchestrator is not available yet — the end-to-end "
        "pipeline (ROADMAP Phase 6) must be built before the benchmark can run. "
        f"Original import error: {exc}"
    )

pipeline = VignocrPipeline(cfg)

# Enumerate the split's images directly from the active dataset's COCO layout
# (keeps the benchmark independent of any data-module CLI).
ds = get_active_dataset()
root = Path(ds["root"])
split_dir = root / ds.get("splits", {}).get(split, split)
coco_name = ds.get("coco_filename", "_annotations.coco.json")
images = sorted(
    p for p in split_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name != coco_name
) if split_dir.is_dir() else []
if not images:
    raise SystemExit(f"no images found under {split_dir} for split '{split}'")

from PIL import Image as PILImage

def _extract(path):
    img = PILImage.open(path).convert("RGB")
    t0 = time.perf_counter()
    rec = pipeline.extract(img, flow=flow)
    return (time.perf_counter() - t0) * 1000.0, rec

# Warmup (model load / first-call JIT) — excluded from timing.
for p in images[:warmup]:
    _extract(p)

lat_ms = []
stage_ms = {}
n_abstain = n_checksum_ok = n_conflict = 0
for p in images:
    ms, rec = _extract(p)
    lat_ms.append(ms)
    for k, v in (rec.timings_ms or {}).items():
        stage_ms.setdefault(k, []).append(float(v))
    if rec.abstentions:
        n_abstain += 1
    if rec.checksum and rec.checksum.verdict in ("ok", "repaired"):
        n_checksum_ok += 1
    if rec.nomenclature and rec.nomenclature.conflicts:
        n_conflict += 1

def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]

n = len(lat_ms)
total_s = sum(lat_ms) / 1000.0
result = {
    "split": split, "flow": flow, "n_images": n,
    "detector": detector, "recognizer": recognizer,
    "model_versions": pipeline.model_versions() if hasattr(pipeline, "model_versions") else {},
    "latency_ms": {
        "mean": mean(lat_ms), "p50": _pct(lat_ms, 0.50),
        "p90": _pct(lat_ms, 0.90), "p95": _pct(lat_ms, 0.95),
        "p99": _pct(lat_ms, 0.99), "min": min(lat_ms), "max": max(lat_ms),
    },
    "throughput_ips": (n / total_s) if total_s > 0 else None,
    "per_stage_ms_mean": {k: mean(v) for k, v in sorted(stage_ms.items())},
    "rates": {
        "abstention_image_rate": n_abstain / n,
        "checksum_ok_rate": n_checksum_ok / n,
        "nomenclature_conflict_rate": n_conflict / n,
    },
}
Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
lm = result["latency_ms"]
print(f"[bench] n={n} thrpt={result['throughput_ips']:.2f} img/s "
      f"p50={lm['p50']:.1f}ms p95={lm['p95']:.1f}ms "
      f"abstain_rate={result['rates']['abstention_image_rate']:.2f} "
      f"checksum_ok={result['rates']['checksum_ok_rate']:.2f}")
PY
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -ne 0 ]] && vdie "pipeline benchmark failed (rc=$rc) — see $VIGNOCR_RUN_DIR/pipeline_benchmark.log"

touch "$VIGNOCR_RUN_DIR/BENCHMARK_OK"
vlog "stage 06 complete — benchmark=$BENCH_JSON run_dir=$VIGNOCR_RUN_DIR"
vlog "=== VignOCR DAG finished: all stages green ==="
