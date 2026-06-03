# VignOCR — Monitoring & Observability

What we watch in production to know VignOCR is *still trustworthy*: per-field
accuracy and confidence, model drift, failed-extraction capture, latency and GPU
utilization, plus the logging/alerting and dashboards that make it actionable.

Monitoring is tightly coupled to the human loop: the same confirm popup that
captures corrections for retraining ([`docs/HITL.md`](HITL.md)) is our **ground-truth
oracle** for production accuracy. We do not need a separate labeling pipeline to
know how the model is doing — the counter tells us.

---

## 1. Per-field accuracy + confidence

The unit of monitoring is the **field**, not the image — a record can be 11/12
fields perfect and still be unusable if the wrong field is `ppa`.

### Accuracy (from the human oracle)

For every confirmed extraction, compare what VignOCR proposed to what the human
confirmed (see HITL capture):

- **Per-field acceptance rate** — fraction confirmed unchanged. The headline
  health metric, tracked **per field name** from `configs/classes.yaml`.
- **Per-field correction rate** — `1 − acceptance`. Broken out by
  `correction_kind` (`value_fix`, `box_fix`, `abstain_resolved`, `false_field`).
- **Business-critical accuracy** — acceptance for `ppa, prix, shp,
  num_enregistrement, num_lot` tracked separately with the tightest alert
  thresholds. These are the fields the business rule depends on.
- **Dispensing-critical accuracy** — `dci, dosage, forme` tracked separately
  (medical safety).
- **By flow** — `selling` vs `receiving` (the stricter selling profile should show
  fewer *confident-wrong* events).

### Confidence (from the model itself, no oracle needed)

- **Confidence distribution per field** — histogram of `FieldRead.confidence`. A
  drifting layout shows up as the distribution sliding *before* accuracy visibly
  drops.
- **Abstention rate per field** — fraction with `status == "abstain"`, vs the
  configured τ (`configs/parsing/fields.yaml → abstention[flow]`). Rising
  abstention = the model increasingly unsure = look upstream (input quality? new
  vignette variant?).
- **Calibration** — confident-and-wrong is the dangerous quadrant. Track the rate
  of `model_confidence ≥ τ` events that the human nonetheless corrected; this is
  the single most important *safety* metric. A reliability plot (predicted conf vs
  empirical accuracy) shows whether the threshold is set right.

### Post-processing health (free, no oracle)

These come straight off the `ExtractionRecord` and need no human label:

- **Checksum verdict mix** — rate of `ok / repaired / mismatch / incomplete`
  (`ChecksumReport.verdict`). A rising `mismatch` rate is an early money-accuracy
  alarm; a rising `repaired` rate means OCR is leaning on the checksum to survive.
- **Nomenclature match rate** — fraction with `NomenclatureReport.matched == true`;
  falling match rate means `num_enregistrement` recognition or the nomenclature
  version is stale.
- **Conflict rate** — `flag_on_conflict` events (`dosage`/`forme`
  OCR↔nomenclature disagreements). Spikes are both a model signal and a
  safety-review trigger.

---

## 2. Drift detection

Production input drifts (new vignette layouts, new lab packaging, seasonal product
mix, camera/lighting changes). We detect it on three signals, **earliest first**:

1. **Input/image drift (leading indicator).** Track distributions of input
   features that don't need a label: image brightness/contrast, blur estimate,
   detected box count per image, average detection score, fraction of images with
   `color_band` detected. A population-stability shift here predicts accuracy loss
   before the oracle confirms it.
2. **Prediction drift (no oracle).** Shift in the model's own outputs vs a
   reference window: confidence distributions per field (§1), abstention rate,
   checksum verdict mix, nomenclature match rate, reimbursability colour mix. These
   are continuous and cheap.
3. **Concept drift (oracle-confirmed, lagging but definitive).** A sustained rise
   in per-field correction rate / confident-wrong rate from the human loop. This is
   the ground truth that a retrain is due.

Each signal is compared against a **reference window** (a frozen recent
known-good period, and/or the validation-set distribution). Alerting is on
*sustained* deviation, not single-batch noise. A drift alert is also a **HITL
prioritization signal** — when field X drifts, its corrections jump the review
queue ([`docs/HITL.md`](HITL.md) §3).

---

## 3. Failed-extraction capture

Anything that doesn't produce a clean, confirmable record is captured for triage —
this is both an SRE signal and a rich source of hard training examples.

Categories captured:

- **Hard failures** — request errors: upload rejected (MIME/size), inference
  exception, timeout, model-not-ready. Counted and rate-alerted.
- **No-detection** — Stage 1 found no boxes (or none for any business-critical
  field). Often a bad photo; sometimes a real regression.
- **Total abstention** — every business-critical field abstained (the human got an
  empty prefill). Tracked as a UX failure even though it's "safe".
- **Checksum mismatch** — `verdict == "mismatch"` survived to the human (money
  could not be reconciled).
- **Pipeline-stage failures** — which stage failed (detect / orient / read / parse
  / correct), from `timings_ms` keys + structured error context.

Captured failures are **de-identified** (same posture as HITL §2 — crop, strip PII)
and routed to the correction queue as high-priority candidates. A production
failure becomes tomorrow's training example.

---

## 4. Latency + GPU utilization

VignOCR runs at a pharmacy counter — **interactive latency** is a feature, not a
nicety.

### Latency

- **End-to-end `/extract` latency** — p50 / p95 / p99. The user-facing SLO.
- **Per-stage latency** — straight from `ExtractionRecord.timings_ms`
  (preprocess, detect, orient+crop, read, parse, correct). Tells you *where* a p99
  regression lives without guessing.
- **Queue/wait time** vs **compute time** — distinguishes "model is slow" from
  "we're under-provisioned".

### GPU / throughput (inference hosts)

- **GPU utilization %** and **GPU memory** per inference worker — under-utilized =
  over-provisioned; pinned at 100% with rising latency = need more capacity.
- **Batch efficiency** and **throughput** (requests/sec) vs concurrency.
- **Model load / warmup** time and **readiness** — `/ready` reports model load
  state (`{"ready": bool, "models": {...}}`); a worker serving before ready is an
  alert.
- **ONNX-vs-torch parity** is verified at export time (not in prod), but the
  **serving runtime** (ONNX Runtime) GPU/CPU split and provider in use are recorded
  so a silent CPU-fallback (which tanks latency) is caught.

---

## 5. Logging & alerting

### Logging

- **Structured logs** via `vignocr.common.get_logger` (structlog) — JSON in prod,
  one event per request with: request id / idempotency key, `flow`,
  `model_versions`, per-stage `timings_ms`, per-field `status`+`confidence`
  (values redacted/hashed — never log raw PII or full field values), checksum
  verdict, nomenclature match + conflict count, abstention list.
- **Correlation** — the request id ties the prod log, the failure capture, and any
  resulting HITL correction together.
- **No PII in logs** — same de-identification rule as HITL §2 and the security
  model. Field *values* are not logged; field *metadata* (status, confidence,
  bbox) is.

### Alerting (page vs ticket)

| Severity | Example trigger                                                                 |
| -------- | ------------------------------------------------------------------------------ |
| **Page** | `/extract` p95 over SLO; error rate spike; `/ready` flapping; CPU-fallback on GPU host; **confident-wrong rate on a dispensing-critical field crosses its safety threshold**. |
| **Page** | Checksum `mismatch` rate spike (money trust); nomenclature match rate collapse. |
| **Ticket** | Sustained drift on an input/prediction signal; per-field correction-rate creep; rising total-abstention rate; GPU chronically under/over-utilized. |

Safety-relevant alarms (confident-wrong on `ppa`/dispensing fields, checksum
mismatch surge) are treated as **incidents**, not capacity issues.

---

## 6. The dashboards that matter

Four views, in priority order:

1. **Safety & accuracy (the one to look at first).**
   - Business-critical per-field acceptance + correction rate (trend).
   - **Confident-wrong rate** per dispensing-critical field (the safety gauge).
   - Checksum verdict mix; nomenclature match + conflict rate.
   - Abstention rate per field vs τ.

2. **Drift.**
   - Input-feature stability (brightness/blur/box-count/colour mix) vs reference.
   - Confidence-distribution shift per field.
   - Concept-drift (oracle correction-rate) trend — overlaid with model-version
     deploy markers so a regression is pinned to a release.

3. **Performance.**
   - `/extract` p50/p95/p99 end-to-end + **per-stage `timings_ms`** breakdown.
   - Throughput vs concurrency; GPU utilization & memory; runtime provider
     (GPU/CPU) per worker.

4. **Reliability & flow.**
   - Error/failure rate by category (§3); no-detection & total-abstention rates.
   - Volume split by `flow` (selling vs receiving) and the corresponding abstention
     profiles in effect.
   - Active model versions in the fleet (`model_versions`) — for canary/rollback.

Every metric keyed by field name or threshold reads it from `configs/` (class
names from `classes.yaml`, τ from `parsing/fields.yaml`) — dashboards are
config-driven too, so adding a field or changing a threshold doesn't require
re-plumbing the monitoring.

---

## See also

- [`docs/HITL.md`](HITL.md) — the correction loop that supplies the accuracy oracle
  and consumes drift/failure signals as training data.
- [`docs/TESTING.md`](TESTING.md) — pre-prod gates (regression goldens, benchmarks)
  that a model must pass before it can be monitored in prod.
- [`docs/INTERFACES.md`](INTERFACES.md) — `ExtractionRecord` fields
  (`timings_ms`, `model_versions`, per-field `status`/`confidence`) that every
  metric above is derived from.
