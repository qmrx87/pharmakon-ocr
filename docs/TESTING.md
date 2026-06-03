# VignOCR — Testing Strategy

The test suite exists to make one promise enforceable: **VignOCR does not silently
produce a wrong, confirmable value.** Money is asserted to the centime, identity to
the character, the safety policy is asserted to *flag rather than overwrite*, and
the whole thing runs **on CPU today** against the synthetic fixture — no GPU, no
network, no `torch`.

Run everything with:

```bash
pip install -e .[dev]
pytest                       # the full CPU suite (parsing, dataset, e2e, benchmarks)
pytest -m "not ml"           # explicit CPU-only (ml-marked tests auto-skip without [ml])
pytest -m ml                 # the GPU/heavy tests (needs pip install -e .[ml])
```

Markers are declared in `pyproject.toml`: `ml` (requires the heavy extra, skipped
on core CPU runs) and `slow` (long-running). The default run is the fast,
deterministic CPU gate.

---

## 1. OCR regression suite — golden vignettes (centime / character exact)

The backbone. A set of **golden vignettes** (synthetic now; real, anonymized,
human-verified cases added over time via [`docs/HITL.md`](HITL.md)) each pinned to
its **exact** expected `ExtractionRecord`. The assertions are exact, not fuzzy:

- **Money to the centime.** `ppa`/`prix`/`shp` assert against the expected
  **Decimal string** (e.g. `"702.56"`) — never a float, never an epsilon
  comparison. A one-centime drift fails the test.
- **Codes/dates to the character.** `num_enregistrement` asserts the normalized
  canonical form (`{a}/{b}/{c}{letter}{d}/{e}`); dates assert the `%Y-%m` output;
  `num_lot` asserts the uppercased canonical string.
- **Status & source pinned.** Each field's `status`
  (`ok/abstain/corrected/conflict/missing`) and `source`
  (`ocr/nomenclature/ocr+nomenclature/checksum/none`) are asserted — a value being
  *right by luck* with the wrong provenance still fails.
- **Reimbursability pinned.** The `color_band` → `{color, eligible, label}` is
  asserted (green→True, red→False, orange→None/abstain).

These goldens are also the **promotion gate** for retrained models (HITL §6 /
MONITORING pre-prod): a new model must reproduce every golden before it can ship.

### The deterministic-parsing core (pure CPU, no ML, no stubs)

The parsing/checksum/nomenclature layer is tested directly — it is pure Python and
the highest-leverage place to assert correctness:

- **Money parser** (`configs/parsing/fields.yaml → money`): mixed separators
  (`"700.06"` and `"702,56"`), thousands separators (`"1 234,56 DA"`), currency
  tokens (`DA/Da/da/DZD`) → all to the exact `Decimal`. Out-of-range
  (`> max_value`) → abstain. **Never** a float anywhere in the path.
- **Checksum** (`prix + shp == ppa`, tolerance `0.00`): assert each verdict —
  `ok` (already consistent), `repaired` (recompute the third when two are
  ≥ `min_conf_to_anchor`), `mismatch` (**flag, never silently accept**),
  `incomplete` (< 2 present).
- **PPA disambiguation**: an image with both an intermediate `700,06+2,50` and a
  final `= 702,56 DA` must select the **final**; with no final line, fall back to
  `sum_prix_shp`.
- **`num_enregistrement` normalization**: spacing variants
  (`"18/97/14G 061/003"`, `"09/22 F 018/235"`) → one canonical form; confusion-map
  repairs (`O→0`, `I→1`, `S→5`, …) on digit slots only.
- **Nomenclature correction — the safety assertions** (`configs/nomenclature/correction.yaml`):
  - `never_overwrite`: a confident nomenclature row **does not** change `ppa`/`tr`.
  - `repair_always`: `product_name`/`laboratoire` repaired from the matched row.
  - `flag_on_conflict`: a **confident OCR↔nomenclature `dosage` disagreement is
    flagged** (`status="conflict"`), **not overwritten** — this is the
    medical-safety test and it is non-negotiable.
  - `repair_if_ocr_low_or_agree`: `dci` repaired only when OCR abstained or agrees.
  - No match (< `min_match_confidence`) → keep OCR values, flag the anchor.
- **Abstention profiles**: the same borderline read abstains under `selling`
  (τ 0.90) but passes under `receiving` (τ 0.75).

---

## 2. Dataset-validation tests

Mirror the integrity asserts in `configs/data.yaml → integrity` as pytest gates
(`tests/test_dataset_integrity.py`), run against the synthetic fixture in CI and
against `real` as part of switchover ([`docs/SWITCHOVER.md`](SWITCHOVER.md)):

- **No image leakage across splits** — by `file_name` stem **and** perceptual hash
  (catches Roboflow's augmented variants of one source frame straddling splits).
- **Every annotation a valid bbox** — `0≤x, 0≤y, w>0, h>0`, within image bounds.
- **Class names ⊆ schema** — every COCO category *name* exists in
  `configs/classes.yaml`; a malformed fixture with an unknown category **fails**.
- **Business-critical coverage** — each split covers `business_critical_fields`
  (hard-fail on synthetic; warn on `real` until annotated).
- **By-name mapping robustness** — a fixture with shuffled category **ids** but
  correct **names** must load identically (proves we never key on id).
- **Determinism of generation** — regenerating the synthetic set with the same
  `seed`/`num_images`/`image_size` yields the **same manifest hash**
  (reproducibility, [`docs/DATASET.md`](DATASET.md) §4).

---

## 3. Inference benchmarking

Performance is a tested property, not folklore (`slow`-marked; the heavy paths are
also `ml`-marked):

- **Per-stage timing** — assert `ExtractionRecord.timings_ms` is populated for
  every stage (preprocess/detect/orient/read/parse/correct) and that the CPU stub
  e2e completes within a generous fixture budget (guards accidental O(n²)
  regressions in the pure-Python core).
- **End-to-end latency budget** — p50/p95 of `/extract` on the fixture model
  asserted under an SLO ceiling (the same metric monitored in prod, MONITORING §4).
- **ONNX ↔ torch parity** (`ml`) — the exported ONNX detector must match the torch
  checkpoint within tolerance on the fixtures (`detection/export.to_onnx` does this
  as part of export; the test pins it). A silent CPU-fallback or a divergent export
  fails here, not in prod.
- **Throughput smoke** (`ml`, `slow`) — batched inference throughput sanity on the
  fixture, so a serving-config regression is visible.

---

## 4. End-to-end workflow tests

Exercise the orchestrator and the API exactly as Pharmakon will
(`tests/test_pipeline_e2e.py`), with the detector/recognizer **stubbed by
deterministic fixture readers** so the whole flow runs on CPU without `[ml]`:

- **Full pipeline on the goldens** — `VignocrPipeline.extract(image, flow=...)`
  reproduces the pinned `ExtractionRecord` end to end (detect → orient+crop → read
  → parse+checksum+PPA → nomenclature → reimbursability → assemble).
- **FastAPI contract** (via `httpx`):
  - `GET /health` → `{"status":"ok"}`.
  - `GET /ready` → model-readiness shape `{"ready": bool, "models": {...}}`.
  - `POST /extract` (multipart image + `flow`) → an `ExtractionRecord` that
    **validates against the Pydantic schema** and serializes money as strings.
- **Flow semantics** — the same image through `selling` vs `receiving` yields the
  stricter/looser abstention behavior.
- **Schema round-trip** — `ExtractionRecord` (and every nested model) serializes to
  JSON and re-parses identically; money fields are strings, never floats; this is
  the stub-payload validation the integration contract depends on.
- **Graceful degradation** — with `[ml]` absent, importing and running the core +
  stubbed pipeline must succeed; importing a heavy module's training entrypoint
  raises the clear `pip install -e .[ml]` `ImportError` (asserted), never a bare
  `ModuleNotFoundError` at import time.

---

## 5. Field-capture test matrix

Real vignettes are photographed at a counter under hostile conditions. This matrix
defines the capture-robustness cases every business-critical field must survive;
each cell is a fixture (synthetic perturbation now; real anonymized samples folded
in via HITL) with a pinned expected outcome — where "expected" is **either the
correct value or a principled `abstain`** (degrading to "à vérifier" is a *pass*;
silently emitting a wrong value is a *fail*).

| Condition            | What it stresses                          | Required behavior                                                            |
| -------------------- | ----------------------------------------- | --------------------------------------------------------------------------- |
| **Low light**        | Recognition under low contrast            | Correct value, or `abstain` (never a confident-wrong money/code).            |
| **Glare / specular** | Washed-out region over a field            | Correct, or `abstain`; if glare hits `ppa`, checksum should repair or flag.  |
| **Skew / perspective** | Detector localization + crop quality    | Box still localizes business-critical fields; value correct or `abstain`.    |
| **Band-over-PPA**    | The colour band overlapping the PPA text  | **PPA disambiguation** still selects the final `= XXX,XX DA`; band still classified for reimbursability. |
| **Rotated lot**      | Vertical `vin` strip (`num_lot`) rotated  | `preprocess.orient` rights it before OCR; `num_lot` reads or `abstain`.      |
| **Degraded print**   | Faded/smudged characters, dot-matrix      | Confusion-map + nomenclature repair the code where possible; else `abstain`. |
| **Vertical dates**   | `date_fab`/`date_exp` in the rotated strip| Oriented + parsed to `%Y-%m`; `date_exp > date_fab` sanity holds.            |
| **Mixed separators** | `,` vs `.` and thousands in one image     | Parsed to the exact `Decimal`; checksum consistent.                          |
| **Orange band**      | Ambiguous reimbursability colour          | `eligible=None`, label "À vérifier" — **abstain, not a guess**.              |
| **Missing field**    | A field absent from the vignette          | `status="missing"`, surfaced; checksum verdict `incomplete` if it's money.   |

The matrix doubles as the **drift regression set**: when MONITORING flags a new
failure mode in production (e.g. a new band variant), the anonymized example is
added as a new row, so the same failure can never silently return.

---

## 6. What "green" means

A change is mergeable only when:

- [ ] `pytest` (CPU suite) is green — parsing goldens, dataset integrity, e2e, schema round-trip.
- [ ] No money path uses `float`; all money assertions are Decimal-string exact.
- [ ] The nomenclature safety tests pass (`ppa`/`tr` untouched; dispensing conflict **flagged, not overwritten**).
- [ ] The CPU core imports and runs **without** `[ml]`; heavy imports fail with the clear remediation message.
- [ ] (When touching models) `pytest -m ml` green, including ONNX↔torch parity.

These gates are the same ones a retrained model must clear before promotion
([`docs/HITL.md`](HITL.md) §6) and before a real-data switchover
([`docs/SWITCHOVER.md`](SWITCHOVER.md) §3).
