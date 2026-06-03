# VignOCR — Human-in-the-Loop (HITL)

VignOCR is designed to **prefill and confirm**, never to act unattended. The human
at the counter is the final authority, and — crucially — **their corrections are
the training signal that makes the model better.** This doc describes how a
cashier/pharmacist correction at the confirm popup is *captured → anonymized →
queued → reviewed → folded into a new dataset version → retrained*, closing the
loop.

The dataset-side contract (versioning, integrity re-checks) lives in
[`docs/DATASET.md`](DATASET.md) §7; this doc is the end-to-end loop.

---

## 0. Where the human sits in the flow

```
 vignette photo ─► VignOCR /extract ─► ExtractionRecord
                                          │  (per-field value + confidence + status)
                                          ▼
                          ┌──────────────────────────────────┐
                          │  Pharmakon confirm popup          │
                          │  prefilled fields; abstentions     │
                          │  ("à vérifier") highlighted        │
                          └──────────────────────────────────┘
                                          │
                 cashier / pharmacist reviews, edits, confirms
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                                ▼
         business transaction                            HITL correction event
         (sale / stock receipt)                          (only when a field changed
                                                          OR an abstention was resolved)
```

Two abstention profiles gate what the human must look at (from
`configs/parsing/fields.yaml → abstention`): **selling** is stricter
(default τ = 0.90 — a wrong dispense is unacceptable) than **receiving**
(default τ = 0.75). Anything below threshold is shown as `abstain` and **requires**
a human value — which is the highest-value correction signal we get.

---

## 1. Capture — what a correction event is

A correction event is emitted by the Pharmakon front end **only when the human
changed something** relative to what VignOCR proposed, or supplied a value for an
abstained/missing field. We do not log silent confirmations as labels (they are
useful as *accuracy* telemetry — see [`docs/MONITORING.md`](MONITORING.md) — but
not as correction events).

Each event carries exactly what re-annotation needs and nothing more:

| Captured                          | Why                                                                 |
| --------------------------------- | ------------------------------------------------------------------- |
| `field` (a `classes.yaml` name)   | Which field was wrong (routes to the right class on re-label).       |
| `model_value` / `model_status`    | What VignOCR proposed (`FieldRead.value` + `status`).               |
| `human_value`                     | The corrected, ground-truth value.                                  |
| `bbox` (optional, corrected)      | If the *box* was wrong/missing, the human-adjusted COCO box.        |
| `model_confidence`                | To distinguish confident-wrong (worst) from low-conf-corrected.     |
| `flow` (`selling`/`receiving`)    | The abstention profile in effect — affects priority.               |
| `model_versions`                  | `{detector, recognizer, nomenclature_version}` — model↔error lineage.|
| `field_crop` (image patch)        | The pixels for that field — the actual training example.            |
| `correction_kind`                 | `value_fix` · `abstain_resolved` · `box_fix` · `false_field`.       |

These map straight onto the canonical types in
[`docs/INTERFACES.md`](INTERFACES.md): `model_value`/`model_status`/
`model_confidence` come from the `FieldRead` the pipeline returned;
`model_versions` is the `ExtractionRecord.model_versions` dict. The crop is the
detection region the recognizer read — so a captured event **is already a labeled
training example**: `(field_crop, field, human_value[, bbox])`.

> **Active learning by construction.** The only events that exist are the cases the
> model got wrong or was unsure about. Folding these back in trains preferentially
> on the hard distribution, not on the easy cases the model already nails.

### Special cases

- **Checksum-implied corrections.** If the human fixes one of `prix`/`shp`/`ppa`
  and the trio now satisfies `prix + shp == ppa`, all three are captured as a
  consistent money triple (the checksum verdict is recorded for audit).
- **Dispensing conflicts.** When a `flag_on_conflict` field (`dosage`, `forme`) or
  any dispensing-critical field was flagged "à vérifier" and the pharmacist
  resolves it, that is a **high-priority** correction (it touches medical safety).
- **Reimbursability.** A corrected `color_band` colour is captured as a region
  label — these are scarce and disproportionately valuable.

---

## 2. Anonymize — de-identify before it leaves the counter

A vignette photo and the surrounding transaction can carry PII (patient name on an
ordonnance in frame, operator id, store, timestamp). **No raw production sample is
ingested as-is.** Before an event is queued:

1. **Crop to the field region.** Persist only the `field_crop` (the labeled box),
   not the full counter photo, so unrelated content (a prescription, a face, a
   screen) never enters the dataset.
2. **Strip identifiers.** Remove operator/cashier id, patient identifiers, store
   id, and precise timestamps from the event payload. Replace with a coarse,
   non-identifying bucket only where needed for stratified QC (e.g. day-granularity,
   anonymized store cohort).
3. **Hash the linkage.** Keep a one-way hash to de-duplicate and to allow deletion
   on request, never a reversible identifier.
4. **Pseudonymous example id.** Each candidate gets a fresh opaque id; the manifest
   (see §5) references that, not anything traceable to a person.

This is the same de-identification posture the security model expects (see
`docs/SECURITY.md` when present). Consent/retention policy for capturing field
crops is a deployment decision and must be settled before HITL capture is enabled.

---

## 3. Queue — the correction backlog

Anonymized candidates land in a durable **correction queue** (append-only), each in
state `new`. The queue is the single inbox the reviewer works from. Capture is
**decoupled** from review and retraining: the counter never blocks on the dataset,
and a bad day of corrections cannot directly perturb a model.

Prioritization for review (highest first):

1. `selling`-flow dispensing-critical corrections (medical safety).
2. Confident-wrong (`model_confidence` high but `human_value` ≠ `model_value`) —
   these reveal systematic model errors.
3. Business-critical fields (`ppa, prix, shp, num_enregistrement, num_lot`).
4. `abstain_resolved` on any field (fills the model's known blind spots).
5. Everything else.

Each candidate is also tagged with the `model_versions` that produced it, so a
spike of errors can be traced to a specific deployed model.

---

## 4. Review — human gate before the dataset

**Nothing auto-enters the dataset.** A reviewer (typically a senior pharmacist or
a trained annotator) works the queue and, per candidate, chooses:

- **Accept** → the `(field_crop, field, human_value[, bbox])` becomes a confirmed
  label, state `accepted`.
- **Fix** → adjust the value/box (the counter human can be wrong too), then accept.
- **Reject** → not a useful/clean example (e.g. unreadable photo, ambiguous frame),
  state `rejected`, with a reason.

Review enforces the same rules the schema does: the `field` must be a
`classes.yaml` name; money is validated as `Decimal` to the centime; codes/dates
must parse under `configs/parsing/fields.yaml`; `color_band` colour must be one of
the configured colours. A candidate that cannot be made schema-valid is rejected,
not forced in.

> The reviewer is also the **drift early-warning sensor**: a sudden rise in
> confident-wrong corrections for one field (e.g. a new vignette layout) is a
> signal to retrain sooner. This dovetails with [`docs/MONITORING.md`](MONITORING.md).

---

## 5. Fold in — a new dataset version (close the loop)

Accepted corrections are batched and folded into the dataset as a **new version**,
never an in-place edit (see [`docs/DATASET.md`](DATASET.md) §4):

1. Append accepted `(crop, field, value[, box])` examples to the appropriate split.
   New examples default to **train**; a curated, frozen slice is reserved for
   **val/test** so the held-out set tracks the evolving real distribution — chosen
   so it shares **no source frame** with train (the perceptual-hash leakage assert
   guards this).
2. Recompute the **manifest hash** → a new dataset version id. Lineage is
   append-only: `v(hash_n) → v(hash_{n+1})`, with the set of folded-in candidate
   ids recorded.
3. Re-run the **full integrity + QC gate** (`docs/DATASET.md` §5–§6): leakage,
   bbox-validity, names ⊆ schema, business-critical coverage, stats review.

## 6. Retrain — and verify before promotion

1. Launch the Narval DAG on the new dataset version
   (`slurm/submit_all.sh` — see [`docs/SWITCHOVER.md`](SWITCHOVER.md) §2 Step 5).
   The run dir records git SHA + config snapshot + **dataset version hash**, so
   model↔data lineage is exact.
2. Gate the retrained model:
   - Phase 4/5 metric gates on the held-out **real** val/test split (mAP,
     business-critical localization recall, CER, abstention precision).
   - The OCR **regression goldens** in [`docs/TESTING.md`](TESTING.md) still pass
     to the centime/character (no regression on known-good cases).
3. **Promote** only on green. The promoted model's version string flows back into
   `ExtractionRecord.model_versions`, so the *next* batch of corrections is
   attributed to the *new* model — and the loop continues.

```
counter correction → anonymize → queue → review → new dataset version
        ▲                                                   │
        └──────────────── retrain + verify ◄────────────────┘
                         (promote on green)
```

---

## 7. Invariants the loop must never break

- **No silent ingestion.** Every production-derived example passes human review.
- **No PII in the dataset.** Only de-identified field crops + schema-valid labels.
- **Append-only lineage.** Corrections create new dataset versions; history is
  auditable, never overwritten.
- **No leakage from growth.** Newly added examples respect the split-leakage assert
  (perceptual hash), so retrained metrics stay honest.
- **Safety policy is upstream of HITL.** HITL improves the model; it does **not**
  relax the runtime guarantees — `ppa`/`tr` are still never auto-overwritten, and
  dispensing conflicts are still flagged, not guessed
  (`configs/nomenclature/correction.yaml`).
