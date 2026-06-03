# VignOCR — Future Directions

VignOCR today reads one document type — the pharmaceutical **vignette** — end to
end. But the architecture was chosen so that "read a vignette" generalizes to
"read pharmacy documents." This doc sketches the natural extensions
(**ordonnances**, **handwritten prescriptions**, **supplier documents**, and
broader **pharma document intelligence**) and — more importantly — *why the current
design accommodates them without a rewrite*.

---

## 0. Why the design extends (the load-bearing idea)

Three properties of the existing system are what make new document types cheap:

1. **Modular two-stage *detect-then-read*.** Stage 1 (RF-DETR) localizes
   *whatever regions a schema defines*; Stage 2 (OCR) reads *whatever a region
   contains*. Neither stage hardcodes "vignette." A new document type is, to first
   order, **a new class schema + new field parsers** — the stage boundaries don't
   move. See the pipeline order in [`docs/INTERFACES.md`](INTERFACES.md):
   `preprocess → detect → orient+crop → recognize → parse → correct → assemble`.
2. **Config as the single source of truth.** The 15 classes, roles, regexes,
   thresholds, and correction policy live in `configs/`, not in code. A new
   document is largely a **new config bundle** (a new `classes.*.yaml` +
   `parsing/*.yaml` + a reference/correction policy) consumed through the same
   `vignocr.common` loaders. The contract — `BBox`, `FieldRead`,
   `ExtractionRecord`, the `status`/`source` semantics — is **document-agnostic**
   and is reused verbatim.
3. **Abstain-over-guess + human-in-the-loop, built in.** Every new field type
   inherits confidence/abstention and the [`docs/HITL.md`](HITL.md) correction
   loop. A new document can ship "safe but cautious" (high abstention) on day one
   and *learn* from corrections, exactly as the vignette path does — no separate
   safety machinery to rebuild.

So the recurring pattern for each extension below is the same: **new schema → reuse
detect → reuse/extend read → new parsers + reference → same `ExtractionRecord`,
same abstention, same HITL, same monitoring.**

---

## 1. Ordonnance (prescription) OCR — printed

The nearest neighbor. A printed/typed ordonnance shares the structure problem
(regions: prescriber block, patient block, date, the list of prescribed
medications with dosage/posology/duration) and the *same medical-safety bar*.

What's reused:

- **Detect-then-read** unchanged — just an ordonnance class schema (prescriber,
  patient, date, drug-line, dosage, posology, duration, signature/stamp region).
- **Decimal/date/code parsers** and the **abstention profiles** (selling-grade
  strictness applies directly — a misread posology is unacceptable).
- **A nomenclature-style correction stage** — prescribed product names/DCI matched
  against the **same national nomenclature** already ingested for vignettes, with
  the same *flag-don't-overwrite* policy for dispensing-critical fields.

What's new:

- **Cross-document reconciliation.** The killer feature: reconcile the ordonnance's
  prescribed drug against the **vignette** scanned at dispensing — same DCI? dosage
  match? reimbursable? This is a natural composition of two `ExtractionRecord`s and
  is exactly why both speak the same schema.
- A document-type **router** in front of the pipeline (vignette vs ordonnance) —
  itself a small detection/classification step that selects the config bundle.

---

## 2. Handwritten prescriptions

The hard frontier, and where the design's caution pays off most.

What's reused:

- **Stage 1 detection** localizes the same ordonnance regions whether printed or
  handwritten — region detection is robust to script style.
- **Abstention + HITL** are the whole strategy here: handwriting recognition is
  uncertain, so the system leans hard on **abstain → pharmacist confirms →
  correction becomes training data**. The loop that improves vignette OCR is the
  same loop that bootstraps handwriting from near-zero.
- **Nomenclature correction** is *more* valuable on handwriting: matching a
  scrawled drug name to the canonical reference (structural edit-distance, already
  built) turns an unreadable token into a confident identity — while still flagging
  dispensing-critical disagreements rather than guessing.

What's new:

- A **handwriting-capable recognizer** behind the existing OCR config switch. The
  `ocr/` interface already anticipates a transformer direction (TrOCR/Donut
  scaffold, per [`docs/ROADMAP.md`](ROADMAP.md) Phase 5) — a handwriting model
  slots in as another backend behind `Recognizer`, no pipeline change.
- **Stricter-than-strict abstention** and mandatory human confirmation for any
  dispensing-relevant handwritten field (a policy/config change, not a code change).

---

## 3. Supplier-document OCR (invoices, delivery notes, purchase orders)

Moves from the *selling* world to the *receiving/back-office* world — and the
**receiving abstention profile already exists** (`configs/parsing/fields.yaml →
abstention.receiving`, looser than selling because a human reconciles stock).

What's reused:

- **Detect-then-read** for semi-structured layouts (header: supplier, invoice no,
  date; line items: product, qty, unit price, line total; totals: subtotal, VAT,
  grand total).
- **Decimal-everywhere money + a checksum analogue.** The vignette's
  `prix + shp == ppa` invariant generalizes to invoices'
  `Σ line_totals + VAT == grand_total` and `qty × unit_price == line_total` — the
  *same* "verify · repair-from-two · flag-on-mismatch" engine, pointed at different
  fields via config.
- **Nomenclature matching** to resolve supplier product codes/names to the internal
  catalog — the existing matcher with a different reference table.

What's new:

- **Table/line-item extraction** (repeating rows) — an extension of the detection
  schema to row regions, plus a row-grouping post-process. The `ExtractionRecord`
  generalizes to a header + a list of line-item records.
- **Three-way match** (purchase order ↔ delivery note ↔ invoice) — again a
  composition of multiple `ExtractionRecord`s, the same reconciliation pattern as
  ordonnance↔vignette.

---

## 4. Broader pharma document intelligence

With several document types speaking one schema language and one safety contract,
VignOCR becomes a **pharmacy document-intelligence platform**, not a single OCR:

- **Unified document router + registry.** One entry point classifies the incoming
  document and dispatches to the right config bundle (vignette / ordonnance /
  invoice / …). New types register a bundle; the serving API
  (`/extract`) and the `ExtractionRecord` contract stay stable.
- **Cross-document graph.** Link ordonnance → dispensed vignette → supplier invoice
  for the same product/batch: end-to-end traceability (was this batch prescribed,
  dispensed, reimbursed, and sourced consistently?). Each edge is a reconciliation
  of records that already share keys (`num_enregistrement`, `num_lot`, DCI).
- **One nomenclature, many consumers.** The ingested national reference becomes the
  shared resolver for every document type — versioned once
  ([`docs/DATASET.md`](DATASET.md)), consumed by vignette, ordonnance, and supplier
  correction alike.
- **One monitoring + HITL plane.** Per-field accuracy/confidence, drift, and the
  correction→retrain loop ([`docs/MONITORING.md`](MONITORING.md),
  [`docs/HITL.md`](HITL.md)) are document-agnostic — adding a document type extends
  the dashboards by config, not by new infrastructure.
- **Batch/back-office ingestion.** Beyond the interactive counter, the same
  pipeline runs in bulk on the Narval DAG for archives/audits (the training infra
  doubles as batch-inference infra).

---

## 5. What stays invariant across all of it

No extension is allowed to weaken the core guarantees that make VignOCR trustworthy
— they are properties of the platform, not of the vignette path:

- **Money is `Decimal` end-to-end**, serialized as strings, exact to the centime.
- **Abstain over guess** — every new field type inherits confidence/abstention; a
  human sees anything uncertain.
- **Flag, never silently overwrite** dispensing-/safety-critical fields on a
  confident disagreement (the `configs/nomenclature/correction.yaml` policy
  generalizes to every reference-corrected field).
- **Config-driven, single source of truth** — new document types are new config
  bundles, not hardcoded branches.
- **Heavy ML stays lazy** — the core (parsing/correction/serving) keeps importing
  and running on CPU; new model backends load behind the same lazy-import guard.
- **Human-in-the-loop closes every loop** — every new document type plugs into the
  same capture → review → new dataset version → retrain → promote cycle.

The roadmap to here is incremental: each document type is a schema + parsers +
reference + a recognizer backend, composed onto the **existing** detect-then-read
spine and the **existing** safety/HITL/monitoring planes. That is the whole point
of the two-stage, config-driven design.
