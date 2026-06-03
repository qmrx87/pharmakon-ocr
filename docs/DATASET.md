# VignOCR — Dataset

How the data layer is organized, the COCO standard it speaks, the **gap between
the real exports and the 15-class target** (and what to do about it), how datasets
are versioned and integrity-checked, the QC bar, and how the dataset *grows* from
human corrections over time.

> Single source of truth: every class name, role, and integrity flag below comes
> from `configs/classes.yaml` and `configs/data.yaml`. This doc never re-defines
> them — it explains how they are used.

---

## 1. Organization

Two dataset *channels*, selected by `active:` in `configs/data.yaml` (or the
`VIGNOCR_DATA_ACTIVE` env var):

```
fixtures/synthetic/        ← active: synthetic  (generated, CPU dev/test — default)
  train/  _annotations.coco.json  + images
  valid/  _annotations.coco.json  + images
  test/   _annotations.coco.json  + images

data/                      ← active: real       (Roboflow COCO export — NOT yet usable, see §3)
  train/  _annotations.coco.json  + images
  valid/  _annotations.coco.json  + images
  test/   _annotations.coco.json  + images
```

- **`synthetic`** is generated deterministically by `src/vignocr/data/synthetic.py`
  (`generate: true`, `seed: 1337`, `num_images: {train:12, val:4, test:4}`,
  `image_size: [640,640]`). It draws vignette-like images with valid COCO boxes
  for **all 15 classes**, so the full downstream pipeline is exercised before any
  real label exists. It is regenerable on demand and never needs to be committed
  as binary blobs.
- **`real`** points at `data/` — the Roboflow export. It exists in the repo but is
  **not** annotated to our schema (see §3). `generate: false`.

Split directory names are configured, not assumed: `splits: {train: train,
val: valid, test: test}` — note Roboflow's `valid` (not `val`) on disk.

The **nomenclature reference** is a separate artifact, not part of the COCO
dataset:
- `NOMENCLATURE-VERSION-FEVRIER-2026-.xlsx` — the real national reference (Feb 2026).
- `fixtures/nomenclature.csv` — the synthetic fixture used now. Ingest the real
  xlsx → csv with `scripts/ingest_nomenclature.py` (Phase 3).

---

## 2. The Roboflow COCO standard in use

The dataset speaks **COCO object-detection JSON**, one `_annotations.coco.json`
per split directory — the Roboflow export convention (`coco_filename:
_annotations.coco.json` in `configs/data.yaml`). Each file has the standard
`images` / `annotations` / `categories` arrays; boxes are COCO **`[x, y, w, h]`
in pixels** (matching `schemas.BBox`).

Roboflow's preprocessing (per the bundled `data/README.roboflow.txt`):

- **Auto-orientation** with EXIF-orientation stripped.
- **Resize to 640×640** (stretch) — matches `image_size: [640, 640]`.
- Augmented to 3 versions per source image (flips, 90° rotations, crop, ±15°
  rotation, brightness, salt-and-pepper noise).

> ⚠️ **Augmentation caveat that touches our schema.** Roboflow's vignette export
> applies horizontal/vertical flips and 90° rotations *at export time*. For our
> task that is dangerous in two ways the re-annotation and our own training
> augmentation must respect:
> 1. **Band-colour semantics** must be preserved — never apply hue/channel ops
>    that could flip green↔red (`color_band` drives CHIFA eligibility). Our
>    detection augmentation is **band-colour-preserving** (see
>    `docs/INTERFACES.md` → `detection/`).
> 2. **Orientation** of the `vin` strip (`num_lot`, `date_fab`, `date_exp`) is
>    semantically vertical; arbitrary 90° flips destroy the orientation prior the
>    OCR stage relies on. Re-annotation should be done on **auto-oriented,
>    un-flipped** source frames where possible.

### Category → class mapping is **by name**, never by id

The COCO loader (`src/vignocr/data/coco.py`, `load_split`) maps each annotation to
a training class **by the category *name*** found in that file's own `categories`
array — never by a hardcoded numeric id. This is deliberate: Roboflow assigns
whatever category ids it likes per export (e.g. `date_info=0, entete=1, vin=2`
today), and our contiguous training ids (`0..14`) live only in
`configs/classes.yaml`. Mapping by name keeps the loader robust across re-exports
and is what makes the integrity assert "every category name ∈ schema" meaningful.

---

## 3. The reality: real exports ≠ the 15-class target (re-annotation required)

The 15-class target schema (`configs/classes.yaml`) is:

```
12 fields  : ppa, prix, shp, num_enregistrement, num_lot, date_fab, date_exp,
             product_name, dci, dosage, forme, laboratoire
3 auxiliary: entete (body region), vin (rotated strip region), color_band (reimbursability)
```

**Inspected 2026-05**, the real exports in the repo carry a *completely different*
and far coarser label set:

| Export   | Roboflow project        | Images (train/valid/test) | Categories present              |
| -------- | ----------------------- | ------------------------- | ------------------------------- |
| `data/`  | *vignette* v2           | 1647 / … / …              | `date_info`, `entete`, `vin`    |
| `data2/` | *Algeria-Drug-label* v3 | 1626 / 154 / 77           | `drug-labels`, `text`           |

Reconciliation against the target:

- `entete` and `vin` **do** exist in our schema (ids 12, 13) — useful structural
  regions, directly reusable.
- `date_info` is a *coarse region*, not our three precise date/lot fields
  (`date_fab`, `date_exp`, `num_lot`). It is a hint, not a label.
- `drug-labels` / `text` (from `data2/`) are generic detection classes with **no
  field semantics** at all — not a subset of our schema.
- **None of the business-critical fields** (`ppa`, `prix`, `shp`,
  `num_enregistrement`, `num_lot`) are annotated in either export.

**Conclusion: re-annotation is required.** The raw images are valuable (real
vignettes, real lighting/skew/glare), but the *labels* must be redrawn to the
15-class schema before training is meaningful. Until then, `active: synthetic`
stays, and the integrity assert `assert_all_business_critical_present` is treated
as a **warning** on `real` (it cannot pass until the fields are annotated).

### Re-annotation plan

1. **Class set = `configs/classes.yaml` names, verbatim.** Annotators draw exactly
   these names so the by-name mapping and the subset assert pass on import.
2. **Source frames:** use the un-flipped, auto-oriented originals; keep the `vin`
   strip's natural vertical orientation.
3. **Reuse where valid:** carry over `entete`/`vin` boxes from `data/` as
   pre-labels to speed annotation; treat `date_info` as a *region hint* to focus
   the annotator on `date_fab`/`date_exp`/`num_lot` — never auto-promote it.
4. **Business-critical first:** prioritize `ppa`, `prix`, `shp`,
   `num_enregistrement`, `num_lot` coverage in every split (the gates depend on
   them).
5. **Export** as Roboflow COCO (`_annotations.coco.json` per split) into `data/`,
   then run the SWITCHOVER checklist (`docs/SWITCHOVER.md`).

---

## 4. Versioning — manifest-hash

A dataset *version* is identified by a **content hash of its manifest**, not by a
mutable tag. The manifest enumerates, per split, every image `file_name` + its
content hash and the full annotation set (boxes + category *names*); the dataset
version id is the hash of that manifest (e.g. `sha256[:12]`).

Why hash-based:

- **Reproducibility:** a training run records the dataset version hash alongside
  the git SHA and the config snapshot in its run dir. The exact data is
  recoverable from the id.
- **Tamper-evidence:** any change to an image or a box changes the hash, so "what
  data was this model trained on?" is always answerable and never ambiguous.
- **HITL growth (see §7):** each fold-in of human corrections produces a **new**
  version hash — an append-only lineage `v(hash_0) → v(hash_1) → …`, never an
  in-place edit.

The synthetic fixture is itself versioned this way: because generation is seeded,
the same `seed`/`num_images`/`image_size` reproduce the same manifest hash, so
"regenerate" is verifiable, not hand-wavy.

---

## 5. Split-integrity asserts (run on every load)

`src/vignocr/data/validate.py::check_integrity` enforces the flags in
`configs/data.yaml → integrity` every time a dataset is loaded. These are
**asserts, not suggestions** — a failing dataset does not silently train:

| Flag (`configs/data.yaml`)               | What it guarantees                                                                                  |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `assert_no_image_leakage_across_splits`  | No image appears in more than one split — by `file_name` **stem** *and* **perceptual hash** (catches Roboflow's augmented near-duplicates of the same source frame straddling train/test). |
| `assert_every_annotation_valid_bbox`     | Every box has `0≤x, 0≤y, w>0, h>0` and lies within the image bounds.                               |
| `assert_class_names_subset_of_schema`    | Every COCO category *name* exists in `classes.yaml` (the by-name mapping has no orphans).          |
| `assert_all_business_critical_present`   | Each split covers all `business_critical_fields`. **Warn (not fail) on `real`** until annotated.   |

The **perceptual-hash leakage check is the one that matters most here**: Roboflow
emits 3 augmented variants per source image; if those variants land in different
splits, val/test metrics are inflated by memorization. The stem+phash check is
what makes a reported mAP trustworthy.

---

## 6. QC — quality control

Beyond the hard asserts, `src/vignocr/data/stats.py::summarize` produces the
review surface used to sign off a dataset version:

- **Per-class annotation counts** (per split and overall) — catch under-represented
  fields before they tank recall. Business-critical classes get explicit attention.
- **Box-size distribution** per class — tiny boxes (e.g. a cramped `num_lot`) flag
  recognition risk; absurd boxes flag annotation errors.
- **Per-split sizes** and the train/val/test ratio — confirm a sane split.
- **Class balance** across splits — every class should appear in train *and* be
  represented in val/test.

QC checklist before promoting a dataset version to "trainable":

- [ ] All four integrity asserts pass (business-critical: pass on real, not just warn).
- [ ] Every class has annotations in **train** and is present in **val** and **test**.
- [ ] Business-critical classes meet a minimum per-split count (set the floor in review).
- [ ] No degenerate boxes (size outliers reviewed).
- [ ] `color_band` annotated with the correct colour semantics on a representative
      mix of green/red/orange (reimbursability cannot be learned otherwise).
- [ ] Stats reviewed and the dataset version hash recorded.

---

## 7. HITL growth / retraining strategy

The dataset is **not static** — it grows from production. The detail of capture,
anonymization, and review lives in [`docs/HITL.md`](HITL.md); the *dataset-side*
contract is:

1. **Corrections become candidate labels.** When a cashier/pharmacist corrects a
   field at the confirm popup (or confirms an abstention's true value), that
   (image-crop, field, corrected value, optional corrected box) is captured as a
   labeled example — exactly the cases the current model got wrong or was unsure
   about (active-learning by construction).
2. **Anonymize + review.** Candidates are de-identified and pass human review
   before entering the dataset (no auto-ingest of raw production data).
3. **Append a new version, never edit in place.** Reviewed candidates are folded
   into a new dataset version → a **new manifest hash** (§4). The lineage is
   append-only and auditable.
4. **Re-run integrity + QC** (§5–§6) on the new version, then retrain via the
   Narval DAG. The run dir records the new dataset hash, so model↔data lineage is
   exact.
5. **Close the loop.** The retrained model is evaluated against the held-out test
   split (which must remain leakage-free w.r.t. the newly added data — the phash
   check guards this) and the regression goldens in [`docs/TESTING.md`](TESTING.md)
   before promotion.

This is how VignOCR improves on *the vignettes it actually sees* without ever
trusting an unreviewed production sample.

---

## See also

- [`docs/SWITCHOVER.md`](SWITCHOVER.md) — the one-line synthetic→real switch + the
  validation checklist to run when real annotations arrive.
- [`docs/HITL.md`](HITL.md) — the human-correction → retraining loop in full.
- [`docs/TESTING.md`](TESTING.md) — dataset-validation tests + the field-capture
  test matrix.
- [`docs/INTERFACES.md`](INTERFACES.md) — `data/` module signatures and
  `CocoSplit`/`Crop` types.
