# VignOCR — Synthetic → Real Switchover

The whole pipeline runs today on a **synthetic fixture** because the real exports
are not annotated to the 15-class schema (see [`docs/DATASET.md`](DATASET.md) §3).
When real annotations land, flipping to them is **one line of config** — but you
do **not** train until the validation checklist below is green.

---

## 1. The one-line change

Switchover is intentionally a single, reversible toggle. Pick **one** of:

**Option A — config (permanent, committed):** edit `configs/data.yaml`:

```yaml
# configs/data.yaml
active: synthetic        # ← change this
```
to
```yaml
active: real
```

**Option B — environment (12-factor, no code change):** leave the file alone and
set the env var (this overrides `active:` at runtime — see
`vignocr.common.config.get_active_dataset`):

```bash
export VIGNOCR_DATA_ACTIVE=real
```

That's it. Everything downstream (loader, validate, stats, detection, ocr,
pipeline) reads the active dataset through `vignocr.common.get_active_dataset()`,
so no other file changes. To roll back, set it to `synthetic` again.

> Prerequisite: the real annotations must already be exported as Roboflow COCO
> (`_annotations.coco.json` per split) into `data/` — the path
> `datasets.real.root: data` in `configs/data.yaml`. The switch only *selects* the
> dataset; it does not create it.

---

## 2. Validation checklist (run before you trust `real`)

Run these **in order**. Do not launch training until every step passes. All
commands assume the repo root (`pharmakon-ocr/`) and `pip install -e .[dev]`.

### Step 0 — point at real data

```bash
export VIGNOCR_DATA_ACTIVE=real          # or set active: real in configs/data.yaml
python -c "from vignocr.common import get_active_dataset as g; d=g(); print(d['name'], d['root'])"
# expect:  real  <abs-path>/pharmakon-ocr/data
```

### Step 1 — split-leakage assert (the one that matters most)

No image — by `file_name` stem **and** perceptual hash — may appear in more than
one split. Roboflow emits 3 augmented variants per source frame; this catches
variants of the same frame straddling train/test.

```bash
python -m vignocr.data.validate --check no_image_leakage_across_splits
```

Pass criterion: **0 leaked images**. (Driven by
`configs/data.yaml → integrity.assert_no_image_leakage_across_splits`.)

### Step 2 — class-coverage assert

Two sub-checks:

1. **Names ⊆ schema** — every COCO category *name* in the export exists in
   `configs/classes.yaml` (the by-name mapping has no orphans).
2. **Business-critical present** — each split covers
   `classes.yaml → business_critical_fields`
   (`ppa, prix, shp, num_enregistrement, num_lot`). On `real` this is the check
   that fails today and must now **pass** (no longer a warning) before training.

```bash
python -m vignocr.data.validate --check class_names_subset_of_schema
python -m vignocr.data.validate --check all_business_critical_present
```

Pass criterion: names is a subset (no unknown categories) **and** all five
business-critical fields appear in train, valid, and test.

### Step 3 — box-validity assert

Every annotation has `0≤x, 0≤y, w>0, h>0` and lies within the image bounds.

```bash
python -m vignocr.data.validate --check every_annotation_valid_bbox
```

Pass criterion: **0 invalid boxes**.

> Steps 1–3 can be run together as the full integrity gate:
> ```bash
> python -m vignocr.data.validate --all          # runs every configs/data.yaml integrity assert
> pytest tests/test_dataset_integrity.py -q       # the same asserts as a test gate
> ```

### Step 4 — stats review (human sign-off)

```bash
python -m vignocr.data.stats                      # per-class counts, box-size dist, split sizes
```

Review per [`docs/DATASET.md`](DATASET.md) §6 QC checklist: every class present in
train and represented in val/test; business-critical classes meet the per-split
floor; no degenerate boxes; `color_band` has a representative green/red/orange mix.
**Record the dataset version (manifest) hash** for run lineage.

### Step 5 — launch the Narval DAG

Only after Steps 1–4 are green:

```bash
# one-time cluster bootstrap (modules, venv, [ml] extra, wheelhouse) — skip if already set up
scripts/setup_narval.sh

# dry-run first: every job must be accepted by the scheduler before real submission
sbatch --test-only slurm/train.sbatch        # repeat per job, or:
slurm/submit_all.sh --dry-run                 # builds the DAG with sbatch --test-only

# submit the full dependency DAG (afterok chaining):
#   validate → train → eval → export(ONNX) → ocr → bench
slurm/submit_all.sh
```

The DAG re-runs `validate` as its **first node** (defense in depth — the gate runs
on the cluster too, not just locally), then trains RF-DETR, evaluates
(mAP@[.5:.95], per-class AP, **business-critical localization recall**), exports
ONNX with a torch-parity check, runs the OCR stage, and benchmarks. Each run dir
records the **git SHA + config snapshot + dataset version hash**.

---

## 3. Exit gates on real data

Switchover is "done" only when the Phase 4/5 gates from
[`docs/ROADMAP.md`](ROADMAP.md) are met on the **real validation split**:

- **Detection:** mAP@.5 ≥ 0.85 overall **and** localization recall = 1.0 on
  `business_critical_fields`.
- **OCR:** field CER ≤ 5% on `business_critical_fields`; abstention precision
  documented.
- The regression goldens in [`docs/TESTING.md`](TESTING.md) still pass.

Until then, keep `active: synthetic` as the safe default and treat `real` as a
candidate under validation.

---

## 4. Rollback

Instant and total — no migration, no state:

```bash
unset VIGNOCR_DATA_ACTIVE        # if you used the env override
# and/or revert configs/data.yaml: active: real -> active: synthetic
```

Because every consumer resolves the dataset through `get_active_dataset()`, the
codebase returns to the synthetic fixture with no other change.
