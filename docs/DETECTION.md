# VignOCR — Stage 1: Detection (RF-DETR medium)

Stage 1 localizes the 15 vignette classes in one forward pass and emits the
`color_band` region for the reimbursability head. It is one of the two independently
deployable ML services (see [ARCHITECTURE.md §6](ARCHITECTURE.md)); the deterministic core
runs without it. This document covers the model choice, the class list, oriented-field
handling, the band-color-preserving augmentation policy, eval metrics, and export.

Module surface (lazy `torch`/`rfdetr` — imported *inside* functions, never at module top):

```python
# detection/  (per docs/INTERFACES.md)
train.run(cfg_path: str, run_dir: Path, resume: Path|None=None) -> Path  # -> best checkpoint
eval.run(ckpt: Path, root: Path, split="valid") -> dict                  # mAP, per-class AP, localization_recall
export.to_onnx(ckpt: Path, out: Path) -> Path                            # + parity check vs torch on fixtures
infer.Detector(ckpt_or_onnx).detect(image) -> list[Detection]            # Detection: name, score, BBox
```

> ML deps come from the `[ml]` extra (`torch==2.4.1`, `torchvision==0.19.1`,
> `rfdetr==1.1.0`, `onnx`, `onnxruntime`, `pycocotools`, `albumentations`,
> `opencv-python-headless`). On `ImportError`, raise a clear message telling the user to
> `pip install -e .[ml]`. The core never needs them.

---

## 1. Why RF-DETR medium for this layout

The vignette is a **fixed-but-photographed** target: the sticker layout is physically
constant (same fields, same relative positions, the diagonal colour band always in the
same place), but each input is a *photograph* — phone camera, uneven lighting, perspective
skew, glare, partial occlusion, and background clutter from the medicine box. The detector
must be robust to the photographic variation while exploiting the strong layout prior.

**RF-DETR** (a real-time DETR-family detector) fits this better than anchor-based
one-stage detectors (YOLO family) or two-stage detectors (Faster R-CNN):

- **Global attention suits a layout prior.** DETR's set-based attention reasons over the
  whole image at once, so it learns the *spatial relationships* between fields (PPA above
  the band, the `vin` strip down one side) rather than scoring boxes from local features
  alone. On a fixed layout this is exactly the right inductive bias and it generalizes
  across perspective skew without re-tuning.
- **No anchor/NMS tuning.** DETR predicts a fixed set of objects directly — no anchor
  scales/ratios to hand-tune for our mix of tiny fields (a date) and large regions
  (`entete`), and no NMS threshold that has to be re-balanced when fields sit close
  together. With only 15 known classes and ≤ a few dozen boxes per image, the set-based
  head is a clean match.
- **Real-time variant for serving.** RF-DETR targets deployable latency and exports to
  ONNX/TensorRT cleanly, which is what Stage 1 needs as a live service.

**Why "medium" and not small/large:**

- The label inventory is *small and fixed* (15 classes) but includes **small,
  business-critical** fields (a date, a lot number) where under-capacity hurts
  localization recall — the metric we cannot miss on (§5). Small/nano backbones tend to
  drop exactly these.
- "Large" buys marginal mAP on such a constrained layout while costing export size and
  per-frame latency in a service that runs on every scan. **Medium** is the capacity/latency
  knee for this problem.
- Empirically (to be confirmed on real val, §5) the medium variant is expected to clear
  the gate (mAP@.5 ≥ 0.85 overall **and** localization recall = 1.0 on business-critical
  fields) with serving latency that fits an interactive prefill flow. The exact variant is
  config-driven; if real-val numbers say otherwise, it is a one-line change, not a rewrite.

The class count and head size derive from `configs/classes.yaml` (`num_classes`,
ordered `names`) via `vignocr.common.get_classes()` — never hardcoded.

---

## 2. Classes the detector localizes

All 15 come from [`configs/classes.yaml`](../configs/classes.yaml). The COCO loader maps
each annotation to a class **by name** (via each file's own `categories` array), so the
head stays robust to whatever ids the real Roboflow export assigns.

| id | name | type | orientation | notes |
|----|------|------|-------------|-------|
| 0 | `ppa` | money | horizontal | business-critical |
| 1 | `prix` | money | horizontal | business-critical |
| 2 | `shp` | money | horizontal | business-critical |
| 3 | `num_enregistrement` | code | horizontal | business-critical · nomenclature anchor |
| 4 | `num_lot` | code | **vertical** | business-critical · rotated, inside `vin` |
| 5 | `date_fab` | date | **vertical** | rotated, inside `vin` |
| 6 | `date_exp` | date | **vertical** | rotated, inside `vin` |
| 7 | `product_name` | text | horizontal | identity |
| 8 | `dci` | text | horizontal | identity, dispensing |
| 9 | `dosage` | text | horizontal | identity, dispensing |
| 10 | `forme` | text | horizontal | identity, dispensing |
| 11 | `laboratoire` | text | horizontal | identity |
| 12 | `entete` | region | horizontal | structural body region |
| 13 | `vin` | region | **vertical** | structural rotated strip (contains lot + dates) |
| 14 | `color_band` | region | **diagonal** | reimbursability — colour, not text |

**Business-critical** (`classes.yaml: business_critical_fields`):
`[ppa, prix, shp, num_enregistrement, num_lot]` — these gate eval (§5).

The detector localizes `color_band` like any other class; the *colour* of that region is
then read by the separate reimbursability head (§7), not by OCR.

---

## 3. Oriented / rotated field handling

Four classes are printed **sideways** inside the vertical `vin` strip
(`classes.yaml: roles.rotated_fields = [num_lot, date_fab, date_exp, vin]`). The detector
itself predicts **axis-aligned COCO `xywh` boxes** — the box tightly encloses the rotated
text, and the *orientation* needed to read it comes from the class's `orientation`
attribute in the schema (`horizontal` / `vertical` / `diagonal`), not from a rotated-box
regression head.

This keeps Stage 1 simple and the COCO contract intact:

1. **Detection** returns an axis-aligned `BBox` per field.
2. **Crop** (pipeline) cuts the region.
3. **Orientation correction** (`ocr/preprocess.orient(crop, orientation)`) rotates the
   crop upright *before* OCR, driven by the schema attribute.

So "oriented field handling" is a **crop-then-rotate** contract between detection and OCR,
config-driven per class — not an oriented-bounding-box model. The `color_band` is
`diagonal`; it is detected as an axis-aligned region and consumed by the colour head, which
does not require de-skewing to a baseline.

> If real annotations later show rotated text that an axis-aligned box encloses too
> loosely (lots of background in the crop), the upgrade path is an oriented-box head — but
> the synthetic-first reality and the current schema commit to axis-aligned + schema-driven
> rotation, and that is what ships today.

---

## 4. Band-color-preserving augmentation

Augmentation must improve robustness to the photographic nuisance variables (lighting,
perspective, blur, noise) **without ever changing the semantics of the `color_band`**. The
band's colour *is* the CHIFA reimbursability signal (`green` = remboursable, `red` =
non-remboursable, `orange` = à vérifier), so any transform that could flip green↔red would
silently corrupt the label.

### Allowed (geometry + band-safe photometrics)
- Affine / perspective warp, small rotation, scale, translation, horizontal flip *only if
  it does not invert layout semantics* — geometry does not touch hue.
- Mild brightness/contrast/gamma, Gaussian/ISO noise, motion/defocus blur, JPEG
  compression, glare/shadow simulation — these mimic camera reality and leave the *hue*
  family intact.
- Cutout/coarse-dropout away from the band, background augmentation.

### Forbidden (and why)
- **Hue rotation / `HueSaturationValue` hue shift** — can rotate green into red/orange.
  ⛔ Inverts the reimbursability label.
- **Random channel shuffle / RGB↔BGR swaps** — green and red channels trade places.
  ⛔ Same failure.
- **Channel-wise inversion, aggressive colour jitter, grayscale on the band** — destroys or
  flips the colour signal the head depends on.
- Anything that makes a `green` band classify as `red` (or vice-versa) on the augmented
  copy while the label still says the original colour.

> Rule of thumb encoded in [ARCHITECTURE.md §6](ARCHITECTURE.md) and enforced both in
> preprocessing and training augmentation: **geometry — yes; hue/channel ops that can flip
> green↔red — no.** Saturation/brightness nudges that *preserve* hue ordering are fine; hue
> *rotation* is not. (Augmentation uses `albumentations`, lazy-imported.)

This constraint is unusual for a detector — most pipelines colour-jitter freely — and it is
the single most important detection-side rule for correctness here.

---

## 5. Evaluation metrics

`eval.run(ckpt, root, split) -> dict` reports the standard COCO suite **plus** the
business rule's own gate (lazy `pycocotools`):

| Metric | What it measures | Why |
|--------|------------------|-----|
| **mAP@[.5:.95]** | COCO primary AP averaged over IoU 0.50–0.95 | overall localization quality |
| **mAP@.5** | AP at IoU 0.50 | headline gate threshold |
| **Per-class AP** | AP for each of the 15 classes | catches a single weak class hidden by the mean (e.g. the diagonal band, a tiny date) |
| **Business-critical localization recall** | recall on `[ppa, prix, shp, num_enregistrement, num_lot]` | 🛑 the business rule depends on these being *found* |

**Localization recall** is the metric that matters most here and it is reported
separately because the **business rule depends on it**: if Stage 1 fails to *find* the PPA
or the lot number, the field is `missing` and a human must enter it — the downstream
prefill is only as good as the boxes Stage 1 produces. A model can have a fine mean mAP
while missing a business-critical class; per-class AP + business-critical recall surface
that.

**Gate (real data, from [ROADMAP.md](ROADMAP.md) Phase 4):**
`mAP@.5 ≥ 0.85` overall **and** **localization recall = 1.0** on `business_critical_fields`.

On the **synthetic fixture today**, eval runs as a CPU smoke check (tiny image counts from
`configs/data.yaml`); the headline numerical gate is meaningful only once the real,
15-class-annotated val set exists ([ROADMAP.md](ROADMAP.md) Phase 9). Reporting machinery
and the business-critical recall computation work now so the gate is ready the day real
data lands. Eval seeds via `vignocr.common.seed_everything` for reproducibility.

---

## 6. Training

`train.run(cfg_path, run_dir, resume=None) -> Path` (best checkpoint). Config-driven,
seeded, and **resumable from checkpoint**. Class list/order, dataset root/splits, and
hyperparameters all come from configs via `vignocr.common`; nothing hardcoded.

- **Compute.** Real training runs on **Narval HPC** (Slurm, A100) — see
  [ROADMAP.md](ROADMAP.md) Phase 7. A CPU smoke path trains a few steps on the synthetic
  fixture to prove the loop end-to-end without a GPU.
- **Provenance.** Each run dir records the git-SHA + a snapshot of the configs used, so
  every checkpoint is traceable to the exact contract that produced it; the resulting
  identifier surfaces in `ExtractionRecord.model_versions["detector"]`.
- **Augmentation.** Band-color-preserving only (§4).
- **Data integrity.** Training only consumes a dataset that passes
  `data/validate.check_integrity` (no split leakage, valid boxes, class-names ⊆ schema,
  business-critical coverage) per `configs/data.yaml: integrity`.

---

## 7. Reimbursability head (colour, not OCR)

The detector provides the `color_band` region; a colour classifier maps its dominant hue
to CHIFA eligibility per `classes.yaml: reimbursability`:

| Colour | `eligible` | Label |
|--------|-----------|-------|
| `green` | `true` | Remboursable (CHIFA) |
| `red` | `false` | Non remboursable |
| `orange` | `null` | À vérifier (variant on some vignettes → **abstain**) |

This is a **distinct signal from text recognition** — no characters are read from the band.
The result is emitted as a `Reimbursability` object on the record. `orange` (and any
low-confidence/unknown read) abstains rather than forcing a yes/no — consistent with the
system-wide "never guess" invariant. The band-color-preserving augmentation policy (§4)
exists precisely so this head's labels stay valid through training.

---

## 8. ONNX / TensorRT export & deployment

`export.to_onnx(ckpt, out) -> Path` exports the trained detector to **ONNX** and runs a
**parity check** against the torch model on fixture inputs (the ONNX outputs must match the
torch outputs within tolerance) before the artifact is accepted. ONNX is the portable
serving format; on NVIDIA serving hardware the ONNX graph is further compiled to
**TensorRT** for lower-latency inference. (`onnx==1.17.0`, `onnxruntime==1.20.1`,
lazy-imported.)

`infer.Detector(ckpt_or_onnx)` accepts **either** a torch checkpoint **or** an exported
ONNX model, so the same `detect(image) -> list[Detection]` API serves training-time
evaluation and production inference.

### Deployment shape
- Stage 1 is an **independently deployable** GPU service behind the stateless API tier
  ([ARCHITECTURE.md §6](ARCHITECTURE.md)): it scales and versions separately from the OCR
  service. Re-export the detector without touching the recognizer (and vice-versa).
- Only **exported ONNX artifacts** (plus their git-SHA/config provenance) are shipped from
  the Narval training tier to serving — never raw training checkpoints.
- The artifact identity flows into `ExtractionRecord.model_versions["detector"]`, so every
  prediction is traceable to the exact model that produced it.
- **Stub for CPU today.** When `[ml]` is absent, the pipeline uses a deterministic fixture
  detector so the end-to-end flow and tests run on CPU — the real ONNX/TensorRT detector
  drops into the same `Detector.detect` seam unchanged.

---

## 9. Cross-references

- [ARCHITECTURE.md](ARCHITECTURE.md) — full pipeline, deployment topology, safety invariant.
- [OCR.md](OCR.md) — Stage 2 consumes the crops Stage 1 produces (incl. the orientation
  contract from §3).
- [INTERFACES.md](INTERFACES.md) — `Detection`, `BBox`, and the `detection/` API.
- [`configs/classes.yaml`](../configs/classes.yaml) — class list, orientations,
  business-critical set, reimbursability colours (single source of truth).
- [ROADMAP.md](ROADMAP.md) Phase 4 — detection entry/exit gates.
