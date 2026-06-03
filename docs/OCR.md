# VignOCR — Stage 2: OCR Recognition

Stage 2 reads text from the per-field crops Stage 1 produced. It is the second
independently deployable ML service (see [ARCHITECTURE.md §6](ARCHITECTURE.md)); the
deterministic parsing/checksum/nomenclature core that consumes its output runs without it.
This document covers the baseline-vs-transformer recognizer choice (with an evidence-based
recommendation and how to decide on val), per-field-type recognition and preprocessing,
confidence scoring and abstention, and the validation hooks that turn raw reads into
trustworthy fields.

Module surface (lazy OCR backend — imported *inside* functions, never at module top):

```python
# ocr/  (per docs/INTERFACES.md)
infer.Recognizer(cfg).read(crop, field_type, orientation) -> FieldRead  # confidence + status (abstain < τ)
preprocess.orient(crop, orientation) -> crop                            # rotate vertical vin fields upright
train.run(cfg_path, run_dir) -> Path
eval.run(...) -> dict                                                   # CER, abstention precision
```

> ML deps come from the `[ml]` extra (`paddleocr==2.9.1`, `paddlepaddle==2.6.2`,
> `opencv-python-headless`; TrOCR/Donut via `torch`/`torchvision` if the transformer
> direction is selected). On `ImportError`, raise a clear message telling the user to
> `pip install -e .[ml]`. The core never needs them. When `[ml]` is absent the pipeline
> uses a deterministic fixture recognizer so the end-to-end flow runs on CPU today.

---

## 1. Baseline vs transformer — recommendation

Two families are in scope. Both sit behind the same `Recognizer.read(...)` seam and are
selected by a **config switch**, so the choice is reversible and made on val numbers, not
taste.

| | **Baseline — PaddleOCR / CRNN** | **Transformer — TrOCR / Donut** |
|---|---|---|
| Approach | CNN + sequence head (CTC), per-crop line recognition | encoder-decoder; TrOCR = crop→text, Donut = layout→structured |
| Data hunger | trains/fine-tunes well on **small** datasets | needs more labeled data; shines with scale |
| Latency / size | light, fast, easy ONNX export — fits an interactive prefill loop | heavier; higher per-crop latency and export size |
| Robustness | strong on short fixed-charset fields (money, dates, codes) | strongest on long/cursive/free text and noisy layouts |
| Confidence | per-token CTC probabilities → easy field-level score | sequence likelihood; calibration needs more care |
| Fit to *our* fields | excellent: most fields are **short, fixed-format** (money, `%m/%Y` dates, `AA/BB/CC<LETTER>DDD/EEE` codes) | overkill for short codes; advantage is on `product_name`/`laboratoire` free text |

### Recommendation

> **Ship the PaddleOCR/CRNN baseline first; keep a TrOCR/Donut scaffold behind a config
> switch and only promote it if val evidence justifies it.**

The evidence-based reasoning, given the **synthetic-first reality** and the real field mix:

1. **Our fields are mostly short and fixed-format.** Money (`Decimal`), dates (`%m/%Y`),
   and the registration/lot codes are exactly the regime where CRNN-style recognizers are
   strong and where a transformer's extra capacity buys little. A per-field-type charset
   (§3) further narrows the problem.
2. **Data is scarce today.** There is **no real 15-class-annotated OCR dataset yet**
   ([ROADMAP.md](ROADMAP.md) Phase 9); we train/prove on a synthetic fixture. Transformers
   are the more data-hungry family; the baseline reaches a useful operating point with far
   less labeled text.
3. **Latency and export.** Stage 2 runs on every scan and exports to ONNX for serving; the
   lighter baseline fits an interactive prefill flow and exports cleanly.
4. **Calibrated confidence is first-class here.** The whole system hinges on honest
   per-field confidence to drive abstention (§4). CTC token probabilities give a
   straightforward, well-understood field-level score; transformer sequence likelihoods
   need more calibration work to be trustworthy at the same bar.

The transformer direction is **not dismissed** — it is the natural upgrade for the
free-text identity fields (`product_name`, `laboratoire`, `dci`) and for noisy real-world
photos, which is why it stays scaffolded behind the switch.

### How to choose on val

Decide per the gate, not a priori. Run both behind the same seam on the **same val split**
and compare with `eval.run(...)`:

- **Primary gate — field CER on business-critical fields** (`[ppa, prix, shp,
  num_enregistrement, num_lot]`): target **CER ≤ 5%** ([ROADMAP.md](ROADMAP.md) Phase 5).
  Pick the family that clears it; if both clear, prefer the lighter/faster one (baseline).
- **Abstention precision** at the configured τ (§4): the recognizer that abstains *more
  precisely* (fewer wrong values slipping through above τ, fewer needless abstentions
  below it) wins on the safety axis.
- **Per-field-type breakdown.** It is legitimate to **route by field type** — e.g. baseline
  for money/code/date, transformer for free-text identity — if val shows the transformer
  meaningfully beats the baseline on `product_name`/`laboratoire` while the baseline wins on
  the short fixed fields. The `field_type` argument to `read(...)` makes this routing
  natural.
- **Latency budget.** Whichever family is chosen must still fit the interactive
  prefill-and-confirm latency target after ONNX export.

Document the chosen family + per-field routing and the val numbers that justified it in the
run record; the decision is config, reproducible, and revisited when real data lands.

---

## 2. Orientation correction comes first

Before recognition, `preprocess.orient(crop, orientation)` rotates the crop upright. Four
classes are printed **vertically** inside the `vin` strip
(`classes.yaml: roles.rotated_fields = [num_lot, date_fab, date_exp, vin]`); orientation is
read from each class's schema attribute (`horizontal` / `vertical` / `diagonal`), so the
correction is config-driven, not hardcoded per field. Feeding a sideways crop to a
horizontal-text recognizer is a primary, avoidable error source — this is the crop-then-
rotate contract Stage 1 sets up ([DETECTION.md §3](DETECTION.md)).

---

## 3. Per-field-type recognition + preprocessing

`read(crop, field_type, orientation)` routes by the field's `type` from
[`configs/classes.yaml`](../configs/classes.yaml). Each type gets a tailored charset/decode
and matching preprocessing — narrowing the hypothesis space is the cheapest accuracy gain
available.

| `field_type` | Fields | Recognition bias | Preprocessing |
|---|---|---|---|
| **money** | `ppa`, `prix`, `shp` | digits + `, . ` separators + currency tokens (`DA`/`DZD`); decode favours numeric runs | binarize, normalize height, strip background; keep separators crisp |
| **code** | `num_enregistrement`, `num_lot` | uppercase alphanumeric + `/ - .`; apply OCR confusion map on digit slots | high-contrast threshold; preserve glyph separation so `O/0`, `I/1`, `S/5`, `B/8`, `Z/2` are distinguishable |
| **date** | `date_fab`, `date_exp` | digits + date separators (`/ - .`) | upright (vertical in `vin`); normalize; denoise small print |
| **text** | `product_name`, `dci`, `dosage`, `forme`, `laboratoire` | full charset incl. accents; length-tolerant | deskew, denoise, contrast; preserve diacritics |

The recognizer emits the **raw** string into `FieldRead.raw`; canonicalization (Decimal
money, `%Y-%m` dates, normalized codes) is done downstream by `parsing/` (§5), not by the
recognizer. The recognizer's only jobs are an accurate `raw` read and a calibrated
`confidence`.

The confusion map for codes lives in
[`configs/parsing/fields.yaml`](../configs/parsing/fields.yaml)
(`num_enregistrement.confusion_map: {O:"0", I:"1", l:"1", S:"5", B:"8", Z:"2"}`) and is
applied on digit slots during code parsing — so OCR's most common digit/letter confusions
are repaired deterministically rather than hoped away.

---

## 4. Confidence scoring + abstention

Every read carries a `confidence` (0..1). The recognizer compares it to the active
threshold **τ** and sets `status`:

- `confidence ≥ τ` → `status="ok"`, value flows to parsing.
- `confidence < τ` → `status="abstain"` (à vérifier) — the field is surfaced blank-with-a-
  flag, **never** a silently-guessed value, and its name is added to
  `ExtractionRecord.abstentions`.

τ is **not** a constant — it is selected by the request `flow` from
[`configs/parsing/fields.yaml`](../configs/parsing/fields.yaml):

```yaml
abstention:
  selling:   { default: 0.90 }   # 🛑 stricter — a wrong dispense is unacceptable
  receiving: { default: 0.75 }
  # Per-field overrides allowed, e.g. selling: { num_lot: 0.92 }
```

**Selling is strictly stricter than receiving.** Dispensing the wrong drug/dose/lot is a
patient-safety event, so the selling flow demands higher confidence and abstains on a
*superset* of the fields receiving would; goods-receipt tolerates more autofill because
errors are caught downstream and never dispensed. This is the recognizer-side expression of
the system-wide safety invariant — **OCR never auto-commits; it prefills and a human
confirms** ([ARCHITECTURE.md §1](ARCHITECTURE.md)). Per-field overrides attach in the same
block (e.g. an even stricter bar on `num_lot` when selling).

### Confidence must be calibrated

The threshold is only meaningful if `confidence` is calibrated — i.e. a 0.90 actually
corresponds to ~90% correctness. Eval therefore reports **abstention precision**
([ROADMAP.md](ROADMAP.md) Phase 5): of the fields the recognizer *did not* abstain on, how
often was the value correct. A model that is overconfident (high confidence on wrong reads)
fails the safety bar even with a good CER, and the choice of recognizer family (§1) weighs
this explicitly.

---

## 5. Validation hooks (deterministic, post-OCR)

A raw read is not trusted on its own — it must survive the deterministic checks in
`parsing/` (pure CPU, no ML). These hooks both *correct* and *catch*, and they run even
when the recognizer is stubbed.

1. **Regex / format validation.** Each field is validated against its pattern in
   `configs/parsing/fields.yaml`:
   - `num_enregistrement` → `^AA/BB/CC<LETTER>DDD/EEE$` with the known letter blocks
     (`D F G E H P R S A B C T`), normalized to `{a}/{b}/{c}{letter}{d}/{e}`.
   - `num_lot` → `^[A-Z0-9][A-Z0-9\-/.]{1,18}$`, uppercased.
   - money → the locale-aware numeric regex, parsed to **`Decimal`** (never float),
     centime-quantized, bounded to `[0.00, 100000.00]` (out of range → abstain).
2. **Date sanity.** Dates parse the accepted formats and emit `%Y-%m`; **`date_exp` must be
   after `date_fab`** — a violation is a contradiction the human is asked to resolve, not a
   value to accept.
3. **Decimal checksum — `prix + shp == ppa`, exact to the centime.** The strongest accuracy
   lever: if two of the three money fields are confident (≥ `min_conf_to_anchor = 0.80`),
   the third is **recomputed** (verdict `repaired`, `source="checksum"`); on a confident
   three-way disagreement the verdict is `mismatch` and the result is **flagged, never
   silently accepted** (`parsing/checksum.verify_and_repair(...)`). This lets a correct PPA
   rescue a misread `prix`, and catches an OCR error that *looks* plausible but breaks the
   arithmetic.
4. **Final-PPA disambiguation.** When PPA appears twice (intermediate `PPA: 700,06+2,50`
   vs final `PPA = 702,56 DA`), the parser deterministically selects the **final**
   `= XXX,XX DA`, falling back to `prix + shp` if no final line exists
   (`parsing/ppa.disambiguate(...)`).

Downstream of these, `nomenclature/` repairs **identity** fields from the matched
`num_enregistrement` record and **flags** (never overwrites) dispensing-critical conflicts
(`dosage`, `forme`) — see [ARCHITECTURE.md §4.8](ARCHITECTURE.md). The net effect: a money
field can be repaired by arithmetic, a code by structure + confusion map, and an identity
field by the nomenclature — but a *dispensing* contradiction is always escalated to the
human, never resolved silently.

---

## 6. Training & eval

`train.run(cfg_path, run_dir) -> Path`; `eval.run(...) -> dict`. Config-driven and seeded
(`vignocr.common.seed_everything`). Real training runs on **Narval HPC**
([ROADMAP.md](ROADMAP.md) Phase 7); a CPU smoke path proves the loop on the synthetic
fixture without a GPU.

- **Metrics.** Field **CER** (per-field and aggregated, with the business-critical subset
  called out) and **abstention precision** at the configured τ.
- **Gate (real data).** Field **CER ≤ 5%** on `business_critical_fields`; abstention
  precision documented ([ROADMAP.md](ROADMAP.md) Phase 5).
- **Synthetic-first.** On the fixture, eval is a CPU smoke check; the numerical gate is
  meaningful once the real 15-class-annotated text exists ([ROADMAP.md](ROADMAP.md)
  Phase 9). The reporting machinery works now so the gate is ready when real data lands.
- **Provenance.** The recognizer identity flows into
  `ExtractionRecord.model_versions["recognizer"]`, tying every read to the exact model that
  produced it.

---

## 7. Deployment

Stage 2 is an **independently deployable** GPU service behind the stateless API tier
([ARCHITECTURE.md §6](ARCHITECTURE.md)). It scales separately from detection — typically
several OCR replicas behind one detector, since one image yields many small crop reads —
and versions separately (re-export the recognizer without retraining the detector).

- **Export.** The chosen recognizer exports to **ONNX** for serving; only the exported
  artifact (plus git-SHA/config provenance) is shipped from Narval to serving.
- **CPU-first.** When `[ml]` is absent, a deterministic fixture recognizer fills the
  `Recognizer.read` seam so the end-to-end pipeline and tests run on CPU today; the real
  ONNX recognizer drops into the same seam unchanged.
- **Config switch.** Baseline vs transformer (and per-field routing) is a config choice
  (§1), so swapping recognizer families is a deployment-time decision backed by val
  numbers, not a code rewrite.

---

## 8. Cross-references

- [ARCHITECTURE.md](ARCHITECTURE.md) — full pipeline, abstention profiles, safety invariant.
- [DETECTION.md](DETECTION.md) — Stage 1 produces the crops + the orientation contract
  Stage 2 consumes.
- [INTERFACES.md](INTERFACES.md) — `FieldRead`, `FieldStatus`/`FieldSource`, and the
  `ocr/` + `parsing/` APIs.
- [`configs/parsing/fields.yaml`](../configs/parsing/fields.yaml) — regexes, money/Decimal
  rules, checksum, PPA disambiguation, abstention thresholds.
- [`configs/classes.yaml`](../configs/classes.yaml) — field types, orientations,
  business-critical set, rotated fields.
- [ROADMAP.md](ROADMAP.md) Phase 5 — OCR entry/exit gates.
