# VignOCR — Module Interfaces (the coherence contract)

Every module is built to **this** spec so the parts compose. Read this before
touching any `src/vignocr/` file. Three global rules bind all modules:

1. **Config-driven.** No hardcoded class names, paths, thresholds, regexes, or
   hyperparameters. Read them via `vignocr.common.load_config(...)` /
   `get_classes()` / `get_active_dataset()`. The field schema lives once in
   `configs/classes.yaml`.
2. **Money is `decimal.Decimal`, never `float`.** Parse → `Decimal`; serialize to
   JSON as a **string** (e.g. `"250.00"`). Quantize to the centime (`"0.01"`).
3. **ML libs are lazy-imported** *inside functions*, never at module top-level, so
   the core (parsing/nomenclature/data/serving/pipeline) imports and tests run on
   CPU **without** `torch`/`rfdetr`/`paddleocr`. Guard with a clear `ImportError`
   message telling the user to `pip install -e .[ml]`.

---

## Canonical data types

Defined as Pydantic v2 models in `src/vignocr/serving/schemas.py` and reused
everywhere (pipeline returns these; serving emits them directly).

```python
FieldStatus = Literal["ok", "abstain", "corrected", "conflict", "missing"]
FieldSource = Literal["ocr", "nomenclature", "ocr+nomenclature", "checksum", "none"]

class BBox(BaseModel):           # COCO xywh in pixels of the input image
    x: float; y: float; w: float; h: float

class FieldRead(BaseModel):
    name: str                    # one of classes.yaml field names
    value: str | None            # normalized canonical value (money as "250.00")
    raw: str | None              # raw OCR text before normalization
    confidence: float            # 0..1 (recognition or correction confidence)
    status: FieldStatus
    source: FieldSource
    bbox: BBox | None

class Reimbursability(BaseModel):
    color: Literal["green", "red", "orange", "unknown"]
    eligible: bool | None        # green->True, red->False, orange/unknown->None
    confidence: float
    label: str

class ChecksumReport(BaseModel):
    verdict: Literal["ok", "repaired", "mismatch", "incomplete"]
    prix: str | None; shp: str | None; ppa: str | None   # Decimal strings
    repaired_field: str | None

class NomenclatureReport(BaseModel):
    matched: bool
    num_enregistrement_normalized: str | None
    match_confidence: float
    conflicts: list[dict]        # [{field, ocr, nomenclature, action}]

class ExtractionRecord(BaseModel):
    image_id: str
    fields: dict[str, FieldRead]            # keyed by field name
    reimbursability: Reimbursability
    checksum: ChecksumReport
    nomenclature: NomenclatureReport
    abstentions: list[str]                  # field names with status == "abstain"
    flow: Literal["selling", "receiving"]   # which abstention profile was applied
    model_versions: dict[str, str]          # {detector, recognizer, nomenclature_version}
    timings_ms: dict[str, float]
```

---

## `data/` — dataset
```python
synthetic.generate(out_dir: Path, *, num: dict[str,int], seed: int, image_size: tuple[int,int]) -> None
coco.load_split(root: Path, split: str) -> CocoSplit            # images, anns, name-keyed categories
coco.crops_for_image(img_path, anns, schema) -> dict[str, list[Crop]]   # field name -> crops (BBox + PIL.Image)
validate.check_integrity(root: Path) -> IntegrityReport         # raises/collects per data.yaml integrity flags
stats.summarize(root: Path) -> dict                             # per-class counts, box stats, split sizes
```
COCO category→class mapping is **by name** via each file's own `categories`. Unknown
names → integrity error (unless `assert_class_names_subset_of_schema` is relaxed).

## `detection/` — RF-DETR medium (lazy `torch`/`rfdetr`)
```python
train.run(cfg_path: str, run_dir: Path, resume: Path|None=None) -> Path   # -> best checkpoint
eval.run(ckpt: Path, root: Path, split="valid") -> dict                   # mAP, per-class AP, localization_recall
export.to_onnx(ckpt: Path, out: Path) -> Path                             # + parity check vs torch on fixtures
infer.Detector(ckpt_or_onnx).detect(image) -> list[Detection]             # Detection: name, score, BBox
```
Augmentation must be **band-color-preserving** (no hue/channel ops that flip green↔red).

## `ocr/` — recognition (lazy OCR backend)
```python
infer.Recognizer(cfg).read(crop, field_type, orientation) -> FieldRead    # confidence + status (abstain<τ)
preprocess.orient(crop, orientation) -> crop                              # rotate vertical vin fields upright
train.run(cfg_path, run_dir) -> Path ; eval.run(...) -> dict (CER, abstention precision)
```
Abstention τ from `parsing/fields.yaml: abstention[flow]`; selling stricter than receiving.

## `parsing/` — deterministic normalization (pure CPU, no ML)
```python
money.parse(text) -> Decimal | None         # canonical, centime-quantized
dates.parse(text, formats) -> date | None
codes.normalize_enregistrement(text) -> str | None
checksum.verify_and_repair(prix: FieldRead, shp, ppa) -> (dict[str,FieldRead], ChecksumReport)
ppa.disambiguate(candidates: list[FieldRead]) -> FieldRead   # picks the final "= XXX,XX DA"
record.build(fields: dict[str,FieldRead], flow) -> partial ExtractionRecord
```

## `nomenclature/` — correction (pure CPU, rapidfuzz)
```python
loader.load_csv(path) -> NomenclatureIndex
match.find(norm_code, index, cfg) -> (row|None, confidence)
correct.apply(fields: dict[str,FieldRead], row, cfg) -> (dict[str,FieldRead], NomenclatureReport)
```
`apply` obeys `correction.yaml.policy`: **never** touch `ppa`/`tr`; repair identity;
**flag** (status="conflict") dispensing-critical disagreements — never overwrite them.

## `pipeline/` — orchestrator
```python
orchestrator.VignocrPipeline(cfg).extract(image, *, flow="selling") -> ExtractionRecord
```
Order: preprocess → detect → per-field orient+crop → recognize → parse+checksum+PPA →
nomenclature correct → reimbursability(color_band) → assemble `ExtractionRecord`.
Detector/recognizer may be **stubbed** (deterministic fixture readers) when `[ml]` is
absent, so the e2e pipeline runs on CPU today.

## `serving/` — FastAPI (stateless, config-driven)
```
GET  /health   -> {"status":"ok"}                 (liveness)
GET  /ready    -> {"ready": bool, "models": {...}} (model load readiness)
POST /extract  -> ExtractionRecord                 (multipart image + flow query/body)
```
Auth, upload validation (MIME/size), and idempotency-key handling per `docs/SECURITY.md`
and `docs/INTEGRATION.md`. Workers are stateless; all config via env (12-factor).
