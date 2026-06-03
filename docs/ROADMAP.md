# VignOCR — Implementation Roadmap

Sequenced, dependency-aware. Each phase has **entry/exit criteria** and the **gating
metric** that must be green before moving on. Status legend: `[ ]` todo · `[~]` in
progress · `[x]` done.

> Current reality: the annotated dataset is **not ready** (real exports carry only
> `{date_info, entete, vin}` or `{text, drug-labels}` — not the 15-class schema). The
> whole pipeline is therefore built and proven on a **synthetic fixture** that matches
> the target COCO schema exactly. Switchover to real data is a single `configs/data.yaml`
> change (`active: synthetic → real`) — see `docs/SWITCHOVER.md`.

---

## Phase 0 — Foundation & contract  `[~]`
**Entry:** repo created. **Exit gate:** `configs/classes.yaml` is the single source of
truth; `pyproject` installs; `vignocr.common` imports clean.
- [x] `configs/classes.yaml` — 12 fields + `entete`/`vin`/`color_band` (single source of truth)
- [x] `configs/data.yaml`, `configs/parsing/fields.yaml`, `configs/nomenclature/correction.yaml`
- [x] `pyproject.toml` — pinned core deps + `[ml]`/`[dev]` extras, ruff/black/pytest
- [x] `src/vignocr/common/` — config loader, structured logging, seeding, metrics
- [x] `docs/INTERFACES.md` — data schema + module APIs (the coherence contract)

## Phase 1 — Data layer + synthetic fixture  `[ ]`
**Entry:** Phase 0. **Exit gate:** synthetic dataset generates deterministically and
passes all integrity asserts; stats print.
- [ ] `data/synthetic.py` — draw ~20 vignette-like images + valid COCO for all 15 classes, train/val/test
- [ ] `data/coco.py` — load COCO, map category **by name** (robust to Roboflow ids)
- [ ] `data/validate.py` — split-leakage, bbox validity, class-names ⊆ schema, business-critical coverage
- [ ] `data/stats.py` — per-class counts, box-size distribution, per-split sizes
- **Gate:** `pytest tests/test_dataset_integrity.py` green; `python -m vignocr.data.stats` prints.

## Phase 2 — Parsing + checksum + PPA disambiguation  `[ ]`
**Entry:** Phase 0. **Exit gate:** money is Decimal everywhere; checksum repairs to the centime; final PPA selected.
- [ ] Decimal money parser (`,`/`.`/thousands/`DA`), date parser, code parsers
- [ ] `prix + shp == ppa` checksum: verify, repair-from-two, flag-on-mismatch verdicts
- [ ] final-PPA disambiguation (`= XXX,XX DA` vs intermediate `a+b`)
- **Gate:** golden-fixture parse tests assert to the centime/character.

## Phase 3 — Nomenclature correction  `[ ]`
**Entry:** Phase 2. **Exit gate:** N°Enregistrement normalized + matched; identity fields repaired; **`ppa`/`tr` never touched**; dispensing conflicts flagged.
- [ ] CSV load + `scripts/ingest_nomenclature.py` (xlsx → csv)
- [ ] normalize (letter blocks `D`/`F`/`G`, spacing, confusion map), structural edit-distance match
- [ ] policy engine (repair_always / repair_if_low_or_agree / flag_on_conflict / never_overwrite) + conflict report
- **Gate:** correction tests incl. an OCR↔nomenclature dosage-conflict that is **flagged, not overwritten**.

## Phase 4 — Detection (RF-DETR medium)  `[ ]`
**Entry:** Phase 1. **Exit gate:** trains on fixture (CPU smoke / Narval A100 real); exports ONNX with parity.
- [ ] train / eval / export(ONNX) / infer, config-driven, seeded, resumable-from-checkpoint
- [ ] band-color-preserving augmentation (NO hue flips that invert green↔red semantics)
- [ ] eval: mAP@[.5:.95], per-class AP, **business-critical localization recall**
- **Gate (real data):** mAP@.5 ≥ 0.85 overall **and** localization recall = 1.0 on `business_critical_fields`.

## Phase 5 — OCR recognition  `[ ]`
**Entry:** Phase 4 crops available. **Exit gate:** per-field recognition with confidence + abstention.
- [ ] baseline recognizer (PaddleOCR/CRNN) + field-type-aware preprocessing + **per-field orientation correction** (rotated `vin` fields)
- [ ] confidence scoring + abstention threshold (selling stricter than receiving)
- [ ] transformer-direction scaffold (TrOCR/Donut) behind a config switch
- **Gate (real data):** field CER ≤ 5% on `business_critical_fields`; abstention precision documented.

## Phase 6 — End-to-end pipeline + serving  `[ ]`
**Entry:** Phases 2–5. **Exit gate:** image → structured JSON; FastAPI serves it; container runs.
- [ ] orchestrator: preprocess → detect → orient+crop → recognize → parse → checksum → correct → JSON
- [ ] FastAPI `/extract`, `/health`, `/ready`; abstention/confidence/checksum metadata in response
- [ ] `Dockerfile`, ONNX-export parity check, `docker run` against fixture model
- **Gate:** `pytest tests/test_pipeline_e2e.py` green on the four/fixture goldens.

## Phase 7 — Narval HPC orchestration  `[ ]`
**Entry:** Phases 4–5 code. **Exit gate:** one-command DAG (validate→train→eval→export→ocr→bench) via `afterok`.
- [ ] `scripts/setup_narval.sh`; parameterized `sbatch` (`--account=$VIGNOCR_ACCOUNT`); checkpoint-to-scratch; git-SHA+config snapshot per run dir
- **Gate:** dry-run `sbatch --test-only` accepts every job; `submit_all.sh` builds the dependency DAG.

## Phase 8 — Integration contract + HITL + monitoring  `[ ]`
**Entry:** Phase 6. **Exit gate:** Pharmakon-side contract documented (prefill-and-confirm, selling vs receiving); HITL loop + monitoring designed.
- [ ] `docs/INTEGRATION.md` (request/response, idempotency, abstention semantics) + stub payloads
- [ ] `docs/HITL.md`, `docs/MONITORING.md`, `docs/SECURITY.md`
- **Gate:** stub payloads validate against the Pydantic schemas.

## Phase 9 — Real-data switchover  `[ ]`  (blocked on annotation)
**Entry:** real annotations land at `ocr/data/` in the 15-class schema. **Exit gate:** `active: real`, integrity asserts pass, Phase 4/5 gates met on real val.
- [ ] flip `data.yaml`, run `docs/SWITCHOVER.md` checklist, launch Narval DAG
