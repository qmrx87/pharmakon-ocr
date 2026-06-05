---
name: vignocr-dataset-gaps
description: VignOCR real-dataset annotation gaps that will silently cripple the trained model
metadata:
  type: project
---

The VignOCR real Roboflow exports do NOT contain the full 17-class schema (configs/classes.yaml), so even once training runs it produces a model that can't meet the business goal. Verified from the COCO files 2026-06-05:

- `data/` (Stage B field detector, 17-class head, 542 train imgs): annotated classes = forme(639) dci(594) num_enregistrement(554) product_name(550) laboratoire(543) date_exp(531) date_fab(525) lot→num_lot(513) ppa(421) ppa_shp(408) tarif_ref→tr(252) dosage(62). **MISSING: prix, shp (fused into ppa_shp), entete, vin, and color_band.**
- `color_band` has ZERO annotations → the CHIFA reimbursability feature (green/red/orange) has no training data despite being a headline feature.
- `dosage` only 62 instances → severe imbalance.
- `data2/` (Stage A, 3-class head): vin(552) entete(551), **date_info = 0 annotations** (declared but empty → really a 2-class problem).

Also: stage 01 validate keys off `VIGNOCR_DATA_ACTIVE` (defaults `synthetic`) while detection configs hardcode `dataset: real`/`vignette` — so **validation checks synthetic data the trainer never sees**. README's data-reconciliation table is stale (predates these exports).

**Geometry (verified):** data/ = already-cropped vignettes (~401x200), field boxes; lot/date_exp/date_fab are tiny vertical strips (~20x160) clustered right; num_enregistrement is a wide bottom box (~231x28). data2/ = wider webcam box photos (416x416); `entete` is a tall-narrow 60x204 strip (~7% area) that per the user is the **lot+date_fab+date_exp block** (NOT the whole vignette body); `vin` is the wide 38%-area region. Zero filename overlap between data/ and data2/ (different image distributions). num_enregistrement exists ONLY in data/ (not data2).

**Business priority (clarified 2026-06):** SaaS/cloud GPU deployment, very low latency. Must-detect set is NARROW: sales flow needs only **LOT** (deduct from correct lot); purchase flow needs **LOT + date_exp + date_fab + num_enrg**, then everything else (product_name/dci/dosage/forme/laboratoire/ppa/tr) is filled from the Nomenclature DB via num_enrg. So a 17-class detector is overkill — a ~4-class detector + targeted OCR (+ barcode decode for num_enrg if present) suffices.

**Why it matters:** these don't crash training — they yield a model that converges and still can't read prices/reimbursability. **How to apply:** treat Stage-B box coverage (not OCR auto-labeling) as the real bottleneck; barcode→nomenclature for identity, small detector + OCR for lot/dates. See [[narval-env-and-failures]].
