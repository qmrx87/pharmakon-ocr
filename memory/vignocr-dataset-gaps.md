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

**Business priority (clarified 2026-06):** SaaS/cloud GPU deployment, very low latency. Must-detect set is NARROW: sales flow needs only **LOT** (deduct from correct lot); purchase flow needs **LOT + date_exp + date_fab + num_enrg + PPA + TR**, then the *identity* fields (product_name/dci/dosage/forme/laboratoire) are filled from the Nomenclature DB via num_enrg. So a ~6-field detector + targeted OCR (+ barcode decode for num_enrg if present) suffices instead of a 17-class head.

**CORRECTION (verified 2026-06-06, AUDIT.md F-04):** the Nomenclature does **NOT** carry price. The real `NOMENCLATURE-VERSION-FEVRIER-2026-.xlsx` (sheet 1, header row 14, ~5,298 drug rows, 19 cols: N°ENREGISTREMENT, CODE, DCI, NOM DE MARQUE, FORME, DOSAGE, CONDITIONNEMENT, LISTE, P1, P2, OBS, LABORATOIRE, PAYS, dates, TYPE, STATUT, DUREE STABILITE) has **NO PPA and NO TR/tariff column at all**. Therefore **ppa AND tr MUST be captured by VignOCR from the vignette** — they can never be back-filled from nomenclature. The earlier wording above ("ppa/tr filled from Nomenclature DB") was WRONG. Also: real `N°ENREGISTREMENT` = `352/01 A 003/06/22` (+ a distinct `CODE` `01 A 003`), which does NOT match the parser regex `AA/BB/CC<LETTER>DDD/EEE` (F-03) — the match key/grammar must be re-derived before nomenclature matching works on real data.

**STATUS UPDATE:** the validate-vs-train dataset mismatch noted below is now FIXED — `slurm/submit_all.sh` defaults `VIGNOCR_DATA_ACTIVE=real` and `slurm/01_validate_data.sbatch` validates BOTH `real` and `vignette` explicitly (the datasets the trainers actually bind).

**Why it matters:** these don't crash training — they yield a model that converges and still can't read prices/reimbursability. **How to apply:** treat Stage-B box coverage (not OCR auto-labeling) as the real bottleneck; barcode→nomenclature for identity, small detector + OCR for lot/dates. See [[narval-env-and-failures]].
