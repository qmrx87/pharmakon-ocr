---
name: pharmakon-ocr-integration
description: How VignOCR + FactureOCR (Claude) wire into the Pharmakon ERP monorepo — architecture, contract, invariants
metadata:
  type: project
---

`pharmakon-ocr` is a Python FastAPI service that lives as a sibling inside the **Pharmakon TS monorepo** (`C:\Users\yacin\Desktop\Devs\Pharmacon\Pharmacon_App\pharmakon`): `apps/server` (Express 4 + Prisma 5 + Postgres, ESM, `{success,data}` envelopes), `apps/web-pharmacy` (React 18 + Zustand + axios), `apps/web-medecin`, `apps/mobile-aalaji` (Flutter). The OCR repo has its OWN git remote (`qmrx87/pharmakon-ocr`); the monorepo has a separate git — **OCR code commits to the OCR repo, Node/web/compose commits to the monorepo**.

**Two Claude-backed OCR capabilities, both prefill-and-confirm (added 2026-06-23):**
- **VignOCR** (`/extract/vignette`, alias of `/extract`) — vignette image → drug fields, for POS/sales validation.
- **FactureOCR** (`/extract/facture`) — supplier-invoice image → `{header, lines[], totals, verification}` for bulk stock intake; `vignocr.facture` package; deterministic arithmetic verifier (qty×unit≈line_total, Σ≈net) is the facture analogue of the vignette `prix+shp=ppa` checksum. See [[vignocr-claude-variant]] for the underlying ClaudeExtractor.

**Data flow:** React → Node (`/api/v1`) → Python OCR (`OCR_SERVICE_URL`, default `http://pharmakon-ocr:8000` in compose). Key files: Node `apps/server/src/services/ocr.service.js` (calls Python, maps to domain shape, resolves `produit_id` tenant-scoped, READ-ONLY) + `routes/ocr.routes.js` (`/ocr/scan-vignette`, `/ocr/extract-facture`, gated `requireModule('VIGNOCR'|'FACTUROCR')`). Web: `services/index.js` `ocrApi.scanVignette/extractFacture`, `components/ocr/VignetteScanPanel.jsx` (wired in `pages/vente/CaisseLibre.jsx` + `pages/achat/NouvelAchat.jsx`), and `NouvelAchat.jsx` `OcrPanel`/`handleOcrResult` (facture). Python: `serving/app.py` + `deps.py` (`get_facture_extractor`, `VIGNOCR_FACTURE_ENABLED`), `Dockerfile` installs `.[claude]`.

**Why / invariants (do not break):**
- **Never auto-commit.** OCR only reads + prefills; the human edits and finalises via the EXISTING `POST /achats/:id/validate` / `ventes/:id/validate`. No stock/lot/price writes from OCR.
- Multipart field name is **`image`** end-to-end (frontend→Node multer); Node→Python uses `file`.
- Vignette field keys from the pipeline: `num_lot, date_fab, date_exp, num_enregistrement, ppa, ppa_shp, tr, product_name` (`configs/v2/claude.yaml`) — Node `_mapVignette` reads these.
- When OCR is **configured but fails → 503** (never fabricate); the in-stock-lot "stub" only runs when `OCR_SERVICE_URL` is unset (dev). FactureOCR also stubs (empty) when no `ANTHROPIC_API_KEY` + `VIGNOCR_ALLOW_STUB=1`.
- Images leave infra (Claude API) → governance/ZDR is the user's call (they chose "enable both now").

**How to apply:** This checkout has NO JS deps installed (no eslint/vitest/prisma/express) → can't run Node/web lint or tests here; validate with `node --check` + review, and tell the user to run `npm install && npm run lint && npm test` in the monorepo. Reuse `efacture.service.js _resolveProduitId` pattern and the `acceptEFacture` SERIALIZABLE transaction if ever adding server-side draft creation (a deliberately-skipped "deeper" option). E-Facture writes the same `FactureAchat/DetailAchat/StockLot` tables — dedup by `numero_facture_fournisseur` (warn-only).
