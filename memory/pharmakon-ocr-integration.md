---
name: pharmakon-ocr-integration
description: How VignOCR + FactureOCR (Claude) are wired into the Pharmakon ERP — architecture, contract, invariants
metadata:
  type: project
---

The Pharmakon app is a **TS monorepo** at `C:\Users\yacin\Desktop\Devs\Pharmacon\Pharmacon_App\pharmakon` (remote `qmrx87/pharmakon_jsx`, PR workflow on `main`): `apps/server` (Express 4 + Prisma 5 + Postgres, ESM, `{success,data}` envelopes), `apps/web-pharmacy` (React 18 + Zustand + axios), `apps/web-medecin`, `apps/mobile-aalaji` (Flutter). `pharmakon-ocr` (Python, remote `qmrx87/pharmakon-ocr`) sits inside it but is the **research/training repo + standalone OCR service** — **the ERP does NOT call it at runtime** (decided 2026-06-23).

**OCR runs in `apps/server`, calling the Claude API DIRECTLY** (`@anthropic-ai/sdk`) — chosen because the server is the Railway deployable and both capabilities are just "image → structured JSON". No separate Python service to deploy.
- **VignOCR** (`POST /api/v1/ocr/scan-vignette`, module `VIGNOCR`) — vignette image → flat drug fields for POS/sales.
- **FactureOCR** (`POST /api/v1/ocr/extract-facture`, module `FACTUROCR`) — supplier-invoice image/PDF → stock-intake prefill (line items) for `NouvelAchat`.

**Data flow:** React → Node `apps/server` → Claude API. Key files (all in the monorepo): `src/services/ocr.service.js` (Anthropic `messages.stream().finalMessage()`, vision image/PDF block + `output_config.format` structured output + cached system prompt + adaptive thinking `effort:low`, model `claude-opus-4-8`; maps to domain shape, resolves `produit_id` tenant-scoped, READ-ONLY), `src/services/ocr.verify.js` (deterministic facture arithmetic check qty×unit≈line_total, Σ≈net — the facture analogue of the vignette prix+shp=ppa checksum), `routes/ocr.routes.js` (both routes, `requireModule`-gated, warn-only dedup by `numero_facture_fournisseur`). Web: `services/index.js` `ocrApi.scanVignette/extractFacture` (client-side `_downscaleImage` before upload — Claude's ~5MB base64 limit), `components/ocr/VignetteScanPanel.jsx` (wired in `pages/vente/CaisseLibre.jsx` + `pages/achat/NouvelAchat.jsx`), `NouvelAchat.jsx` `OcrPanel`/`handleOcrResult` (facture). Prompts/schemas are **ported into ocr.service.js** (self-contained — not read from pharmakon-ocr). See [[vignocr-claude-variant]] for the original Python ClaudeExtractor the prompts came from.

**Why / invariants (do not break):**
- **Never auto-commit.** OCR only reads + prefills; the human edits and finalises via the EXISTING `POST /achats/:id/validate` / `ventes/:id/validate`. No stock/lot/price writes from OCR. `NouvelAchat` has a per-line completeness guard (qty>0, prix_achat_ht>0, péremption) so incomplete OCR lines show a targeted message, not a backend 400.
- `isConfigured()` = **`ANTHROPIC_API_KEY` present**. Configured + Claude fails → 503 (never fabricate). Vignette falls back to the in-stock-lot demo stub ONLY when the key is unset (dev); FactureOCR has no stub → 503 when unset.
- Multipart field name is **`image`** end-to-end (frontend → multer); the server reads `file.buffer`/`file.mimetype` (image/* → image block, application/pdf → document block).
- Vignette fields the model returns (each `{value,confidence}`): `num_lot, date_fab, date_exp, num_enregistrement, ppa, ppa_shp, tr, product_name` — `_mapVignette` reads these.
- Images are sent to the Claude API (leave infra) → governance/ZDR is the user's call (they chose "enable both now").

**How to apply / run:** dev — set `ANTHROPIC_API_KEY` in the server `.env` (loaded via `dotenv/config`), restart, scan. Docker — `docker-compose.yml` passes `ANTHROPIC_API_KEY`/`OCR_CLAUDE_MODEL` to the server (the old `pharmakon-ocr` compose service + `OCR_SERVICE_URL` were removed). This checkout has NO JS deps installed → can't run eslint/vitest here; validate with `node --check` + review, tell the user to `npm install && npm run lint && npm test`. Reuse `efacture.service.js _resolveProduitId` + the `acceptEFacture` SERIALIZABLE transaction if ever adding server-side draft creation (a deliberately-skipped deeper option). E-Facture writes the same `FactureAchat/DetailAchat/StockLot` tables.
