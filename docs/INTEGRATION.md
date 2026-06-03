# VignOCR ↔ Pharmakon — Integration Contract

This document specifies the **Pharmakon-side contract** for consuming the VignOCR service.
It describes how Pharmakon should call the service and wire the results into its Vente and
Achat/inventory flows.

> **Decoupling.** VignOCR stays independent. **This repo does not contain or modify any
> Pharmakon code.** This is a contract Pharmakon implements against; the only shared artifact
> is the JSON schema, which mirrors `src/vignocr/serving/schemas.py`.

> **Authority.** Schema, endpoints, and abstention semantics come from:
> - `src/vignocr/serving/schemas.py` — the wire types (`ExtractionRecord` et al.)
> - `docs/INTERFACES.md` — endpoints + canonical data types
> - `docs/SERVING.md` — sync vs async paths, health/readiness
> - `configs/parsing/fields.yaml: abstention` — **selling stricter than receiving**
> - `configs/classes.yaml` — field names / reimbursability
> - `docs/SECURITY.md` — auth, upload validation, audit
> - `docs/NOMENCLATURE_CORRECTION.md` — conflict semantics surfaced to the operator

---

## 1. The golden rule: extract → human-validate → **then** write

VignOCR is an **advisory prefill engine**. It never mutates Pharmakon's stock, ledgers, or
prices. The ordering is non-negotiable in both flows:

```
scan  ──►  VignOCR extracts  ──►  Pharmakon prefills the form
                                  ──►  HUMAN reviews & validates
                                       ──►  THEN Pharmakon writes (stock / lot / price)
```

A machine read is a *suggestion*. Nothing in Pharmakon's database changes until a person
confirms. This protects against a misread silently deducting the wrong stock or recording the
wrong price.

---

## 2. Endpoints Pharmakon calls

(Full contract in `docs/SERVING.md`.) Pharmakon is a client of:

| Flow                        | Call                                       | Sync? | `flow` value |
| --------------------------- | ------------------------------------------ | ----- | ------------ |
| Vente (selling popup)       | `POST /extract`                            | sync  | `selling`    |
| Achat / receiving / invent. | `POST /batch` + `GET /jobs/{id}` (or webhook) *(planned, `docs/SERVING.md` §4)* | async | `receiving` |
| Achat — available today     | `POST /extract` per vignette (client loop) | sync  | `receiving`  |
| (single receiving scan)     | `POST /extract`                            | sync  | `receiving`  |

Every call carries `Authorization` (when auth is enabled, `docs/SECURITY.md`) and an
`Idempotency-Key` (§8). The service echoes the chosen profile back in
`ExtractionRecord.flow`. **The implemented surface today is `POST /extract` (+ `/health`,
`/ready`); the async `/batch`+`/jobs` path is the planned receiving architecture.**

---

## 3. Request schema

`POST /extract` (sync) is `multipart/form-data`:

| Part / param        | Type                  | Notes                                                                       |
| ------------------- | --------------------- | --------------------------------------------------------------------------- |
| `file`              | file (multipart)      | the vignette photo/scan; MIME prefix + size validated (SECURITY.md)         |
| `flow`              | form field            | `selling`/`receiving`; selects abstention profile; defaults to `selling`    |
| `idempotency_key`   | form field            | client-generated UUID; echoed back for caller-side de-dup (§8)              |
| `Authorization`     | header                | `Bearer <token>` when auth is enabled (SECURITY.md)                         |

(The current handler binds `file`, `flow`, and `idempotency_key` from the multipart body and
echoes the key back as an `Idempotency-Key`/`X-Request-ID` response header.)

`POST /batch` (async) takes `images[]` (multiple files) or a manifest, same `flow`, same
headers; it returns `202 { job_id, count, status }` and the results are fetched via
`GET /jobs/{job_id}` or delivered to a registered callback URL.

### 3.1 Stub request (sync selling)

```http
POST /extract HTTP/1.1
Host: vignocr.internal
Authorization: Bearer eyJhbGciOi...
Content-Type: multipart/form-data; boundary=----vign

------vign
Content-Disposition: form-data; name="flow"

selling
------vign
Content-Disposition: form-data; name="idempotency_key"

4d1f0e2a-7c33-4b2a-9b6e-2f1a9c0e5e21
------vign
Content-Disposition: form-data; name="file"; filename="vignette.jpg"
Content-Type: image/jpeg

<binary JPEG bytes>
------vign--
```

(`flow` may also be passed as a `?flow=selling` query param. The image part name is `file`.)

---

## 4. Response schema — `ExtractionRecord`

The response **mirrors `src/vignocr/serving/schemas.py` exactly**. Pharmakon should treat
these field names as the contract:

```python
FieldStatus = Literal["ok", "abstain", "corrected", "conflict", "missing"]
FieldSource = Literal["ocr", "nomenclature", "ocr+nomenclature", "checksum", "none"]

class BBox:            x: float; y: float; w: float; h: float
class FieldRead:       name; value: str|None; raw: str|None; confidence: float;
                       status: FieldStatus; source: FieldSource; bbox: BBox|None
class Reimbursability: color: "green"|"red"|"orange"|"unknown"; eligible: bool|None;
                       confidence: float; label: str
class ChecksumReport:  verdict: "ok"|"repaired"|"mismatch"|"incomplete";
                       prix; shp; ppa: str|None; repaired_field: str|None
class NomenclatureReport: matched: bool; num_enregistrement_normalized: str|None;
                       match_confidence: float; conflicts: list[{field, ocr, nomenclature, action}]
class ExtractionRecord: image_id; fields: dict[str, FieldRead]; reimbursability;
                       checksum; nomenclature; abstentions: list[str];
                       flow: "selling"|"receiving"; model_versions; timings_ms
```

Key consumption rules for Pharmakon:
- **Money values are Decimal strings** (`"702.56"`), never numbers. Parse as decimal; do not
  `parseFloat`. (`schemas.py: money_str`.)
- `fields` is keyed by `classes.yaml` field name (`ppa`, `prix`, `shp`,
  `num_enregistrement`, `num_lot`, `date_fab`, `date_exp`, `product_name`, `dci`, `dosage`,
  `forme`, `laboratoire`).
- A field with `value: null` / `status: "missing"` was not detected; `status: "abstain"`
  means detected-but-below-threshold (§6).
- `reimbursability` is a **separate signal** (band colour), not OCR — drive the CHIFA
  badge from it, not from any text field.
- `nomenclature.conflicts` drives "à vérifier" badges (§5.2, `docs/NOMENCLATURE_CORRECTION.md`).

### 4.1 Stub response (sync selling — happy path with one abstention)

```json
{
  "image_id": "vignette-4d1f0e2a",
  "flow": "selling",
  "fields": {
    "ppa":               { "name": "ppa",  "value": "702.56", "raw": "PPA = 702,56 DA", "confidence": 0.98, "status": "ok",       "source": "checksum",          "bbox": {"x": 412, "y": 96, "w": 120, "h": 28} },
    "prix":              { "name": "prix", "value": "700.06", "raw": "700.06",          "confidence": 0.97, "status": "ok",       "source": "ocr",               "bbox": {"x": 412, "y": 60, "w": 110, "h": 24} },
    "shp":               { "name": "shp",  "value": "2.50",   "raw": "2,50",            "confidence": 0.96, "status": "ok",       "source": "ocr",               "bbox": {"x": 540, "y": 60, "w": 70,  "h": 24} },
    "num_enregistrement":{ "name": "num_enregistrement", "value": "18/97/14G061/003", "raw": "18/97/14G 061/003", "confidence": 0.95, "status": "ok", "source": "ocr", "bbox": {"x": 40, "y": 30, "w": 220, "h": 26} },
    "num_lot":           { "name": "num_lot",  "value": "L22A0457", "raw": "L22A0457",  "confidence": 0.93, "status": "ok",       "source": "ocr",               "bbox": {"x": 300, "y": 200, "w": 24, "h": 140} },
    "date_fab":          { "name": "date_fab", "value": "2022-03", "raw": "03/2022",    "confidence": 0.94, "status": "ok",       "source": "ocr",               "bbox": {"x": 330, "y": 200, "w": 24, "h": 110} },
    "date_exp":          { "name": "date_exp", "value": "2025-03", "raw": "03/2025",    "confidence": 0.94, "status": "ok",       "source": "ocr",               "bbox": {"x": 360, "y": 200, "w": 24, "h": 110} },
    "product_name":      { "name": "product_name", "value": "AUGMENTIN", "raw": "AUGMENT1N", "confidence": 0.99, "status": "corrected", "source": "nomenclature", "bbox": {"x": 40, "y": 70, "w": 260, "h": 30} },
    "dci":               { "name": "dci",  "value": "AMOXICILLINE + ACIDE CLAVULANIQUE", "raw": "AMOXICILLINE + AC. CLAV.", "confidence": 0.92, "status": "ok", "source": "ocr+nomenclature", "bbox": {"x": 40, "y": 104, "w": 300, "h": 24} },
    "dosage":            { "name": "dosage", "value": "1 g", "raw": "1 g", "confidence": 0.95, "status": "ok", "source": "ocr+nomenclature", "bbox": {"x": 40, "y": 130, "w": 80, "h": 22} },
    "forme":             { "name": "forme",  "value": "Comprimé", "raw": "Comprime", "confidence": 0.91, "status": "ok", "source": "ocr+nomenclature", "bbox": {"x": 130, "y": 130, "w": 120, "h": 22} },
    "laboratoire":       { "name": "laboratoire", "value": "GSK", "raw": "G5K", "confidence": 0.88, "status": "abstain", "source": "ocr", "bbox": {"x": 40, "y": 156, "w": 120, "h": 22} }
  },
  "reimbursability": { "color": "green", "eligible": true, "confidence": 0.97, "label": "Remboursable (CHIFA)" },
  "checksum":        { "verdict": "ok", "prix": "700.06", "shp": "2.50", "ppa": "702.56", "repaired_field": null },
  "nomenclature":    { "matched": true, "num_enregistrement_normalized": "18/97/14G061/003", "match_confidence": 0.93, "conflicts": [] },
  "abstentions":     ["laboratoire"],
  "model_versions":  { "detector": "rfdetr-m@1.2.0", "recognizer": "paddle-crnn@0.4.1", "nomenclature_version": "2026-02" },
  "timings_ms":      { "detect": 180.2, "ocr": 240.7, "parse": 3.1, "nomenclature": 1.4, "total": 1480.0 }
}
```

Here `laboratoire` was read at `0.88` — **below the selling threshold `0.90`** — so it comes
back `abstain` and appears in `abstentions`. Under `receiving` (τ`0.75`) the *same* read would
have been `ok`. This is the stricter-selling rule in action (§6).

---

## 5. Flow A — Vente (prefill-and-confirm at the till)

The selling popup. Latency-sensitive (target ~2 s, `docs/SERVING.md` §3), and the **strictest**
abstention profile because a wrong dispense is unacceptable.

```
1. Cashier scans the vignette at the till.
2. Pharmakon → POST /extract?flow=selling  (image, auth, idempotency-key).
3. VignOCR returns an ExtractionRecord in ~2 s.
4. Pharmakon opens a CONFIRM POPUP, prefilled from the record:
     • product (product_name/dci/dosage/forme/laboratoire)
     • price block (ppa/prix/shp) with the checksum verdict
     • lot + expiry (num_lot/date_exp)
     • reimbursability (CHIFA) badge from `reimbursability`
     • "à vérifier" badges on every abstain / conflict / missing field
5. Cashier reviews, corrects any flagged field, and VALIDATES.
6. ===> ONLY THEN does Pharmakon deduct stock / record the sale. <===
   (If the cashier cancels, nothing is written.)
```

### 5.1 What blocks a one-click confirm

Pharmakon should require explicit cashier attention (not allow blind "OK") when, for a field
it depends on:
- `status == "abstain"` (below selling τ) — the value is shown but must be confirmed/typed.
- `status == "conflict"` — OCR disagrees with nomenclature on a dispensing-critical field
  (`dosage`/`forme`); the operator must resolve (`docs/NOMENCLATURE_CORRECTION.md` §5.1).
- `status == "missing"` — not detected; operator enters it.
- `checksum.verdict == "mismatch"` — the price block does not add up
  (`prix + shp != ppa`); operator verifies the price.
- `reimbursability.color in ("orange","unknown")` — CHIFA eligibility could not be
  determined; operator decides.

### 5.2 Reimbursability & checksum in the popup

- Drive the CHIFA badge from `reimbursability` only: `green→Remboursable`, `red→Non
  remboursable`, `orange/unknown→À vérifier` (`classes.yaml: reimbursability`).
- Show `checksum.verdict`: `ok`/`repaired` → green tick (note: `repaired_field` tells the
  cashier the service recomputed one value from the other two); `mismatch` → red, block
  one-click; `incomplete` → amber (a price field was missing).

### 5.3 The stock-deduction guarantee

Stock deduction is triggered by **step 6**, the cashier's validation event — *never* by the
arrival of the `ExtractionRecord` at step 3. VignOCR has no write access to inventory; it
cannot deduct stock even in principle (`docs/SERVING.md` §9). This repo does not implement
step 6 — Pharmakon does.

---

## 6. Abstention semantics — selling is STRICTER than receiving

From `configs/parsing/fields.yaml: abstention`:

```yaml
abstention:
  selling:   { default: 0.90 }   # stricter — a wrong dispense is unacceptable
  receiving: { default: 0.75 }
```

- The `flow` query param chooses the profile; the service applies it and echoes
  `ExtractionRecord.flow`.
- A field whose recognition/correction confidence is **below** the active threshold is
  emitted with `status="abstain"` and listed in `abstentions`. Its `value` may still be shown
  to the human as a hint, but Pharmakon must treat it as **unconfirmed**.
- **The same physical read can be `ok` for receiving and `abstain` for selling.** Pharmakon
  must therefore always send the correct `flow`; never reuse a `receiving` extraction to
  auto-confirm a sale.
- Per-field overrides may be added under `abstention` in config (e.g. a higher τ for
  `num_lot` on selling); Pharmakon needs no change — the service applies them and the field
  simply comes back `abstain` when below its own bar.

---

## 7. Flow B — Achat / product-creation / inventory (receiving)

Throughput-oriented, looser abstention (τ `0.75`), human reviews a worksheet rather than a
single popup.

```
1. Operator scans a delivery (one or many vignettes).
2. Pharmakon → POST /batch?flow=receiving (images[], auth, idempotency-key)
              ◄── 202 { job_id, count, status:"queued" }
3. Pharmakon polls GET /jobs/{job_id} (or receives the callback) until status:"done".
4. For each ExtractionRecord, Pharmakon PREFILLS:
     • the LOT line: num_lot, date_fab, date_exp, quantity (operator types qty)
     • the PRODUCT: product_name/dci/dosage/forme/laboratoire (+ num_enregistrement anchor)
       — used to match an existing product or pre-create a new one
     • price block ppa/prix/shp with checksum verdict (informational on receiving)
5. Operator reviews the worksheet, resolves abstain/conflict/missing rows, and VALIDATES.
6. ===> ONLY THEN does Pharmakon write: create/match product, create lot, adjust stock. <===
```

Receiving differences from selling:
- **Async** (batch) by default — see `docs/SERVING.md` §4. A single receiving scan may use
  sync `/extract?flow=receiving`.
- **Looser τ** surfaces more values for the operator to accept, since a human is reviewing
  the whole worksheet anyway.
- Dispensing-critical conflicts (`dosage`/`forme`) are **still flagged, never auto-applied**
  — the dispensing-safe rule is independent of flow.
- Stock/product/lot writes happen only at **step 6**, after validation. VignOCR never writes.

---

## 8. Idempotency

- Every `POST /extract` (and `POST /batch`, when the async path lands) should carry a
  client-generated **`Idempotency-Key`** (UUID), sent as a form field today and/or header.
- **The VignOCR service is stateless and does not itself store keys.** It **echoes** the key
  back on the response (`Idempotency-Key` + an `X-Request-ID` header) and binds it to its
  structured logs so the **caller** can correlate a retry. A retried `/extract` is safe to
  send but **will re-run inference** — extraction is a pure read with no side effects, so this
  is harmless (just GPU cost). (`docs/SERVING.md` §6.)
- **Server-side de-duplication is the integration layer's responsibility.** If Pharmakon wants
  a retried scan to skip re-inference, it caches the prior `ExtractionRecord` against the key
  on its side, keyed by the echoed value.
- **Critically, idempotency must guard the write, not the extraction.** Extraction is
  side-effect-free; the dangerous step is **step 6** (sale / stock / lot write). Pharmakon
  must make *that* write idempotent on its own side so a double-confirm or retry never
  double-deducts stock. VignOCR cannot help here — it has no write access.

---

## 9. Error & abstention status codes

| HTTP | Meaning                                | Pharmakon action                                                  |
| ---- | -------------------------------------- | ----------------------------------------------------------------- |
| 200  | extraction returned                    | open prefill; honour `abstain`/`conflict`/`missing` (§5.1)        |
| 202  | batch accepted                         | poll `GET /jobs/{id}` or await callback                           |
| 400  | bad request (no `flow`/malformed form)  | fix client call                                                   |
| 401/403 | auth failure (when auth is enabled, SECURITY.md) | refresh token / check scope                              |
| 413  | upload too large                       | re-capture / downscale before resend (SECURITY.md)                |
| 415  | unsupported media type (bad MIME prefix)| send an accepted image MIME (SECURITY.md)                         |
| 422  | empty or undecodable image             | ask operator to re-capture; do **not** retry blindly             |
| 429  | rate-limited (Retry-After, when enabled)| back off and retry; **fall back to manual entry** for selling     |
| 503  | pipeline unavailable / busy            | back off and retry; fall back to manual entry                     |
| 5xx  | inference error (500)                  | show "scan failed — enter manually"; never block the sale on OCR  |

(415/413/422/503/500 are emitted by the current `/extract` handler; 400/401/403/429 apply
once request-validation/auth/rate-limit middleware is enabled per `docs/SECURITY.md`.)

**Abstention is not an error.** A `200` with fields in `abstentions` or
`status="conflict"`/`"missing"` is the *normal* safe path: the service is explicitly declining
to assert a low-confidence value, and the human fills the gap. Pharmakon must always allow the
operator to **complete the transaction by hand** if VignOCR abstains, errors, or is
unreachable — OCR is an accelerator, never a hard dependency of selling or receiving.

---

## 10. Invariants (Pharmakon must uphold)

1. **Extract → human-validate → THEN write.** No stock/price/lot write before validation, in
   either flow.
2. Always send the correct `flow`; **selling uses the stricter threshold** and a `receiving`
   extraction must never auto-confirm a sale.
3. Treat any `abstain` / `conflict` / `missing` field, `checksum.mismatch`, or
   `reimbursability` orange/unknown as **requires human attention** before confirm.
4. Parse money as **Decimal strings**; never float.
5. Send an `Idempotency-Key` on every call; guard your own stock writes separately.
6. OCR is an accelerator: manual entry is always available when the service abstains, errors,
   or is down.
7. VignOCR is never modified to fit Pharmakon — integrate against this schema only.
