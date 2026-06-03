# VignOCR — Security & Privacy

Security posture for the inference service: authenticated endpoints, strict upload
validation, safe file handling, secure short-retention storage, and **full audit logging of
every extraction and every human correction**. The service handles only the data printed on a
vignette — **no PII beyond what is on the vignette**.

> **Authority.** Endpoints, flows, and schema this hardens come from:
> - `docs/SERVING.md` — endpoints, stateless workers, container
> - `docs/INTEGRATION.md` — request/response, idempotency, error semantics
> - `src/vignocr/serving/schemas.py` — the records that get audited
> - `configs/` via `vignocr.common` — all limits/keys/paths are config/env-driven, never hardcoded
>
> Security knobs (auth mode, size/MIME limits, retention TTL, CORS, audit sink) are supplied
> via **env / config (12-factor)**; this document defines the policy, not literal secrets.

---

## 1. Threat model (scope)

| In scope                                                              | Out of scope (Pharmakon-side)                          |
| --------------------------------------------------------------------- | ------------------------------------------------------ |
| Unauthenticated access to `/extract`, `/batch`, `/jobs`               | Pharmakon's own auth / RBAC for cashiers               |
| Malicious / malformed uploads (oversized, wrong type, image bombs, polyglots) | Stock/ledger write authorization (Pharmakon owns step 6) |
| Leakage/retention of vignette images & extracted text                 | Network perimeter / VPC setup of the deployment        |
| Tampering with or loss of the audit trail                             | Patient EHR / prescription data (never sent here)      |
| Abuse / DoS via request floods                                        |                                                        |

`/health` and `/ready` are the **only** unauthenticated endpoints (liveness/readiness only;
they expose no vignette data) — see `docs/SERVING.md` §5.2.

---

## 2. Authenticated endpoints

> **Status.** This section is the **required policy** for any deployed instance. The current
> `src/vignocr/serving/app.py` ships the upload-validation and stateless-handling guarantees
> (§3–§5) but does **not** yet bundle auth/rate-limit middleware — that is added at deploy
> time (gateway or FastAPI dependency). Do not expose `/extract` publicly without it.

- **Every business endpoint requires auth.** `POST /extract`, `POST /batch`, and
  `GET /jobs/{id}` reject unauthenticated requests with `401`. Only `GET /health` and
  `GET /ready` are open.
- **Bearer credential.** Callers send `Authorization: Bearer <token>` (service token / signed
  JWT, mode via env). Tokens are validated on every request; expired/invalid → `401`,
  insufficient scope → `403` (`docs/INTEGRATION.md` §9).
- **Least privilege & scoping.** A token is scoped to a single Pharmakon tenant/pharmacy.
  `GET /jobs/{id}` returns a job **only** to the caller that created it (jobs are namespaced
  per authenticated principal) — no cross-tenant result reads.
- **Transport.** TLS (HTTPS) is mandatory for all traffic; non-TLS is refused. Tokens never
  travel in the URL/query string (headers only) so they don't land in access logs.
- **No secrets in the image or repo.** Keys/issuer config arrive via env at runtime
  (`docs/SERVING.md` §8); nothing sensitive is baked into the container or committed.
- **Rate limiting.** Per-principal rate limits protect the GPU pool; breaches return `429`
  with `Retry-After` (`docs/INTEGRATION.md` §9), and the client falls back to manual entry.

---

## 3. Upload validation (MIME / size / scan)

Uploads are validated **before** any inference. Limits come from config/env
(`VIGNOCR_ALLOWED_UPLOAD_PREFIX`, `VIGNOCR_MAX_UPLOAD_MB`), not hardcoded constants.

**Enforced today** (`src/vignocr/serving/app.py: _read_validated_upload` / `_decode_image`):

1. **MIME prefix check.** The `Content-Type` must start with the configured prefix (default
   `image/`), else `415`. *(Hardening: also sniff magic bytes against an explicit allow-list —
   JPEG/PNG/WebP — so a spoofed `Content-Type` cannot slip past; see "Recommended" below.)*
2. **Streaming size cap.** The body is read in bounded 1 MiB chunks and rejected with `413`
   the moment it exceeds `VIGNOCR_MAX_UPLOAD_MB`, so an oversized upload never gets fully
   buffered. An empty upload → `422`.
3. **Decode ("scan").** The bytes must decode as a real image (`PIL.Image.open(...).load()`)
   and are converted to RGB; an undecodable payload → `422` (`docs/INTEGRATION.md`: operator
   re-captures, no blind retry). `load()` forces a full decode, so a truncated/garbage file
   fails here rather than downstream.

**Recommended hardening** (policy for the production deployment):

4. **Pixel-count bound (decompression-bomb guard).** Cap `width × height` (and set
   `Image.MAX_IMAGE_PIXELS`) so a tiny file that expands to gigapixels is rejected with `422`
   before it exhausts memory.
5. **Anti-polyglot.** Reject files that are valid images *and* something else (image+HTML/
   script). Only the decoded pixel payload is ever used; no uploaded bytes are interpreted as
   code, markup, or a path.
6. **Multipart hygiene.** Only the expected parts (`file`, `flow`, `idempotency_key`) are
   read; unexpected parts are ignored; the client-supplied filename is treated as untrusted
   data and never used to form a server path (§4).

---

## 4. Safe file handling

- **In-memory / sandboxed temp only.** Decoding happens in memory or in an isolated temp dir
  with a service-generated random name. The client-supplied `filename` is **never** used to
  build a server path (no path traversal, no overwrite).
- **No execution, ever.** Uploaded bytes are only ever passed to the image decoder + pipeline.
  They are never written to an executable location, never `eval`'d, never used as a command
  argument.
- **Deterministic cleanup.** Any temp artifact is deleted in a `finally` — including on
  error/timeout. Workers are **stateless** (`docs/SERVING.md` §6): nothing persists on the box
  after the request beyond the configured retention store (§5).
- **Bounded resources.** Per-request CPU/GPU time and memory are bounded; a hard timeout
  returns a structured `5xx`/`503` (`docs/INTEGRATION.md`) rather than hanging a worker.
- **Resource isolation.** GPU workers run as a non-root user in the container with a read-only
  application filesystem where possible; only the temp dir and audit sink are writable.

---

## 5. Secure storage & retention

- **Minimize.** The service's job is to return an `ExtractionRecord`, not to be a document
  store. By default it retains **nothing** persistent of the image beyond the in-flight
  request; async batch results live only until fetched/expired.
- **Bounded retention.** Where transient storage is needed (async job results, idempotency
  de-dup window), it has a **configurable TTL** and is purged on expiry. The idempotency key
  and its cached record expire after the configured window (`docs/INTEGRATION.md` §8).
- **Encryption.** Any at-rest store (object storage for async results, audit log sink) is
  encrypted at rest; all transport is TLS.
- **Access-controlled.** Stored artifacts and audit logs are reachable only by the service
  identity and authorized operators; job results are namespaced per principal (§2).
- **Right to purge.** Because retention is short and keyed, a tenant's transient data ages out
  automatically; there is no long-lived corpus of vignette images accumulating by default.
- **Model artifacts** (ONNX detector/recognizer, nomenclature CSV) are pulled from controlled
  object storage at boot (`docs/SERVING.md` §6); their versions are recorded in
  `model_versions` for traceability.

---

## 6. Audit logging — every extraction and every correction

A complete, tamper-evident audit trail is a **hard requirement** (medical-adjacent, money on
the vignette). Two event classes are logged:

### 6.1 Extraction events (machine output)

On every `/extract` (and every record in a `/batch`), append an audit entry:

| Field                      | Source                                                    |
| -------------------------- | -------------------------------------------------------- |
| `event: "extraction"`      | —                                                        |
| `timestamp` (UTC)          | server                                                   |
| `principal` / tenant       | auth token (§2)                                          |
| `idempotency_key`          | request header                                           |
| `image_id`                 | `ExtractionRecord.image_id`                              |
| `flow`                     | `selling` / `receiving`                                  |
| `fields` summary           | per field: `value`, `status`, `source`, `confidence`    |
| `checksum.verdict`         | `ChecksumReport`                                         |
| `nomenclature.matched` + `conflicts` | `NomenclatureReport`                           |
| `abstentions`              | list of abstained field names                            |
| `reimbursability`          | colour + eligibility                                     |
| `model_versions`           | detector / recognizer / nomenclature_version             |
| `timings_ms`               | per-stage + total                                        |

This makes every machine assertion reproducible and reviewable: *what* was read, *how
confident*, *which model*, *what the nomenclature proposed*, and *what was flagged*.

### 6.2 Human-correction events (HITL)

Whenever a human overrides or confirms a flagged value, a correction event is recorded so the
human-in-the-loop decision is auditable and feeds retraining:

| Field                      | Meaning                                                              |
| -------------------------- | ------------------------------------------------------------------- |
| `event: "correction"`      | —                                                                   |
| `timestamp` (UTC), `principal` | who corrected, when                                             |
| `image_id`, `field`        | which extraction / which field                                       |
| `machine_value` + `machine_status` + `machine_confidence` | what VignOCR proposed                |
| `human_value`              | what the operator entered/confirmed                                  |
| `was_conflict` / `was_abstain` | whether the field had been flagged                              |
| `nomenclature_value`       | the nomenclature suggestion, if any (for conflict resolutions)       |

> **Where corrections are emitted.** The human validates inside Pharmakon
> (`docs/INTEGRATION.md` §5/§7). Pharmakon reports the resolved correction back to VignOCR's
> audit sink (the correction-report contract is part of the integration), so the **machine
> proposal ↔ human resolution** pair is captured end-to-end. This repo defines the schema and
> the sink; it does not implement Pharmakon's UI.

### 6.3 Audit integrity

- **Append-only.** The audit sink is write-once / append-only; entries are never mutated or
  deleted within the retention window (longer than transient image retention).
- **Tamper-evident.** Entries carry a monotonic sequence/timestamp; the sink supports
  integrity verification (e.g. hash-chaining) so missing or altered entries are detectable.
- **No image bytes in the audit log.** The audit trail records the *structured* extraction +
  corrections (text/status/confidence), referenced by `image_id` — not the raw image — so the
  trail stays small and the minimization rule (§7) holds.

---

## 7. Data minimization — no PII beyond the vignette

- The service ingests **only** the vignette image and returns **only** the fields defined in
  `configs/classes.yaml` (drug identity, lot/dates, prices, registration code,
  reimbursability). These are **product/commercial** data printed on the box — **not patient
  data**.
- **No patient identifiers, prescriptions, or EHR data are sent to, processed by, or stored by
  VignOCR.** If a vignette photo incidentally includes surrounding content, only the detected
  field crops are used by the pipeline; the service does not extract or persist anything
  outside the schema.
- `tr` (dispensing rule) and prices are handled as on the vignette/nomenclature and never
  enriched with external personal data (`docs/NOMENCLATURE_CORRECTION.md` §6).
- This keeps VignOCR **out of scope for patient-data regimes**: its data class is
  drug-product + price, retained briefly, audited structurally.

---

## 8. Invariants

1. All business endpoints (`/extract`, `/batch`, `/jobs`) are **authenticated** over TLS; only
   `/health` and `/ready` are open and expose no vignette data.
2. Uploads are validated (**MIME allow-list, size cap, decode/scan, anti-polyglot**) before
   any inference; the client filename is never trusted to form a path.
3. Workers are stateless; temp files are sandboxed and deleted in `finally`; nothing executes
   uploaded bytes.
4. Persistent retention is **minimal and TTL-bounded**, encrypted at rest, access-controlled,
   and namespaced per principal.
5. **Every extraction and every human correction is audit-logged**, append-only and
   tamper-evident, without storing raw image bytes.
6. **No PII beyond what is on the vignette** is processed or stored — drug-product + price
   only, never patient data.
7. All limits, keys, retention, and CORS are **config/env-driven**; no secrets in the repo or
   image.
