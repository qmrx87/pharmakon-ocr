# VignOCR — Serving Architecture

How the trained pipeline is served: a cloud-GPU inference service with a **synchronous
low-latency path** for the point-of-sale selling popup and an **asynchronous queued path**
for batch receiving/inventory. Workers are stateless; everything is config-driven (12-factor).

> **Authority.** Endpoint contract, schema, and abstention behaviour come from:
> - `docs/INTERFACES.md` — endpoints + `serving/` surface + the `ExtractionRecord` schema
> - `src/vignocr/serving/schemas.py` — the Pydantic models emitted on the wire
> - `configs/parsing/fields.yaml: abstention` — selling vs receiving thresholds
> - `configs/classes.yaml` — the field schema the response is built from
> - `docs/SECURITY.md` — auth, upload validation, audit
> - `docs/INTEGRATION.md` — the Pharmakon-side flows that drive the two paths
>
> No class names, thresholds, or paths are hardcoded in the service; they load via
> `vignocr.common` from `configs/`.

---

## 1. Design goals

| Goal                          | How it is met                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| **Sub-~2 s selling popup**    | sync `POST /extract` on warm GPU workers; one image; no queue hop. (§3, §6)                |
| **High-throughput receiving** | async submit → queue → worker pool → poll/callback; batches of vignettes. (§4)             |
| **Stateless GPU workers**     | no session state, no local writes of business data; models loaded at boot from object store|
| **Config-driven**             | model paths, thresholds, CORS, limits all from env + `configs/` (12-factor)                |
| **Safe-by-default**           | abstention surfaced, never silently guessed; selling threshold stricter than receiving     |
| **Horizontally scalable**     | autoscale on queue depth (async) and concurrency/latency (sync); health/readiness gated    |

---

## 2. Topology

```
                         ┌───────────────────────────────────────────────┐
   Pharmakon backend     │                 VignOCR service                │
   (selling popup,       │                                               │
    receiving, etc.)     │   ┌────────────┐        ┌──────────────────┐  │
        │                │   │  API tier  │  sync  │  GPU worker pool  │  │
        │  HTTPS + auth  │   │  FastAPI   ├───────►│  (stateless)      │  │
        ├───────────────►│   │  (CPU,     │        │  RF-DETR + OCR    │  │
        │   POST /extract│   │   many     │        │  + parsing +      │  │
        │   (sync)       │   │   replicas)│        │  nomenclature     │  │
        │                │   │            │  async │  (ONNX on GPU)    │  │
        │  POST /batch   │   │            ├──┐     └──────────────────┘  │
        ├───────────────►│   └────────────┘  │            ▲              │
        │  GET /jobs/{id}│          │         ▼            │             │
        │                │          │   ┌──────────┐  ┌────┴─────────┐   │
        │                │          └──►│  queue   │─►│ batch workers│   │
        │                │              │ (broker) │  │ (stateless)  │   │
        │                │              └──────────┘  └──────────────┘   │
        │                │   models pulled at boot from object store      │
        │                │   (ONNX detector + recognizer + nomenclature   │
        │                │    CSV, versioned)                             │
        └────────────────┘                                               │
                         └───────────────────────────────────────────────┘
```

- **API tier** is CPU-only, cheap, and scales wide. It authenticates, validates the upload
  (`docs/SECURITY.md`), and either calls a GPU worker synchronously or enqueues a job.
- **GPU workers** run the `pipeline.orchestrator.VignocrPipeline` — detect → orient+crop →
  recognize → parse/checksum/PPA → nomenclature correct → reimbursability → assemble
  `ExtractionRecord`. They hold **no business state**.
- **Queue/broker** decouples the bursty receiving workload from the latency-sensitive
  selling workload.

---

## 3. Synchronous path — the selling popup

Used by the Vente prefill-and-confirm flow (`docs/INTEGRATION.md` §Vente). A single vignette
is scanned at the till; the cashier waits on a confirm popup, so latency matters.

```
POST /extract?flow=selling     (multipart: file=<image>, idempotency_key=<uuid>)
Authorization: Bearer <token>                                   (when auth enabled)
                       │
                       ▼  auth + MIME-prefix/size/decode validation (SECURITY.md)
                       ▼  warm GPU worker runs the full pipeline
                       ▼  abstention profile = selling (τ=0.90, STRICTER)
                       ◄── 200  ExtractionRecord (JSON)
                            + headers: Idempotency-Key (echoed), X-Request-ID
```

- **`flow=selling`** selects the stricter abstention profile (`fields.yaml:
  abstention.selling.default = 0.90`). Anything below threshold comes back as
  `status="abstain"` and lands in `ExtractionRecord.abstentions` — it is **not** guessed.
- **Latency budget:** target end-to-end p95 ≈ **2 s** (network + validation + single-image
  GPU inference). Achieved by keeping workers warm (models resident in GPU memory) and never
  putting the selling request behind the batch queue.
- **No stock side effects.** The service only *extracts*. Stock deduction happens in
  Pharmakon **after** the cashier validates the popup (`docs/INTEGRATION.md`). The service is
  read-only with respect to inventory.

---

## 4. Asynchronous path — batch receiving / inventory

Used by Achat / product-creation / inventory (`docs/INTEGRATION.md` §Achat). Many vignettes
arrive together; throughput matters more than per-item latency, and the operator is not
blocking on a single popup.

```
POST /batch?flow=receiving     (multipart: images[]=<files>  OR  a manifest)
Authorization: Bearer <token>
Idempotency-Key: <uuid>
   ◄── 202 Accepted  { "job_id": "...", "count": N, "status": "queued" }

GET /jobs/{job_id}
   ◄── 200  { "job_id": "...", "status": "running|done|error",
              "done": k, "total": N,
              "results": [ ExtractionRecord, ... ] }      # when done
```

- Optionally a **callback URL** (webhook) can be registered per job so Pharmakon is notified
  on completion instead of polling — same `ExtractionRecord` payload.
- **`flow=receiving`** selects the looser profile (`fields.yaml:
  abstention.receiving.default = 0.75`) — a human reviews the receiving worksheet, so the
  service surfaces more reads rather than abstaining aggressively.
- Batch jobs run on the **batch worker pool**, which autoscales on **queue depth** and can
  scale to zero between deliveries; the selling pool stays warm independently.

---

## 5. Endpoint contract

The canonical surface (from `docs/INTERFACES.md`), plus the async additions:

| Method & path        | Purpose                          | Auth | Body / params                                        | Response                                   |
| -------------------- | -------------------------------- | ---- | ---------------------------------------------------- | ------------------------------------------ |
| `GET  /health`       | **liveness** (process up)        | none | —                                                    | `{"status":"ok"}`                          |
| `GET  /ready`        | **readiness** (models loaded)    | none | —                                                    | `{"ready", "models", "flow_default", "stub"}` |
| `POST /extract`      | sync single-vignette extraction  | yes¹ | multipart `file`, `flow`, `idempotency_key`          | `ExtractionRecord` (+ echoed key header)   |
| `POST /batch` *(planned)*    | async batch submit       | yes¹ | multipart `images[]`/manifest; `flow`                | `202 {job_id, count, status}`              |
| `GET  /jobs/{id}` *(planned)*| async job status/result  | yes¹ | —                                                    | `{job_id, status, done, total, results[]}` |

¹ Auth is the documented requirement (`docs/SECURITY.md`); enforced once the auth middleware
is enabled. `/health` and `/ready` are always open. `/batch` and `/jobs` are planned (§4).

`flow` is `selling | receiving` (`Flow` literal in `schemas.py`); it selects the abstention
profile and is echoed back on every `ExtractionRecord.flow`.

> **Implemented today vs. planned.** The FastAPI app (`src/vignocr/serving/app.py`) ships
> `GET /health`, `GET /ready`, and `POST /extract` — the sync surface in `docs/INTERFACES.md`.
> The **async `POST /batch` + `GET /jobs/{id}` + queue/worker-pool design (§4) is the planned
> receiving architecture**, not yet in code. Until it lands, a single receiving scan uses
> `POST /extract?flow=receiving`, and batches are driven client-side as a loop of `/extract`
> calls. The default `flow` when omitted is `selling` (the stricter profile), overridable via
> `VIGNOCR_DEFAULT_FLOW`.

### 5.1 Response body — `ExtractionRecord`

Emitted verbatim from `src/vignocr/serving/schemas.py` (mirrored in `docs/INTERFACES.md`).
The serving layer **does not invent fields** — it returns what the pipeline assembled:

```python
class ExtractionRecord(BaseModel):
    image_id: str
    fields: dict[str, FieldRead]            # keyed by classes.yaml field name
    reimbursability: Reimbursability        # color_band head, not OCR
    checksum: ChecksumReport                # prix + shp == ppa verdict
    nomenclature: NomenclatureReport        # match + conflicts
    abstentions: list[str]                  # field names with status == "abstain"
    flow: Literal["selling", "receiving"]   # which abstention profile was applied
    model_versions: dict[str, str]          # {detector, recognizer, nomenclature_version}
    timings_ms: dict[str, float]
```

`model_versions` lets Pharmakon (and the audit log) pin exactly which detector/recognizer/
nomenclature produced a record — essential for reproducibility and incident review
(`docs/SECURITY.md`). A stub request/response pair lives in `docs/INTEGRATION.md`.

### 5.2 Health vs readiness

- **`/health`** — liveness only. Returns `200 {"status":"ok"}` as soon as the process is up,
  *before* models are loaded. Used by the orchestrator to decide whether to restart the
  container. Must never touch the GPU or models (so a model-load hang doesn't kill a healthy
  process via the liveness probe).
- **`/ready`** — returns `{"ready": true, "models": {detector, recognizer,
  nomenclature_version}, "flow_default": "selling", "stub": false}` once the pipeline
  singleton is built (models loaded / warm). If the real pipeline cannot load and stubbing is
  disabled (`VIGNOCR_ALLOW_STUB=0`), it returns `503` with `ready=false`. `stub=true` signals
  the deterministic stub is serving (CPU-only / no weights) — honest so operators aren't
  fooled. The load balancer must **not** route `/extract` (or planned `/batch`) traffic to a
  replica until `/ready` is true, so cold replicas never serve the selling popup unloaded.

---

## 6. Stateless GPU workers

- **No business state.** Workers hold model weights + config only. They never persist
  vignettes, results, or PII locally beyond the in-flight request (`docs/SECURITY.md`
  retention). Any temp file is deleted in a `finally`.
- **Models at boot.** Each worker pulls its versioned ONNX detector, ONNX/recognizer, and the
  nomenclature CSV from object storage at startup (paths/URIs via env), loads them onto the
  GPU, runs a warm-up, then flips `/ready`. `model_versions` is populated from these
  artifacts.
- **Idempotency key is echoed, not stored.** The worker is fully stateless: it does **not**
  persist `Idempotency-Key`s. The key is bound to the structured logs and echoed back on the
  response (header + `X-Request-ID`) so the **caller** can correlate and de-duplicate retries.
  Server-side de-duplication, if desired, is the integration layer's responsibility
  (`docs/INTEGRATION.md` §8). A retried selling scan is therefore safe to send but will
  re-run inference.
- **Interchangeable.** Because workers are identical and stateless, the pool can be scaled,
  recycled, or rescheduled freely; a crash loses only in-flight requests, which the client
  retries with the same idempotency key.

---

## 7. Autoscaling & resilience

| Pool             | Scale signal                                   | Floor / ceiling                                  |
| ---------------- | ---------------------------------------------- | ------------------------------------------------ |
| API tier (CPU)   | request concurrency / CPU                      | small floor ≥ 2 for HA; scale wide               |
| Selling GPU pool | in-flight sync concurrency + p95 latency       | **warm floor ≥ 1** so the popup is never cold    |
| Batch GPU pool   | queue depth / oldest-message age               | floor may be **0** between deliveries            |

- **Selling is protected from receiving.** Separate pools (or strict priority) ensure a large
  receiving batch can never starve the latency-sensitive selling popup of GPU.
- **Backpressure.** If the selling pool is saturated, the API returns `503` with `Retry-After`
  rather than queueing a sync request behind a batch (the popup must stay fast or fail fast).
- **Graceful drain.** On scale-in / deploy, a worker stops accepting new work, finishes
  in-flight requests, then exits. Readiness flips false first so the LB stops routing.
- **Timeouts.** Sync `/extract` has a hard server-side timeout; on breach it returns a
  structured error (`docs/INTEGRATION.md` error semantics) so the cashier sees "scan failed,
  enter manually", never a hang.

---

## 8. Container

A single image runs in two roles (API or worker) selected by env, so build once / deploy
twice:

- **Base:** CUDA runtime + Python; installs the package with the `[ml]` extra
  (`pip install -e .[ml]`) for torch/onnxruntime-gpu etc. The **core** still imports on a
  CPU-only base (lazy ML imports per the global rules), which is what CI and the API tier use.
- **Entrypoint:** `uvicorn` for the API role; the worker-loop for the GPU role
  (`ROLE=api|worker` via env).
- **Config:** strictly via env (`VIGNOCR_*`, model URIs, `ALLOWED_ORIGINS`, abstention
  overrides if any, broker URL). No secrets baked into the image (`docs/SECURITY.md`).
- **Probes:** container/orchestrator liveness → `/health`; readiness → `/ready`.
- **CPU fallback.** Because ML libs are lazy-imported, the same image can run the pipeline
  with **stubbed** detector/recognizer on CPU (deterministic fixture readers) for smoke tests
  and local dev, exactly as the pipeline contract allows (`docs/INTERFACES.md`).

---

## 9. Invariants

1. The service **only extracts**; it never writes stock or business data. Stock effects are
   Pharmakon's, after human validation (`docs/INTEGRATION.md`).
2. `flow=selling` is always the **stricter** abstention profile; `flow=receiving` looser
   (`fields.yaml: abstention`). Below threshold → `abstain`, surfaced, never guessed.
3. Money is a Decimal **string** in every response (`schemas.py: money_str`), never a float.
4. Workers are stateless; readiness gates traffic; selling GPU never starves behind batch.
5. Every response carries `model_versions` + `timings_ms` for reproducibility and audit.
6. All config (thresholds, paths, CORS, limits) is env/`configs/`-driven — nothing hardcoded.
