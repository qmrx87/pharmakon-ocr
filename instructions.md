# Project Instructions — VignOCR (MANDATORY)

> These rules are **binding** for every contributor and every AI agent working in this
> repository. They exist to keep `AUDIT.md` the project's single source of truth and to
> prevent unvalidated implementation. Read this file and `AUDIT.md` before doing anything.

---

## 1. `AUDIT.md` is the single source of truth

[`AUDIT.md`](AUDIT.md) is the authoritative **Cahier des Charges / system specification /
architecture decision record / knowledge base / AI-context document**. When this file,
`README.md`, `docs/*`, `memory/*`, code comments, or your own assumptions disagree with
`AUDIT.md`, **`AUDIT.md` governs** — and the discrepancy must be reconciled (update the
loser, log it in the AUDIT Findings Register §17).

## 2. `AUDIT.md` MUST be kept current (non-negotiable)

You **must** update `AUDIT.md` in the **same change** whenever you:

- change architecture or a public contract/interface;
- change business requirements or workflows;
- add, remove, refactor, or re-scope a module;
- change datasets, annotations, the schema (`configs/classes.yaml`), or the nomenclature;
- discover a new constraint, risk, or finding;
- resolve or open a Finding (§17), or change implementation status (§18).

When you update `AUDIT.md`:
1. Bump the **Document Version** and **Last Update Date** on the cover page.
2. Add a row to the **Revision History**.
3. Update the **Implementation Status Tracker (§18)** and the **Findings Register (§17)**.
4. Keep the **Table of Contents** and **Table of Figures** aligned with the content.
5. If a stale doc caused the change, reconcile it and note it under finding **F-doc**.

A change that touches behavior/architecture/data **but not `AUDIT.md`** is **incomplete**
and must not be merged.

## 3. Understand before you build

The first responsibility on any task is to **fully understand the existing system** using
`AUDIT.md` + the verified facts it cites. **Do not assume.** If information is missing,
**add a clarification question to `AUDIT.md` §19** and surface it — do not invent an answer.

## 4. Interactive validation BEFORE implementation (hard gate)

**Never proceed directly to implementation, refactor, redesign, or optimization.** After
producing any audit, architecture proposal, implementation plan, or major design change:

1. **Summarize** the decisions and their rationale.
2. **Ask** targeted clarification questions (record them in §19).
3. **Validate** the affected business workflows (§14).
4. **Validate** the architectural assumptions (§6, §10, §13).
5. **Wait for explicit approval.**

Only **after approval** does implementation begin. The current baseline (`AUDIT.md` v1.0)
is **DRAFT — awaiting validation**; §19 must be approved before coding resumes.

## 5. Preserve the safety invariants

These are load-bearing and must never be weakened without explicit, documented approval:

- **Money is `decimal.Decimal` end-to-end**, serialized as a centime-quantized string —
  never `float`.
- **`prix + shp == ppa`** to the centime (verify · repair · **flag**; never silently accept
  a mismatch).
- **Nomenclature MUST NEVER overwrite `ppa` or `tr`**, and MUST NEVER silently overwrite a
  dispensing-critical field (`dci`, `dosage`, `forme`) on a confident conflict — it **flags**.
- **PPA and TR are captured by OCR from the vignette** — the national nomenclature has **no
  price column** (Finding F-04). Never back-fill price from the nomenclature.
- **Abstain over guess.** Selling (τ=0.90) is stricter than receiving (τ=0.75).
- **Extract → human-validate → THEN write.** VignOCR never writes to inventory/ledgers.

## 6. Config-driven, no hardcoding

Class names, paths, regexes, thresholds, hyperparameters, and policy live in `configs/`
(read via `vignocr.common`). Do not hardcode them in `src/`. `configs/classes.yaml` is the
**single** definition of the field schema.

## 7. Heavy ML libs are lazy-imported

The deterministic core (parsing, nomenclature, pipeline, serving) runs on **CPU without**
`torch`/`rfdetr`/`paddleocr`/`onnxruntime`/`cv2`. Import heavy libs **inside** the functions
that use them, behind the `[ml]` extra. Keep the core importable and testable on CPU.

## 8. Keep project memory honest

`memory/*` captures institutional knowledge. When a memory note is contradicted by verified
fact, **correct the note** and cite the AUDIT finding. (Example already applied: the earlier
"ppa/tr filled from nomenclature" note is wrong — corrected per F-04.)

---

### Quick checklist before opening a PR
- [ ] Did I read `AUDIT.md` for the affected area?
- [ ] Did I update `AUDIT.md` (version, revision history, §17, §18, ToC/figures)?
- [ ] Did I preserve every §5 safety invariant?
- [ ] Did I add clarification questions instead of assuming?
- [ ] For any design/architecture change: did I run the §4 validation gate and get approval?
