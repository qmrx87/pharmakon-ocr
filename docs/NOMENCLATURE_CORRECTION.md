# VignOCR — Nomenclature Correction

How the `N°Enregistrement` (registration number) is used to **repair** the identity
fields read from a vignette against the official drug nomenclature, and — just as
important — what the engine is **forbidden** from touching.

> **Authority.** Every name, threshold, weight, and column below comes from config.
> Nothing here is hardcoded. The bindings:
> - `configs/nomenclature/correction.yaml` — CSV layout, normalization, match, policy
> - `configs/parsing/fields.yaml` — `num_enregistrement` structure, letter set, confusion map
> - `configs/classes.yaml` — `roles.identity_fields`, `roles.dispensing_critical`, `roles.never_correct`, `roles.anchor_field`
> - `docs/INTERFACES.md` — `NomenclatureReport`, `FieldRead`, the `nomenclature/` module API
>
> The module surface (from `INTERFACES.md`):
> ```python
> loader.load_csv(path) -> NomenclatureIndex
> match.find(norm_code, index, cfg) -> (row | None, confidence)
> correct.apply(fields: dict[str, FieldRead], row, cfg) -> (dict[str, FieldRead], NomenclatureReport)
> ```

---

## 1. Why correct at all?

The `num_enregistrement` is a **structured, checksum-like anchor** (`roles.anchor_field`)
that uniquely keys a drug in the national nomenclature. The free-text identity fields
(`product_name`, `dci`, `dosage`, `forme`, `laboratoire`) are far noisier under OCR than
a short rigid code. So we treat the *code* as the most reliable handle: read it, normalize
it, match it to a nomenclature row, and use that row to repair or cross-check the
free-text identity fields.

This is **strictly a dispensing-safety convenience**: it improves the prefill the cashier
sees, but it never changes the figures the pharmacy is legally and commercially bound by
(`ppa`, `tr`). See §6.

---

## 2. CSV schema

The nomenclature is delivered as an Excel workbook
(`NOMENCLATURE-VERSION-FEVRIER-2026-.xlsx`, per `configs/data.yaml: nomenclature.source_xlsx`)
and ingested to a flat CSV via `scripts/ingest_nomenclature.py`. The CSV is the runtime
artifact loaded by `loader.load_csv`.

Layout (`correction.yaml: csv`):

| Setting       | Value                                                                            |
| ------------- | -------------------------------------------------------------------------------- |
| `path`        | `fixtures/nomenclature.csv` (synthetic now; real CSV swaps in with no code change)|
| `key_column`  | `num_enregistrement`                                                             |
| `encoding`    | `utf-8`                                                                           |
| `columns`     | `num_enregistrement, product_name, dci, dosage, forme, laboratoire, tr`          |

```csv
num_enregistrement,product_name,dci,dosage,forme,laboratoire,tr
16/99/17D034/022,DOLIPRANE,PARACETAMOL,1000 mg,Comprimé,SANOFI,prise unique
18/97/14G061/003,AUGMENTIN,AMOXICILLINE + ACIDE CLAVULANIQUE,1 g,Comprimé,GSK,1 cp x 2/j
09/22F018/235,INEXIUM,ESOMEPRAZOLE,40 mg,Gélule,ASTRAZENECA,1 gél/j
```

Notes:
- The **key column is normalized on load** (same normalization as the read code, §3) so
  matching compares like with like. The loader builds a name-keyed index over the
  normalized key (`NomenclatureIndex`).
- `tr` ("taux de remboursement" / dispensing rule) is carried in the CSV **for display
  only**. It is on the `never_overwrite` list with `ppa` — the engine never writes it back
  onto a vignette read and never lets a vignette overwrite it (§6).
- Only the seven `columns` are read; any extra spreadsheet columns are ignored so the
  ingest stays forward-compatible.

---

## 3. Normalizing the read code

The detector + OCR produce a raw `num_enregistrement` string. Before matching we collapse
it to a single canonical form so that the dozens of cosmetic ways the same code can be
written all converge. Driven by `correction.yaml: normalize`:

| Step                    | Flag                    | Effect                                                        |
| ----------------------- | ----------------------- | ------------------------------------------------------------- |
| Uppercase               | `uppercase: true`       | `14g061` → `14G061`                                           |
| Strip internal spaces   | `strip_internal_spaces` | `14G 061`, `14 G061`, `14 G 061` → `14G061`                   |
| Glue letter block       | `glue_letter_block`     | preserves the `AA/BB/CC<LETTER>DDD/EEE` structure while gluing|
| Apply confusion map     | `apply_confusion_map`   | fix OCR digit↔letter slips on the **digit slots only**        |

### 3.1 The structure

From `fields.yaml: fields.num_enregistrement`, every code is:

```
AA / BB / CC <LETTER> DDD / EEE
└┬┘   └┬┘  └┬┘   │     └┬┘   └┬┘
 a     b    c   letter  d     e
```

- `a, b, c, d, e` are digit blocks (`a,b,c` two digits; `d,e` three digits).
- `<LETTER>` is one of the known **letter blocks** `[D, F, G, E, H, P, R, S, A, B, C, T]`
  (`fields.yaml: fields.num_enregistrement.letters`; extend as real data shows new ones).
- **Spacing around the letter varies** across vignettes — exactly why
  `strip_internal_spaces` + `glue_letter_block` exist.

Parsing regex (`fields.yaml`):
```
^\s*(?P<a>\d{2})\s*/\s*(?P<b>\d{2})\s*/\s*(?P<c>\d{2})\s*(?P<letter>[A-Z])\s*(?P<d>\d{3})\s*/\s*(?P<e>\d{3})\s*$
```
Canonical output (`normalized_format`): `{a}/{b}/{c}{letter}{d}/{e}` → e.g. `18/97/14G061/003`.

### 3.2 Confusion map (digit slots only)

OCR confuses lookalike glyphs. `fields.yaml: fields.num_enregistrement.confusion_map`:

```
O→0   I→1   l→1   S→5   B→8   Z→2
```

This is applied **only to the five digit blocks** (`a,b,c,d,e`), never to the `letter`
slot — the letter is the most identity-bearing token (it encodes the registration cohort)
and the same glyphs are *legitimate* letters there. Example: a raw read `1O/99/17DO34/O22`
normalizes to `10/99/17D034/022`.

`codes.normalize_enregistrement(text) -> str | None` (`parsing/`) performs this and returns
`None` when the string cannot be coerced into the structure — in which case the engine
treats the anchor as unreadable (§7).

---

## 4. Matching — structural edit distance with priors

Driven by `correction.yaml: match`. `match.find(norm_code, index, cfg)` returns
`(row | None, confidence)`.

- **Strategy:** `structural_edit_distance` — Levenshtein over the normalized code, but
  costs are **weighted per structural block** rather than uniform per character.
- **Library:** `rapidfuzz` (`fuzzy_lib: rapidfuzz`) — pure-CPU, no ML dependency, so this
  runs today.
- **Budget:** `max_edit_distance: 2` — at most two character repairs on the ~12-char code.
  Beyond that we refuse to guess.
- **Acceptance:** `min_match_confidence: 0.75`. Below this there is **no match**: OCR values
  are kept verbatim and the anchor is flagged (§7).

### 4.1 Block weights (the structural priors)

`match.block_weights`:

```yaml
a: 1.0   b: 1.0   c: 1.0   letter: 2.0   d: 1.0   e: 0.5
```

The intent:
- **`letter` costs 2.0** — it is the most identity-bearing block, so flipping it to make a
  match should be "expensive" and rarely chosen. A candidate that requires changing the
  letter is penalized hard.
- **`e` costs 0.5** — the trailing block is the most volatile and least identity-bearing, so
  a near-miss there is cheap to forgive.
- `a, b, c, d` are neutral (1.0).

`confidence` is derived from the weighted distance against the matched candidate (1.0 = exact
normalized match; decreasing as weighted edits accumulate), and is what populates
`NomenclatureReport.match_confidence`. A match is only returned when `confidence >=
min_match_confidence` **and** weighted edits stay within `max_edit_distance`.

---

## 5. The repair policy (per-field)

This is the **medical-safety core**. `correct.apply` walks each identity field and applies
the action assigned to it in `correction.yaml: policy`, using the matched `row`. The four
actions and the fields they govern:

| Policy bucket                 | Fields (`correction.yaml`)        | Behaviour                                                                                                                  |
| ----------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `never_overwrite`             | `ppa`, `tr`                       | **Forbidden.** Nomenclature must not change these. Vignette-specific / commercial. (§6)                                   |
| `repair_always`               | `product_name`, `laboratoire`     | Nomenclature is source of truth. The matched row value replaces the OCR value.                                            |
| `repair_if_ocr_low_or_agree`  | `dci`                             | Use the nomenclature value **only if** OCR abstained / was low-confidence (`< ocr_low_conf_threshold`) **or** already agrees. If OCR is confident and *disagrees*, keep OCR and record it. |
| `flag_on_conflict`            | `dosage`, `forme`                 | **Dispensing-critical.** On a confident OCR↔nomenclature disagreement, **DO NOT overwrite** — set `status="conflict"`, surface "à vérifier", and emit a conflict entry. (§5.1)             |

Supporting thresholds:
- `ocr_low_conf_threshold: 0.75` — the line below which an OCR read is considered "low" for
  the `dci` rule.
- `emit_conflict_report: true` — always populate `NomenclatureReport.conflicts`.

These field lists are **mirrored** in `classes.yaml: roles` (`identity_fields`,
`dispensing_critical`, `never_correct`) so parsing and correction agree on what each field
*is*. The correction engine reads them from `correction.yaml: policy`; both must stay
consistent (the two files are the contract).

### 5.1 The DISPENSING-SAFE rule (`dosage` / `forme`)

`dosage` and `forme` are in `roles.dispensing_critical`. A wrong dosage or wrong galenic
form on the prefill is a patient-safety hazard, so the engine is built to **never silently
overwrite them**:

1. OCR confident **and agrees** with nomenclature → `status="ok"`,
   `source="ocr+nomenclature"` (highest trust — both independent signals concur).
2. OCR low-confidence / abstained → keep the field as `abstain` (do **not** auto-fill from
   nomenclature; a human must look). It is *not* treated as agreement.
3. OCR confident **and disagrees** with nomenclature → **conflict**:
   `status="conflict"`, the OCR value is **kept** (not replaced), a
   `NomenclatureConflict{field, ocr, nomenclature, action="flagged"}` is appended, and the
   field is surfaced "à vérifier" for the human.

The cashier/operator resolves a conflict explicitly. Silent replacement is never an option
for these two fields.

### 5.2 `dci` — repair-if-low-or-agree

`dci` is identity-bearing but is in `repair_if_ocr_low_or_agree`, a middle ground:
- OCR low/abstained → fill from nomenclature (`status="corrected"`, `source="nomenclature"`).
- OCR agrees → `source="ocr+nomenclature"`.
- OCR confident and disagrees → **keep OCR**, record the disagreement as a conflict entry
  (`action="kept_ocr"`); it is *not* silently overwritten either, but unlike
  `dosage`/`forme` the field is not forced into the `conflict` status — it stays the OCR read.

### 5.3 `product_name` / `laboratoire` — repair always

These are descriptive, not dispensing-critical. The matched nomenclature row is canonical
spelling, so when there is a match the OCR value is replaced (`status="corrected"`,
`source="nomenclature"`). If OCR already matched, `source="ocr+nomenclature"`.

---

## 6. What is NEVER overwritten — `ppa` and `tr`

`correction.yaml: policy.never_overwrite: [ppa, tr]`, mirrored by
`classes.yaml: roles.never_correct: [ppa, tr]`. The nomenclature engine treats these as
read-only in both directions:

- **`ppa`** (public price) is a *vignette-specific* figure — it is printed on the physical
  vignette and validated by the **checksum** (`prix + shp == ppa`, `fields.yaml: checksum`),
  not by the nomenclature. The nomenclature CSV does not even carry a price column for this
  reason. `correct.apply` must skip `ppa` entirely; its value/`status`/`source` come solely
  from parsing + checksum.
- **`tr`** is carried in the CSV for *display*, but a vignette read never overwrites the
  CSV's `tr`, and the CSV's `tr` never overwrites a field on the extraction record (there is
  no `tr` field class — it has no `id` in `classes.yaml`; it exists only as a CSV column and
  a policy guard).

Rationale: prices and reimbursement terms are commercial/legal facts tied to the specific
vignette and the checksum invariant, **not** properties the drug-identity lookup is allowed
to assert. Mixing them would let a fuzzy code match silently rewrite money.

---

## 7. Match failure / unreadable anchor

If the code cannot be normalized (§3, `normalize_enregistrement` → `None`) **or** no
candidate clears `min_match_confidence`:

- `match.find` returns `(None, confidence)`.
- `correct.apply` makes **no changes** to any identity field — all OCR reads are kept
  verbatim.
- `NomenclatureReport` is emitted with `matched=false`,
  `num_enregistrement_normalized=<normalized-or-null>`, `match_confidence=<best score>`,
  `conflicts=[]`.
- The anchor field `num_enregistrement` is surfaced for human attention (its own
  parse/abstention status is unchanged; the operator is told the lookup did not resolve).

No best-effort substitution is performed below threshold — we would rather show the raw OCR
than a wrongly-matched drug identity.

---

## 8. Confidence & conflict reporting (the output)

`correct.apply` returns the (possibly updated) `fields` plus a `NomenclatureReport`
(`docs/INTERFACES.md`, `src/vignocr/serving/schemas.py`):

```python
class NomenclatureConflict(BaseModel):
    field: str
    ocr: str | None
    nomenclature: str | None
    action: Literal["flagged", "kept_ocr", "kept_nomenclature"]

class NomenclatureReport(BaseModel):
    matched: bool
    num_enregistrement_normalized: str | None
    match_confidence: float            # 0..1, from the weighted match (§4)
    conflicts: list[NomenclatureConflict]
```

Per-field outcomes are reflected on each `FieldRead`:

| Situation                                  | `FieldRead.status` | `FieldRead.source`     | conflict entry?            |
| ------------------------------------------ | ------------------ | ---------------------- | -------------------------- |
| repaired from nomenclature                 | `corrected`        | `nomenclature`         | no                         |
| OCR and nomenclature agree                 | `ok`               | `ocr+nomenclature`     | no                         |
| `dosage`/`forme` confident disagreement    | `conflict`         | `ocr` (value kept)     | yes, `action="flagged"`    |
| `dci` confident disagreement               | `ok` (OCR kept)    | `ocr`                  | yes, `action="kept_ocr"`   |
| OCR abstained, filled from nomenclature    | `corrected`        | `nomenclature`         | no                         |
| no match / unreadable code                 | unchanged (OCR)    | unchanged              | report `matched=false`     |

`confidence` on a corrected `FieldRead` is the match confidence (the trust we have in the
nomenclature row); on an agreement it is the higher of the OCR and match confidences. The
`conflicts` list is what the integration layer renders as "à vérifier" badges in the confirm
popup (`docs/INTEGRATION.md`), and what the audit log records as the machine's proposal
versus the human's resolution (`docs/SECURITY.md`).

---

## 9. Invariants (must always hold)

1. `ppa` and `tr` are **never** written by the nomenclature engine. (`never_overwrite`)
2. `dosage` and `forme` are **never silently overwritten**; a confident disagreement is a
   `conflict`, surfaced and logged. (`flag_on_conflict`, dispensing-safe)
3. Matching never accepts below `min_match_confidence` and never exceeds
   `max_edit_distance` weighted edits.
4. The `letter` block is the costliest to change (`block_weights.letter = 2.0`); the digit
   confusion map is applied to digit blocks only, never the letter.
5. Below threshold or unreadable → keep OCR verbatim, `matched=false`. No best-effort guess.
6. Every correction and every conflict is reported (`emit_conflict_report: true`) so the
   integration + audit layers can show and record it.
