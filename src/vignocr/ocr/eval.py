"""Evaluate the recognizer: CER per field type + abstention quality.

``run(cfg_path, ckpt=None, split=None) -> dict`` runs the configured
:class:`~vignocr.ocr.infer.Recognizer` over the gold field crops of a split and
reports, per field *type* and overall:

* **CER** — character error rate (Levenshtein distance / gold length), the
  Phase-5 gate metric (``<= 5%`` on business-critical fields, per the config).
* **abstention precision** — of the reads the recognizer did *not* abstain on,
  the fraction that are exactly correct. A high value means "when we commit, we
  are right"; the complement is the cost of wrong auto-fills.
* **abstention rate** — fraction of reads sent to human review.

The recognizer lazy-imports its backend; CER is computed with ``rapidfuzz``
(a core dep), so this module needs no ``[ml]`` extra to *import*. There is **no
fixture/stub**: it evaluates against whatever the active dataset provides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from vignocr.common import get_active_dataset, get_classes, get_logger, load_config, seed_everything
from vignocr.common.config import load_yaml  # not re-exported from the package facade
from vignocr.ocr.infer import Recognizer

log = get_logger(__name__)


def run(
    cfg_path: str | None = None,
    ckpt: Path | None = None,
    split: str | None = None,
    *,
    flow: str | None = None,
) -> dict[str, Any]:
    """Evaluate recognition on a split and return a metrics dict.

    Args:
        cfg_path: OCR recognition config path; ``None`` loads ``configs/ocr/recognition.yaml``.
        ckpt: optional fine-tuned checkpoint directory to load into the backend
            (sets ``rec_model_dir`` for the baseline). ``None`` uses the config's model.
        split: dataset split name; ``None`` uses the active dataset's ``val`` split.
        flow: abstention profile to evaluate under (``selling``/``receiving``);
            ``None`` uses the config default. Selling is stricter.

    Returns:
        ``{"overall": {...}, "per_type": {type: {...}}, "per_field": {name: {...}},
        "gate": {...}, "n": int}`` where each metrics block has ``cer``,
        ``abstention_precision``, ``abstention_rate``, and counts.
    """
    cfg = load_yaml(cfg_path) if cfg_path else load_config("ocr/recognition")
    seed_everything(int(cfg.get("train", {}).get("seed", 1337)))

    if ckpt is not None:
        # Point the baseline backend at the fine-tuned recognition model.
        cfg = _with_checkpoint(cfg, ckpt)

    dataset = get_active_dataset()
    split = split or dataset["splits"]["val"]
    schema = get_classes()
    business_critical = set(schema.business_critical_fields)

    recognizer = Recognizer(cfg)
    examples = _collect_labeled_crops(dataset, split, schema)
    log.info("ocr.eval.start", dataset=dataset["name"], split=split, examples=len(examples))

    # Accumulators keyed by field type and field name.
    agg_type: dict[str, _Acc] = {}
    agg_field: dict[str, _Acc] = {}
    overall = _Acc()
    bc_acc = _Acc()  # business-critical-only (the gate)

    for ex in examples:
        gold = ex["label"]
        if gold is None:
            # No gold transcription -> can't score CER for this crop; skip it
            # rather than counting an unverifiable read.
            continue
        read = recognizer.read(
            ex["image"], ex["type"], ex["orientation"], flow=flow, field_name=ex["name"]
        )
        pred = read.value or ""
        abstained = read.status == "abstain"

        for acc in (
            overall,
            agg_type.setdefault(ex["type"], _Acc()),
            agg_field.setdefault(ex["name"], _Acc()),
        ):
            acc.add(gold, pred, abstained)
        if ex["name"] in business_critical:
            bc_acc.add(gold, pred, abstained)

    cer_gate = float(cfg.get("eval", {}).get("cer_gate", 0.05))
    result = {
        "n": overall.n,
        "overall": overall.metrics(),
        "per_type": {t: a.metrics() for t, a in sorted(agg_type.items())},
        "per_field": {f: a.metrics() for f, a in sorted(agg_field.items())},
        "business_critical": bc_acc.metrics(),
        "gate": {
            "cer_gate": cer_gate,
            "business_critical_cer": bc_acc.metrics()["cer"],
            "passed": (bc_acc.n > 0 and bc_acc.metrics()["cer"] <= cer_gate),
        },
    }
    log.info(
        "ocr.eval.done",
        overall_cer=result["overall"]["cer"],
        bc_cer=result["business_critical"]["cer"],
        gate_passed=result["gate"]["passed"],
    )
    return result


# --------------------------------------------------------------------------- #
# Metric accumulator
# --------------------------------------------------------------------------- #


class _Acc:
    """Accumulate CER + abstention stats for one bucket (type / field / overall)."""

    def __init__(self) -> None:
        self.n = 0
        self.edit_sum = 0
        self.char_sum = 0
        self.n_abstain = 0
        self.n_committed = 0  # reads we did NOT abstain on
        self.n_committed_correct = 0  # of those, exactly correct

    def add(self, gold: str, pred: str, abstained: bool) -> None:
        self.n += 1
        self.edit_sum += Levenshtein.distance(pred, gold)
        self.char_sum += max(1, len(gold))  # avoid /0 on an empty gold string
        if abstained:
            self.n_abstain += 1
        else:
            self.n_committed += 1
            if pred == gold:
                self.n_committed_correct += 1

    def metrics(self) -> dict[str, Any]:
        cer = (self.edit_sum / self.char_sum) if self.char_sum else 0.0
        abst_rate = (self.n_abstain / self.n) if self.n else 0.0
        abst_prec = (self.n_committed_correct / self.n_committed) if self.n_committed else None
        return {
            "cer": cer,
            "abstention_rate": abst_rate,
            "abstention_precision": abst_prec,
            "n": self.n,
            "n_abstain": self.n_abstain,
            "n_committed": self.n_committed,
            "n_committed_correct": self.n_committed_correct,
        }


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


def _collect_labeled_crops(
    dataset: dict[str, Any], split: str, schema: Any
) -> list[dict[str, Any]]:
    """Crop every annotated text field in ``split``, attaching gold labels.

    Uses the ``data`` subpackage's COCO API, lazy-imported so this entrypoint
    imports without the data deps resolved. Gold transcriptions live in the
    fixture's ``ground_truth.json`` (keyed by image ``file_name`` -> ``fields``);
    we join them to crops by field name. Real (un-fixtured) datasets without a
    ground-truth sidecar yield ``label=None`` crops, which the CER loop skips.
    """
    from vignocr.data import coco  # noqa: PLC0415

    root = Path(dataset["root"])
    split_data = coco.load_split(root, split)
    gold_by_image = _load_ground_truth(root)
    # Map COCO category ids -> names via the file's OWN categories (robust to
    # whatever ids a Roboflow export assigns; never a hardcoded id table).
    cat_map = split_data.cat_id_to_name

    out: list[dict[str, Any]] = []
    for image in split_data.images:
        img_path = split_data.image_path(image)
        anns = split_data.annotations_for(int(image["id"]))
        gold_fields = gold_by_image.get(image["file_name"], {}).get("fields", {})
        by_field = coco.crops_for_image(img_path, anns, cat_map)
        for field_name, crops in by_field.items():
            spec = schema.by_name(field_name)
            if spec["type"] == "region":  # not text
                continue
            for crop in crops:
                out.append(
                    {
                        "name": field_name,
                        "type": spec["type"],
                        "orientation": spec["orientation"],
                        "image": crop.image,
                        "label": gold_fields.get(field_name),
                    }
                )
    return out


def _load_ground_truth(root: Path) -> dict[str, Any]:
    """Best-effort load of the fixture ground-truth sidecar (empty if absent).

    Gold transcriptions are only available for the synthetic fixture; a real
    dataset has no ``ground_truth.json`` and yields an empty map (CER is then
    skipped per crop). Lazy-imported with the rest of the data layer.
    """
    from vignocr.data import load_ground_truth  # noqa: PLC0415

    try:
        return load_ground_truth(root)
    except FileNotFoundError:
        log.warning("ocr.eval.no_ground_truth", root=str(root))
        return {}


def _with_checkpoint(cfg: dict[str, Any], ckpt: Path) -> dict[str, Any]:
    """Return a copy of ``cfg`` with the baseline backend pointed at ``ckpt``."""
    import copy  # noqa: PLC0415

    cfg = copy.deepcopy(cfg)
    backends = cfg.setdefault("backends", {})
    baseline = backends.setdefault("baseline", {})
    baseline["rec_model_dir"] = str(ckpt)
    return cfg
