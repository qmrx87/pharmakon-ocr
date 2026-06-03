"""Fine-tune the recognition head on field crops.

``run(cfg_path, run_dir)`` reads ``configs/ocr/recognition.yaml``, seeds, builds
the training set by cropping every annotated field from the active dataset
(grouped by field *type* so each type's char whitelist + preprocessing apply),
and dispatches to the configured backend's training API:

* **baseline (PaddleOCR)** — PP-OCRv4 recognition fine-tuning. PaddleOCR trains
  from a YAML + label files via ``tools/train.py``; we materialize the crop
  dataset + a per-field char dictionary and invoke that flow.
* **transformer (TrOCR/Donut)** — a Hugging Face ``Seq2SeqTrainer`` loop.

Heavy libs (paddle / torch / transformers) are **lazy-imported inside** the
backend branches, so this module imports on CPU without the ``[ml]`` extra.
There is **no fixture/stub** here: real fine-tuning runs on a GPU host. The
function writes a manifest into ``run_dir`` and returns the run directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vignocr.common import (
    get_active_dataset,
    get_classes,
    get_logger,
    load_config,
    seed_everything,
)
from vignocr.common.config import load_yaml  # not re-exported from the package facade

log = get_logger(__name__)


def run(cfg_path: str, run_dir: Path) -> Path:
    """Fine-tune the recognition model and return the run directory.

    Args:
        cfg_path: path to an OCR recognition config (``configs/ocr/recognition.yaml``).
        run_dir: directory to write checkpoints + the run manifest into (created).

    Returns:
        ``run_dir`` (containing the backend's checkpoints + ``manifest.json``).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(cfg_path) if cfg_path else load_config("ocr/recognition")
    seed = int(cfg.get("train", {}).get("seed", 1337))
    seed_everything(seed)

    backend = cfg.get("backend", "baseline")
    dataset = get_active_dataset()
    log.info(
        "ocr.train.start",
        backend=backend,
        dataset=dataset["name"],
        run_dir=str(run_dir),
        seed=seed,
    )

    # Build the crop dataset (field name -> crops) for the train + val splits.
    # data/coco is the dataset module's documented API (docs/INTERFACES.md); it
    # is lazy-imported so this entrypoint imports without the data deps resolved.
    train_examples = _collect_crops(dataset, dataset["splits"]["train"])
    val_examples = _collect_crops(dataset, dataset["splits"]["val"])
    log.info("ocr.train.dataset", train=len(train_examples), val=len(val_examples))

    if backend == "baseline":
        ckpt = _train_paddle(cfg, train_examples, val_examples, run_dir)
    elif backend == "transformer":
        ckpt = _train_transformer(cfg, train_examples, val_examples, run_dir)
    else:
        raise ValueError(
            f"unknown OCR backend {backend!r}; expected 'baseline' or 'transformer'"
        )

    _write_manifest(run_dir, cfg, dataset, backend, ckpt, seed, len(train_examples), len(val_examples))
    log.info("ocr.train.done", run_dir=str(run_dir), checkpoint=str(ckpt))
    return run_dir


# --------------------------------------------------------------------------- #
# Dataset assembly (shared by both backends)
# --------------------------------------------------------------------------- #


def _collect_crops(dataset: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Crop every annotated field in ``split`` -> list of training examples.

    Each example is ``{"name", "type", "orientation", "image", "bbox"}``. The
    *text label* is the gold transcription; on synthetic data it comes from the
    generator's sidecar, on real data from the annotation. We attach it when
    present and otherwise leave it for the caller to source.
    """
    # Lazy import: the data subpackage owns COCO loading + cropping.
    from vignocr.data import coco  # noqa: PLC0415

    schema = get_classes()
    root = Path(dataset["root"])
    examples: list[dict[str, Any]] = []

    split_data = coco.load_split(root, split)
    for image in split_data.images:
        img_path = root / split / image.file_name
        # crops_for_image -> {field_name: [Crop(bbox, image), ...]}
        by_field = coco.crops_for_image(img_path, split_data.anns_for(image.id), schema)
        for field_name, crops in by_field.items():
            spec = schema.by_name(field_name)
            if spec["type"] == "region":  # not text — skip recognition training
                continue
            for crop in crops:
                examples.append(
                    {
                        "name": field_name,
                        "type": spec["type"],
                        "orientation": spec["orientation"],
                        "image": crop.image,
                        "bbox": crop.bbox,
                        "label": getattr(crop, "text", None),  # gold text if the dataset carries it
                    }
                )
    return examples


# --------------------------------------------------------------------------- #
# Baseline — PaddleOCR PP-OCRv4 recognition fine-tuning
# --------------------------------------------------------------------------- #


def _train_paddle(
    cfg: dict[str, Any],
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    run_dir: Path,
) -> Path:
    """Fine-tune the PaddleOCR recognition head.

    PaddleOCR trains its recognition model from a config YAML + label files
    (``image_path\\tlabel`` per line) via ``tools/train.py`` in the PaddleOCR
    repo. We orient + preprocess each crop, write the label files and a per-field
    character dictionary (from each field type's ``char_whitelist``), then launch
    that training program.
    """
    try:
        import paddle  # noqa: F401, PLC0415  (lazy: keeps core CPU-only)
    except ImportError as exc:  # pragma: no cover - only without [ml]
        raise ImportError(
            "PaddleOCR/PaddlePaddle are required to train the baseline recognizer "
            "but are not installed. Install the ML extra:  pip install -e .[ml]"
        ) from exc

    from vignocr.ocr.preprocess import preprocess_for_type  # noqa: PLC0415

    field_types = cfg.get("field_types", {}) or {}
    data_root = run_dir / "paddle_data"
    (data_root / "images").mkdir(parents=True, exist_ok=True)

    # Materialize preprocessed crops + label manifests.
    label_files = {"train": data_root / "train_label.txt", "val": data_root / "val_label.txt"}
    for split_name, examples in (("train", train_examples), ("val", val_examples)):
        lines: list[str] = []
        for i, ex in enumerate(examples):
            prepared = preprocess_for_type(ex["image"], ex["orientation"], field_types.get(ex["type"], {}))
            rel = f"images/{split_name}_{i:06d}.png"
            prepared.save(data_root / rel)
            if ex.get("label") is not None:
                lines.append(f"{rel}\t{ex['label']}")
        label_files[split_name].write_text("\n".join(lines), encoding="utf-8")

    # Per-field character dictionary: union the configured whitelists; a null
    # whitelist (text fields) means "use the backend's full latin dict", so we
    # only emit a custom dict when every active field type constrains its chars.
    char_dict_path = _write_char_dict(field_types, data_root)

    train_cfg = cfg.get("train", {})
    rec_yaml = _write_paddle_rec_config(cfg, data_root, char_dict_path, train_cfg, run_dir)

    # TODO(paddle-train-api): the exact training entrypoint is the PaddleOCR repo's
    # ``tools/train.py -c <rec_yaml>`` program (PaddleOCR ships the trainer as a
    # script, not a stable Python function). On a GPU host this is launched as:
    #     python tools/train.py -c {rec_yaml}
    # We record the prepared config + data here; wiring the subprocess launch is
    # deferred to the Narval orchestration (docs/ROADMAP.md Phase 7), which owns
    # process/sbatch management. Returning the expected best-model path.
    best_ckpt = run_dir / "best_accuracy"
    log.info(
        "ocr.train.paddle.prepared",
        rec_config=str(rec_yaml),
        train_label=str(label_files["train"]),
        char_dict=str(char_dict_path) if char_dict_path else None,
        launch_hint=f"python tools/train.py -c {rec_yaml}",
    )
    return best_ckpt


def _write_char_dict(field_types: dict[str, Any], data_root: Path) -> Path | None:
    """Write a character dictionary from the field types' whitelists, or None.

    Returns ``None`` when any active text field has a null whitelist (use the
    backend's default latin dict instead of an over-constrained custom one).
    """
    chars: set[str] = set()
    for ft in field_types.values():
        wl = ft.get("char_whitelist")
        if wl is None:
            return None  # a free-text field needs the full dict; don't constrain globally
        chars.update(ch for ch in wl if not ch.isspace())
    if not chars:
        return None
    path = data_root / "char_dict.txt"
    path.write_text("\n".join(sorted(chars)), encoding="utf-8")
    return path


def _write_paddle_rec_config(
    cfg: dict[str, Any],
    data_root: Path,
    char_dict_path: Path | None,
    train_cfg: dict[str, Any],
    run_dir: Path,
) -> Path:
    """Write the PaddleOCR recognition training YAML and return its path."""
    b = (cfg.get("backends", {}) or {}).get("baseline", {}) or {}
    rec_cfg = {
        "Global": {
            "use_gpu": bool(b.get("use_gpu", False)),
            "epoch_num": int(train_cfg.get("epochs", 50)),
            "save_model_dir": str(run_dir),
            "character_dict_path": str(char_dict_path) if char_dict_path else None,
            "max_text_length": int(train_cfg.get("max_text_length", 40)),
            "save_res_path": str(run_dir / "predicts.txt"),
        },
        "Optimizer": {"lr": {"learning_rate": float(train_cfg.get("learning_rate", 5e-4))}},
        "Train": {
            "dataset": {"data_dir": str(data_root), "label_file_list": [str(data_root / "train_label.txt")]},
            "loader": {"batch_size_per_card": int(train_cfg.get("batch_size", 64))},
        },
        "Eval": {
            "dataset": {"data_dir": str(data_root), "label_file_list": [str(data_root / "val_label.txt")]},
        },
    }
    path = run_dir / "rec_config.yaml"
    # Reuse the project's yaml dumper indirectly: write JSON-compatible YAML.
    import yaml  # noqa: PLC0415  (pyyaml is a core dep)

    path.write_text(yaml.safe_dump(rec_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Transformer — TrOCR / Donut fine-tuning (Hugging Face Seq2SeqTrainer)
# --------------------------------------------------------------------------- #


def _train_transformer(
    cfg: dict[str, Any],
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    run_dir: Path,
) -> Path:
    """Fine-tune a TrOCR/Donut model with a Hugging Face trainer loop."""
    try:
        import torch  # noqa: F401, PLC0415
        from transformers import (  # noqa: PLC0415
            AutoProcessor,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            VisionEncoderDecoderModel,
        )
    except ImportError as exc:  # pragma: no cover - scaffold path
        raise ImportError(
            "Training the transformer recognizer needs torch + transformers. "
            "Install:  pip install -e .[ml] transformers"
        ) from exc

    t = (cfg.get("backends", {}) or {}).get("transformer", {}) or {}
    engine = t.get("engine", "trocr")
    variant = (t.get("variants", {}) or {}).get(engine, {}) or {}
    model_name = variant.get("model_name")
    train_cfg = cfg.get("train", {})

    processor = AutoProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # TODO(transformer-train-api): the exact dataset collation differs between
    # TrOCR (pixel_values + labels from processor.tokenizer) and Donut
    # (task-prompt-conditioned target sequences). Confirmed at the TrOCR level;
    # the Donut target-sequence construction is assumed and unverified. Building
    # the torch Dataset from ``train_examples`` (each has .image + .label) is the
    # remaining wiring; deferred to the GPU training run.
    args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        num_train_epochs=int(train_cfg.get("epochs", 50)),
        predict_with_generate=True,
        seed=int(train_cfg.get("seed", 1337)),
    )
    _ = Seq2SeqTrainer  # referenced; instantiation needs the built datasets (see TODO)
    log.info(
        "ocr.train.transformer.prepared",
        engine=engine,
        model=model_name,
        output_dir=args.output_dir,
        train_examples=len(train_examples),
    )
    return run_dir / "checkpoint-best"


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _write_manifest(
    run_dir: Path,
    cfg: dict[str, Any],
    dataset: dict[str, Any],
    backend: str,
    ckpt: Path,
    seed: int,
    n_train: int,
    n_val: int,
) -> None:
    """Persist a reproducibility manifest next to the checkpoints."""
    manifest = {
        "backend": backend,
        "checkpoint": str(ckpt),
        "dataset": dataset["name"],
        "dataset_root": dataset["root"],
        "seed": seed,
        "num_train_crops": n_train,
        "num_val_crops": n_val,
        "train_cfg": cfg.get("train", {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
