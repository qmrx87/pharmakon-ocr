"""Fine-tune PaddleOCR's text-recognition model on the corrected vignette crops.

Stage C closes the bootstrap loop:

    autolabel  -> (human correction in Roboflow / spreadsheet)  -> finetune

It reads ``configs/ocr/finetune.yaml`` + the corrected dataset directory
(``paddle.txt`` per split, single ``char_dict.txt`` at the root) and produces a
checkpoint that ``configs/ocr/recognition.yaml: backend_checkpoint`` points at.

PaddleOCR's training surface is version-dependent, so the heavy hand-off is
delegated to PaddleOCR's own ``tools/train.py`` via a generated PPOCR YAML —
the safest path across releases. The Python entrypoint validates inputs, writes
the PPOCR YAML + a run snapshot, and shells out to the trainer.

Public API (per docs/INTERFACES.md):

    run(cfg_path, run_dir, dataset_dir, base_ckpt=None) -> Path   # best checkpoint
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vignocr.common import get_logger, load_config, repo_root, seed_everything

log = get_logger(__name__)

_ML_HINT = "OCR fine-tuning needs the ML extra (paddleocr/paddlepaddle). Run: pip install -e .[ml]"


# --------------------------------------------------------------------------- #
# Reproducibility metadata
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(repo_root()),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        pass
    return "unknown"


# --------------------------------------------------------------------------- #
# Dataset readiness checks (cheap, pure-Python; reject before allocating GPU)
# --------------------------------------------------------------------------- #


def _audit_dataset(dataset_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Verify the corrected dataset looks usable; warn if mostly auto-labels remain.

    The auto-labeler emits placeholder strings prefixed ``AUTO_`` for the stub
    backend and the PaddleOCR backend often produces them on the first pass when
    the recognizer is under-confident. We warn (not fail) if too many remain in
    the corrected dataset — a sign the human review is still incomplete.
    """
    ds = cfg["dataset"]
    splits = list(ds.get("splits", {}).values()) or ["train", "valid", "test"]
    char_dict_path = dataset_dir / ds.get("char_dict", "char_dict.txt")
    if not char_dict_path.exists():
        raise FileNotFoundError(
            f"char_dict.txt missing at {char_dict_path}. Run stage 04 autolabel first."
        )

    audit: dict[str, Any] = {"splits": {}, "warnings": []}
    min_ratio = float(cfg.get("dataset", {}).get("min_corrected_ratio", 0.5))
    for split in splits:
        paddle_txt = dataset_dir / split / ds.get("paddle_label", "paddle.txt")
        if not paddle_txt.exists():
            raise FileNotFoundError(
                f"{paddle_txt} missing. Run stage 04 autolabel for split={split!r}."
            )
        n = 0
        n_auto = 0
        with paddle_txt.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                n += 1
                # Tab-separated "<path>\t<text>". Trailing AUTO_<FIELD> markers
                # are how the autolabel stub (and uncorrected paddle outputs)
                # surface "needs review".
                if "\tAUTO_" in line:
                    n_auto += 1
        corrected_ratio = 1.0 - (n_auto / n) if n else 0.0
        audit["splits"][split] = {"rows": n, "auto_remaining": n_auto, "corrected_ratio": corrected_ratio}
        if corrected_ratio < min_ratio:
            audit["warnings"].append(
                f"split={split!r} only {corrected_ratio:.0%} corrected "
                f"({n_auto}/{n} still AUTO_*) — review may be incomplete."
            )
    return audit


# --------------------------------------------------------------------------- #
# PPOCR config generation
# --------------------------------------------------------------------------- #


def _emit_ppocr_yaml(
    cfg: dict[str, Any],
    dataset_dir: Path,
    out_yaml: Path,
    run_dir: Path,
    base_ckpt: Path | None,
) -> Path:
    """Write a PPOCR-compatible training YAML mapped from our finetune.yaml.

    Kept narrow on purpose: only the keys PaddleOCR's ``tools/train.py`` needs.
    Adjust here if you target a different PP-OCR version.
    """
    ds = cfg["dataset"]
    model = cfg.get("model", {})
    train = cfg.get("train", {})
    aug = cfg.get("augment", {})

    char_dict = dataset_dir / ds.get("char_dict", "char_dict.txt")
    train_split = ds["splits"].get("train", "train")
    val_split = ds["splits"].get("val", "valid")
    train_label = dataset_dir / train_split / ds.get("paddle_label", "paddle.txt")
    val_label = dataset_dir / val_split / ds.get("paddle_label", "paddle.txt")

    ppocr = {
        "Global": {
            "use_gpu": True,
            "epoch_num": int(train.get("epochs", 80)),
            "log_smooth_window": 20,
            "print_batch_step": int(cfg.get("eval", {}).get("log_every", 50)),
            "save_model_dir": str(run_dir / "ppocr_ckpts"),
            "save_epoch_step": int(train.get("save_epoch_step", 2)),
            "eval_batch_step": [0, int(train.get("eval_epoch_step", 1)) * 2000],
            "cal_metric_during_train": True,
            "pretrained_model": str(base_ckpt) if base_ckpt else model.get("pretrained"),
            "checkpoints": None,
            "save_inference_dir": str(run_dir / "ppocr_inference"),
            "use_visualdl": False,
            "infer_img": "",
            "character_dict_path": str(char_dict),
            "character_type": "ch",
            "max_text_length": int(model.get("max_text_length", 80)),
            "infer_mode": False,
            "use_space_char": True,
            "save_res_path": str(run_dir / "ppocr_pred.txt"),
        },
        "Optimizer": {
            "name": train.get("optimizer", "Adam"),
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": {
                "name": train.get("lr_scheduler", "Cosine"),
                "learning_rate": float(train.get("lr", 0.001)),
                "warmup_epoch": int(train.get("warmup_epoch", 2)),
            },
            "regularizer": {"name": "L2", "factor": float(train.get("weight_decay", 1.0e-5))},
        },
        "Architecture": {
            "model_type": "rec",
            "algorithm": model.get("algorithm", "CRNN"),
            "Transform": None,
            "Backbone": {"name": model.get("backbone", "MobileNetV3"), "scale": 0.5},
            "Neck": {"name": "SequenceEncoder", "encoder_type": "rnn", "hidden_size": 96},
            "Head": {"name": "CTCHead", "fc_decay": 0.0},
        },
        "Loss": {"name": "CTCLoss"},
        "PostProcess": {"name": "CTCLabelDecode"},
        "Metric": {"name": "RecMetric", "main_indicator": "acc"},
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": str(dataset_dir / train_split),
                "label_file_list": [str(train_label)],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"CTCLabelEncode": None},
                    {
                        "RecAug": {
                            "tia_prob": 0.0,
                            "crop_prob": 0.0,
                            "reverse_prob": 0.0,
                            "noise_prob": 0.0,
                            "jitter_prob": float(aug.get("brightness", 0.2)),
                            "blur_prob": float(aug.get("gaussian_blur_p", 0.1)),
                        }
                    },
                    {"RecResizeImg": {"image_shape": model.get("image_shape", [3, 48, 320])}},
                    {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                ],
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": int(train.get("batch_size", 128)),
                "drop_last": True,
                "num_workers": int(train.get("num_workers", 4)),
            },
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": str(dataset_dir / val_split),
                "label_file_list": [str(val_label)],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"CTCLabelEncode": None},
                    {"RecResizeImg": {"image_shape": model.get("image_shape", [3, 48, 320])}},
                    {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                ],
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": int(train.get("batch_size", 128)),
                "num_workers": int(train.get("num_workers", 4)),
            },
        },
    }

    import yaml as _yaml

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(_yaml.safe_dump(ppocr, sort_keys=False), encoding="utf-8")
    return out_yaml


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def run(
    cfg_path: str | Path,
    run_dir: Path | str,
    dataset_dir: Path | str | None = None,
    base_ckpt: Path | str | None = None,
) -> Path:
    """Fine-tune PaddleOCR recognition. Returns the best checkpoint path.

    Args:
        cfg_path: ``"ocr/finetune"`` or an absolute YAML path.
        run_dir: directory for logs / PPOCR YAML / snapshot / checkpoints.
        dataset_dir: corrected dataset root (the stage-04 output, post-review).
            Defaults to ``cfg.dataset.root``.
        base_ckpt: optional pretrained PaddleOCR rec checkpoint to start from.

    Returns:
        Path to the best checkpoint produced by PaddleOCR (``best_accuracy.pdparams``
        under ``<run_dir>/ppocr_ckpts``).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(cfg_path))
    seed = int(cfg.get("train", {}).get("seed", 1337))
    seed_everything(seed)

    dataset_dir = Path(dataset_dir or cfg["dataset"]["root"])
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root() / dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"corrected dataset not found: {dataset_dir}. "
            "Run stage 04 (autolabel) and review the labels first."
        )

    # 1) Reproducibility snapshot (config + git SHA + dataset audit).
    audit = _audit_dataset(dataset_dir, cfg)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "cfg_path": str(cfg_path),
        "resolved_config": cfg,
        "dataset_dir": str(dataset_dir),
        "base_ckpt": str(base_ckpt) if base_ckpt else None,
        "audit": audit,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    snapshot_src = (
        (repo_root() / "configs" / f"{cfg_path}.yaml")
        if not str(cfg_path).endswith((".yaml", ".yml")) and not Path(str(cfg_path)).is_absolute()
        else Path(str(cfg_path))
    )
    if snapshot_src.exists():
        shutil.copy2(snapshot_src, run_dir / "finetune.snapshot.yaml")
    for warn in audit.get("warnings", []):
        log.warning("finetune.audit", msg=warn)

    # 2) PPOCR YAML + invoke the trainer.
    ppocr_yaml = _emit_ppocr_yaml(
        cfg, dataset_dir, run_dir / "ppocr_rec.yaml", run_dir, Path(base_ckpt) if base_ckpt else None
    )
    log.info("finetune.start", ppocr_yaml=str(ppocr_yaml), run_dir=str(run_dir))

    try:
        import paddleocr  # noqa: F401 — surfaces a clean error before exec
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_ML_HINT) from exc

    # PaddleOCR ships its training entry as ``ppocr/tools/train.py`` (sometimes
    # ``tools/train.py``). The exact path depends on the installed package
    # layout — we resolve it at runtime to stay version-compatible.
    train_script = _locate_paddle_train_script()
    cmd = [sys.executable, str(train_script), "-c", str(ppocr_yaml)]
    log.info("finetune.exec", cmd=" ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(run_dir), env=os.environ.copy())
    if completed.returncode != 0:
        raise RuntimeError(
            f"PaddleOCR fine-tune failed (rc={completed.returncode}); see logs under {run_dir}"
        )

    best = run_dir / "ppocr_ckpts" / "best_accuracy.pdparams"
    if not best.exists():  # PaddleOCR sometimes names it best_accuracy.states
        for candidate in (run_dir / "ppocr_ckpts").glob("best_accuracy.*"):
            best = candidate
            break
    log.info("finetune.done", best_ckpt=str(best))
    return best


def _locate_paddle_train_script() -> Path:
    """Find PaddleOCR's ``tools/train.py`` regardless of the install layout."""
    try:
        import paddleocr as _po
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_ML_HINT) from exc
    pkg_root = Path(getattr(_po, "__file__", "")).parent
    candidates = [
        pkg_root / "tools" / "train.py",
        pkg_root / "ppocr" / "tools" / "train.py",
        pkg_root.parent / "tools" / "train.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    # TODO(paddleocr-api): if the installed paddleocr does not bundle tools/,
    # vendor PaddleOCR's tools/train.py into scripts/ and point this here.
    raise FileNotFoundError(
        "PaddleOCR tools/train.py not found in the installed package. "
        "Vendor PaddleOCR's tools/train.py into scripts/ and update _locate_paddle_train_script()."
    )
