"""Train RF-DETR (medium) on the active vignette dataset.

Config-driven (``configs/detection/rfdetr_medium.yaml`` + ``classes.yaml`` for the
head width), seeded, TensorBoard-logged (offline), resumable from a checkpoint, and
reproducible (writes a config snapshot + git SHA into ``run_dir``). All heavy libs
(torch / rfdetr / tensorboard) are imported lazily inside :func:`run`.

Public API (per docs/INTERFACES.md):
    run(cfg_path, run_dir, resume=None) -> Path   # path to the best checkpoint
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vignocr.common import (
    get_active_dataset,
    get_classes,
    get_logger,
    load_config,
    repo_root,
    seed_everything,
)

log = get_logger(__name__)

_ML_HINT = "Detection training needs the ML extra. Run: pip install -e .[ml]"

# Augmentations that could flip green<->red band semantics. They live in the
# config's `augmentation.forbidden` block and MUST stay off — asserted below.
_FORBIDDEN_AUG_KEYS = (
    "hue_shift",
    "channel_shuffle",
    "rgb_shift",
    "to_grayscale",
    "color_jitter_strong",
)


# --------------------------------------------------------------------------- #
# Reproducibility helpers (pure-Python; no ML import)
# --------------------------------------------------------------------------- #
def _git_sha() -> str:
    """Best-effort current git SHA (``unknown`` if not a repo / git absent)."""
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
    except Exception:  # noqa: BLE001 - reproducibility metadata must never crash a run
        pass
    return "unknown"


def _resolve_checkpoint_dir(cfg: dict[str, Any], run_dir: Path) -> Path:
    """Checkpoints go to SCRATCH on HPC, else under the repo.

    Resolution order:
        1. env ``VIGNOCR_SCRATCH`` -> ``$VIGNOCR_SCRATCH/<cfg dir>`` (Narval)
        2. config ``train.checkpoint.dir`` (absolute used as-is, else repo-relative)
    The ``run_dir`` name is appended so concurrent runs never collide.
    """
    cfg_dir = (
        cfg.get("train", {}).get("checkpoint", {}).get("dir", "scratch/detection/rfdetr_medium")
    )
    scratch = os.environ.get("VIGNOCR_SCRATCH")
    base = (
        Path(scratch) / cfg_dir
        if scratch
        else (Path(cfg_dir) if Path(cfg_dir).is_absolute() else repo_root() / cfg_dir)
    )
    ckpt_dir = base / run_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _assert_band_color_preserving(cfg: dict[str, Any]) -> None:
    """Hard safety gate: no augmentation may flip green<->red band semantics.

    The ``color_band`` class encodes CHIFA reimbursability via HUE alone, so any
    hue/channel/strong-colour op is forbidden. We refuse to start training if one
    is enabled (defends against a careless config edit reaching a GPU run).
    """
    forbidden = cfg.get("augmentation", {}).get("forbidden", {})
    enabled = [k for k in _FORBIDDEN_AUG_KEYS if forbidden.get(k)]
    # Also catch a stray non-zero saturation in the photometric block.
    if float(cfg.get("augmentation", {}).get("photometric", {}).get("saturation", 0.0)) != 0.0:
        enabled.append("photometric.saturation")
    if enabled:
        raise ValueError(
            "Band-color-preserving constraint violated: augmentation(s) "
            f"{enabled} are enabled but could flip green<->red reimbursability "
            "semantics. Keep them disabled (see rfdetr_medium.yaml comments)."
        )


def _snapshot_run(cfg_path: str, cfg: dict[str, Any], run_dir: Path, ckpt_dir: Path) -> None:
    """Write a reproducibility snapshot (resolved config + git SHA + env) to run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    # Copy the raw config file verbatim.
    src = (
        (repo_root() / "configs" / f"{cfg_path}.yaml")
        if not Path(cfg_path).is_absolute()
        else Path(cfg_path)
    )
    if src.exists():
        shutil.copy2(src, run_dir / "config_snapshot.yaml")
    schema = get_classes()
    ds = get_active_dataset()
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "cfg_path": cfg_path,
        "resolved_config": cfg,
        "num_classes": schema.num_classes,
        "class_names": schema.names,
        "dataset": {"name": ds["name"], "root": ds["root"], "splits": ds.get("splits", {})},
        "checkpoint_dir": str(ckpt_dir),
        "scratch": os.environ.get("VIGNOCR_SCRATCH"),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    log.info("train.snapshot", run_dir=str(run_dir), git_sha=meta["git_sha"])


def _coco_dataset_dir(ds: dict[str, Any]) -> Path:
    """RF-DETR consumes a Roboflow COCO directory (train/ valid/ test/).

    The active dataset's ``root`` already follows that layout (one split dir per
    ``splits`` value, each with the ``coco_filename``). RF-DETR's loader expects
    exactly this, so we hand it the root.
    """
    root = Path(ds["root"])
    if not root.exists():
        raise FileNotFoundError(
            f"dataset root not found: {root}. Generate the synthetic fixture "
            "(vignocr.data.synthetic) or point VIGNOCR_DATA_ACTIVE at real data."
        )
    return root


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def run(cfg_path: str, run_dir: Path | str, resume: Path | str | None = None) -> Path:
    """Train RF-DETR medium and return the path to the best checkpoint.

    Args:
        cfg_path: detector config name (e.g. ``"detection/rfdetr_medium"``) or an
            absolute path to a YAML file.
        run_dir: directory for logs / TensorBoard / the reproducibility snapshot.
        resume: optional checkpoint to resume from (else fresh from pretrained).

    Returns:
        Path to the best checkpoint (on scratch when ``VIGNOCR_SCRATCH`` is set).
    """
    run_dir = Path(run_dir)
    cfg = load_config(cfg_path)

    # 1) Safety + reproducibility BEFORE importing/allocating anything heavy.
    _assert_band_color_preserving(cfg)
    seed = int(cfg.get("train", {}).get("seed", 1337))
    seed_everything(seed)
    ckpt_dir = _resolve_checkpoint_dir(cfg, run_dir)
    _snapshot_run(cfg_path, cfg, run_dir, ckpt_dir)

    schema = get_classes()
    ds = get_active_dataset()
    dataset_dir = _coco_dataset_dir(ds)
    tcfg = cfg.get("train", {})
    mcfg = cfg.get("model", {})

    # 2) Lazy ML imports (kept here so the module imports on CPU without [ml]).
    #    rfdetr requires torch transitively, so this import fails fast (with the
    #    [ml] hint) when the extra is absent.
    try:
        from rfdetr import RFDETRMedium
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_ML_HINT) from exc

    tb_writer = _maybe_tensorboard(run_dir)

    # 3) Build the model. num_classes ALWAYS from classes.yaml (never the config).
    num_classes = schema.num_classes
    resolution = int(mcfg.get("resolution", 640))
    log.info(
        "train.start",
        model=mcfg.get("name", "rfdetr_medium"),
        num_classes=num_classes,
        resolution=resolution,
        epochs=tcfg.get("epochs"),
        batch_size=tcfg.get("batch_size"),
        dataset=str(dataset_dir),
        resume=str(resume) if resume else None,
    )
    model = RFDETRMedium(
        num_classes=num_classes,
        resolution=resolution,
        pretrain_weights=str(resume) if resume else None,
    )

    early = tcfg.get("early_stop", {})
    # TODO(rfdetr-api): rfdetr==1.1.0 exposes Model.train(dataset_dir=..., epochs=...,
    # batch_size=..., grad_accum_steps=..., lr=..., lr_encoder/lr_backbone=...,
    # output_dir=..., resume=..., early_stopping=..., tensorboard=...). Argument
    # names below follow that documented surface; if the installed build differs,
    # adapt the kwargs here (the rest of this function — seeding, snapshot, scratch
    # checkpoints, parity with classes.yaml — is API-independent).
    train_kwargs: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "epochs": int(tcfg.get("epochs", 50)),
        "batch_size": int(tcfg.get("batch_size", 4)),
        "grad_accum_steps": int(tcfg.get("grad_accum_steps", 1)),
        "lr": float(tcfg.get("lr", 1e-4)),
        "lr_backbone": float(tcfg.get("lr_backbone", 1e-5)),
        "weight_decay": float(tcfg.get("weight_decay", 1e-4)),
        "clip_max_norm": float(tcfg.get("clip_max_norm", 0.1)),
        "num_workers": int(tcfg.get("num_workers", 2)),
        "output_dir": str(ckpt_dir),
        "resume": str(resume) if resume else None,
        "early_stopping": bool(early.get("enabled", False)),
        "early_stopping_patience": int(early.get("patience", 10)),
        "early_stopping_min_delta": float(early.get("min_delta", 0.001)),
        "tensorboard": True,  # rfdetr writes TB events into output_dir (offline)
        "seed": seed,
    }
    # Drop None-valued kwargs so we don't override rfdetr defaults with null.
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

    try:
        model.train(**train_kwargs)
    except TypeError as exc:
        # Surface an actionable message if the installed rfdetr signature differs.
        raise TypeError(
            "rfdetr RFDETRMedium.train() rejected our kwargs "
            f"({sorted(train_kwargs)}). Align them with the installed rfdetr "
            f"version's train() signature. Original error: {exc}"
        ) from exc
    finally:
        if tb_writer is not None:
            tb_writer.close()

    best = _resolve_best_checkpoint(ckpt_dir, tcfg)
    log.info("train.done", best_checkpoint=str(best))
    return best


def _maybe_tensorboard(run_dir: Path) -> Any | None:
    """Return an offline TensorBoard SummaryWriter, or None if unavailable.

    rfdetr writes its own TB events into ``output_dir`` (scratch); this writer
    captures the run-level reproducibility scalars next to the snapshot. Offline
    by construction — SummaryWriter only writes local event files.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # noqa: BLE001 - TB is optional; never block training
        return None
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/git_sha", _git_sha())
    return writer


def _resolve_best_checkpoint(ckpt_dir: Path, tcfg: dict[str, Any]) -> Path:
    """Locate the best checkpoint rfdetr wrote into ``ckpt_dir``.

    Tries common rfdetr/DETR names first, then the newest ``*.pth``/``*.pt``.
    """
    for candidate in (
        "checkpoint_best_total.pth",
        "checkpoint_best.pth",
        "checkpoint_best_ema.pth",
        "best.pth",
    ):
        p = ckpt_dir / candidate
        if p.exists():
            return p
    ckpts = sorted(
        [*ckpt_dir.glob("*.pth"), *ckpt_dir.glob("*.pt")],
        key=lambda p: p.stat().st_mtime,
    )
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(
        f"no checkpoint found in {ckpt_dir} after training; expected a "
        "checkpoint_best*.pth (check rfdetr's output_dir / save settings)."
    )
