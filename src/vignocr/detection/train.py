"""Train RF-DETR (medium) on the active vignette dataset.

Config-driven (``configs/detection/rfdetr_medium.yaml`` + ``classes.yaml`` for the
head width), seeded, TensorBoard-logged (offline), resumable from a checkpoint, and
reproducible (writes a config snapshot + git SHA into ``run_dir``). All heavy libs
(torch / rfdetr / tensorboard) are imported lazily inside :func:`run`.

Public API (per docs/INTERFACES.md):
    run(cfg_path, run_dir, resume=None) -> Path   # path to the best checkpoint
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vignocr.common import (
    get_logger,
    load_config,
    repo_root,
    seed_everything,
)
from vignocr.detection._resolve import resolve_class_schema, resolve_dataset

log = get_logger(__name__)

_ML_HINT = "Detection training needs the ML extra. Run: pip install -e .[ml]"

# Offline-init guidance: shown when no cached pretrained weights are available
# and no resume checkpoint was passed. Compute nodes on Narval have no internet,
# so a fresh init would hang trying to download from a Hub URL — we refuse to
# start in that state and tell the operator exactly how to fix it.
_OFFLINE_HINT = (
    "RF-DETR has no cached COCO-pretrained weights and no resume checkpoint. "
    "On offline compute nodes (Narval) a fresh init would hang trying to "
    "download from the Hub. Three ways out:\n"
    "  1. (RECOMMENDED) Cache the weights on the login node:\n"
    "         bash scripts/fetch_pretrained.sh\n"
    "     The trainer then picks them up automatically on every subsequent\n"
    "     compute-node submission.\n"
    "  2. (DEGRADED, but RUNS TODAY) Train from random init by exporting:\n"
    "         export VIGNOCR_ACCEPT_FROM_SCRATCH=1\n"
    "     Stage A/B will converge poorly without the DINOv2 backbone, but\n"
    "     the pipeline runs end-to-end so the rest of the DAG (eval, export,\n"
    "     OCR autolabel) can be exercised.\n"
    "  3. (LOGIN-NODE SMOKE TEST) Allow online init this one time:\n"
    "         export VIGNOCR_ALLOW_ONLINE_PRETRAIN=1\n"
    "     Useful only if you're on a node WITH internet."
)

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
    ds = resolve_dataset(cfg)
    num_classes, class_names = resolve_class_schema(cfg, ds)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "cfg_path": cfg_path,
        "resolved_config": cfg,
        "num_classes": num_classes,
        "class_names": class_names,
        "dataset": {"name": ds["name"], "root": ds["root"], "splits": ds.get("splits", {})},
        "checkpoint_dir": str(ckpt_dir),
        "scratch": os.environ.get("VIGNOCR_SCRATCH"),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    log.info("train.snapshot", run_dir=str(run_dir), git_sha=meta["git_sha"])


def _persist_label_map(ds: dict[str, Any], *dirs: Path) -> list[str] | None:
    """Persist the detector's class_id -> name map from the dataset's COCO file.

    WHY THIS EXISTS (correctness landmine):
        RF-DETR trains DIRECTLY on the raw COCO file (we hand it ``dataset_dir``),
        so its predicted ``class_id`` is an index into that file's ``categories``
        (ordered by id) — NOT into ``configs/classes.yaml``. For the reconciled
        real export the two do NOT agree (the COCO has ``drug-labels``/``text`` and
        names like ``lot``/``tarif_ref``; classes.yaml has 17 names in a different
        order). Inference that decodes ids via classes.yaml would therefore assign
        the WRONG field name to every box. We persist the authoritative ordered
        name list (with ``data.yaml: coco_aliases`` applied so downstream sees the
        schema names ``num_lot``/``tr``) next to the checkpoint; ``infer.Detector``
        loads it and can never silently scramble labels.

    Best-effort: provenance must never crash a training run.
    """
    try:
        root = Path(ds["root"])
        splits = ds.get("splits", {}) or {}
        coco_name = ds.get("coco_filename", "_annotations.coco.json")
        coco_path = root / splits.get("train", "train") / coco_name
        if not coco_path.is_file():
            log.warning("train.label_map_no_coco", path=str(coco_path))
            return None
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        cats = sorted(data.get("categories", []), key=lambda c: int(c["id"]))
        aliases = ds.get("coco_aliases", {}) or {}
        # index == RF-DETR contiguous class_id (COCO categories sorted by id).
        names = [aliases.get(c["name"], c["name"]) for c in cats]
        payload = {
            "source": str(coco_path),
            "dataset": ds.get("name"),
            "names": names,
            "raw_categories": [{"id": int(c["id"]), "name": c["name"]} for c in cats],
            "aliases_applied": aliases,
            "note": "names[i] is the class for RF-DETR class_id==i (COCO ids sorted asc).",
        }
        for d in dirs:
            (Path(d) / "class_names.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        log.info("train.label_map", n=len(names), names=names)
        return names
    except Exception as exc:  # noqa: BLE001 - provenance must never crash a run
        log.warning("train.label_map_failed", err=str(exc))
        return None


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


def _resolve_pretrained_weights(cfg: dict[str, Any]) -> str | None:
    """Return a local file path to RF-DETR's COCO-pretrained backbone, or None.

    Resolution order:
        1. cfg.model.pretrain_weights (explicit)               -> use as-is
        2. env VIGNOCR_PRETRAINED_RFDETR (explicit)            -> use as-is
        3. $VIGNOCR_PRETRAINED_DIR/rfdetr_medium_coco.pth      -> use if exists
        4. ~/.roboflow/models/rf-detr-medium.pth               -> use if exists
           (rfdetr 1.x caches its hosted models here; if the user ever called
           RFDETRMedium() on the login node, the file is already there)
        5. $ROBOFLOW_MODELS_DIR/rf-detr-medium.pth             -> use if exists
        6. None  (rfdetr will then try the Hub — only OK on login node)

    Cases (1)-(5) are CACHE HITS — the function returns a path to a real file.
    Case (6) is the only path that needs internet; we let the caller decide
    whether to allow it (compute-node jobs refuse, login-node smoke tests can
    opt in via VIGNOCR_ALLOW_ONLINE_PRETRAIN=1).
    """
    explicit = (cfg.get("model") or {}).get("pretrain_weights")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p)
        log.warning("train.pretrained.missing", explicit=str(p))

    env = os.environ.get("VIGNOCR_PRETRAINED_RFDETR")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        log.warning("train.pretrained.env_missing", env=str(p))

    cache_dir = os.environ.get("VIGNOCR_PRETRAINED_DIR")
    if cache_dir:
        for name in ("rfdetr_medium_coco.pth", "rf-detr-medium-coco.pth",
                     "rf-detr-medium.pth"):
            p = Path(cache_dir) / name
            if p.is_file():
                return str(p)

    # rfdetr 1.x's CANONICAL cache dir — `~/.roboflow/models/`. The library
    # writes here on first construction (RFDETRMedium auto-downloads to this
    # path). Override the prefix with $ROBOFLOW_MODELS_DIR.
    roboflow_dir = Path(os.environ.get("ROBOFLOW_MODELS_DIR",
                                       str(Path.home() / ".roboflow" / "models")))
    for name in ("rf-detr-medium.pth", "rf-detr-medium-coco.pth"):
        p = roboflow_dir / name
        if p.is_file():
            return str(p)

    return None


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

    ds = resolve_dataset(cfg)
    num_classes, class_names = resolve_class_schema(cfg, ds)
    dataset_dir = _coco_dataset_dir(ds)
    # Persist the authoritative id->name map (from the COCO RF-DETR actually
    # trains on) next to the checkpoint AND in the run dir, so inference decodes
    # class ids correctly instead of assuming classes.yaml ordering.
    _persist_label_map(ds, ckpt_dir, run_dir)
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

    # 3) Build the model. num_classes from cfg.model -> dataset.class_names ->
    #    classes.yaml (resolved above) so the head width is always config-bound.
    resolution = int(mcfg.get("resolution", 640))

    # OFFLINE INIT GATE: on a compute node, rfdetr's hub download has no network
    # to reach. Prefer (in order): resume ckpt > cached pretrained > online init
    # > random init (opt-in). The online and from-scratch paths are opt-in so we
    # never silently waste a GPU allocation on a doomed hub call OR train a
    # randomly-initialised model that won't converge meaningfully.
    init_source: str
    if resume:
        pretrain_weights = str(resume)
        init_source = "resume"
    else:
        cached = _resolve_pretrained_weights(cfg)
        if cached:
            pretrain_weights = cached
            init_source = "cache"
        elif os.environ.get("VIGNOCR_ALLOW_ONLINE_PRETRAIN"):
            pretrain_weights = None
            init_source = "online"
            log.warning("train.online_pretrain", note="hub download enabled — only safe on login node")
        elif os.environ.get("VIGNOCR_ACCEPT_FROM_SCRATCH"):
            # Explicit opt-in to train without COCO-pretrained backbone. Yields
            # a degraded model but unblocks the pipeline end-to-end so the rest
            # of the stages can be exercised. Logged as a WARNING so it shows
            # up in any post-mortem.
            pretrain_weights = None
            init_source = "from_scratch_opt_in"
            log.warning(
                "train.from_scratch",
                note="VIGNOCR_ACCEPT_FROM_SCRATCH=1 set; training from random init. "
                     "Expect degraded mAP. Cache pretrained weights with "
                     "scripts/fetch_pretrained.sh for production runs.",
            )
        else:
            raise FileNotFoundError(_OFFLINE_HINT)

    log.info(
        "train.start",
        model=mcfg.get("name", "rfdetr_medium"),
        num_classes=num_classes,
        resolution=resolution,
        epochs=tcfg.get("epochs"),
        batch_size=tcfg.get("batch_size"),
        dataset=str(dataset_dir),
        resume=str(resume) if resume else None,
        init_source=init_source,
        pretrain_weights=pretrain_weights,
    )
    model = RFDETRMedium(
        num_classes=num_classes,
        resolution=resolution,
        pretrain_weights=pretrain_weights,
    )

    early = tcfg.get("early_stop", {})
    # rfdetr's REAL train() surface (verified against rfdetr 1.x docs,
    # rfdetr.roboflow.com/.../training-parameters):
    #   dataset_dir, output_dir, epochs, batch_size, grad_accum_steps, resume,
    #   lr, lr_encoder (NOT lr_backbone), weight_decay, seed, tensorboard,
    #   early_stopping{,_patience,_min_delta,_use_ema}, ...
    # Two names from our config DON'T exist on rfdetr's API and previously got
    # passed through verbatim:
    #   * lr_backbone  -> rfdetr calls the backbone/encoder LR `lr_encoder`.
    #   * clip_max_norm / num_workers -> not part of rfdetr's train() surface.
    # We MAP lr_backbone onto lr_encoder and DROP the two unknowns so we don't
    # gamble on the signature-filter below catching them (it only fires when
    # train() uses explicit params; if rfdetr uses **kwargs + a strict pydantic
    # TrainConfig, an unknown kwarg raises a ValidationError our TypeError guard
    # would NOT catch). clip_max_norm/num_workers stay readable in the config for
    # provenance; they're simply not forwarded.
    # VIGNOCR_MAX_EPOCHS caps epochs for a fast smoke test (e.g. =1) WITHOUT
    # editing the config — exercises the real train path (kwargs, label-map
    # persistence, checkpointing) in a few minutes. Unset => use the config value.
    _epochs = int(os.environ.get("VIGNOCR_MAX_EPOCHS") or tcfg.get("epochs", 50))
    train_kwargs: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "epochs": _epochs,
        "batch_size": int(tcfg.get("batch_size", 4)),
        "grad_accum_steps": int(tcfg.get("grad_accum_steps", 1)),
        "lr": float(tcfg.get("lr", 1e-4)),
        # config key stays `lr_backbone` (human-meaningful); rfdetr kwarg is lr_encoder.
        "lr_encoder": float(tcfg.get("lr_backbone", tcfg.get("lr_encoder", 1.5e-4))),
        "weight_decay": float(tcfg.get("weight_decay", 1e-4)),
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

    # SIGNATURE-BASED FILTER: different rfdetr wheels have slightly different
    # train() signatures (1.1 vs 1.2 vs nightly). Rather than blow up on the
    # first unknown kwarg, inspect the installed signature and drop kwargs the
    # binding does not accept — logging what we dropped so the run is auditable.
    try:
        sig = inspect.signature(model.train)
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not accepts_var_kw:
            allowed = set(sig.parameters.keys())
            dropped = {k: v for k, v in train_kwargs.items() if k not in allowed}
            if dropped:
                log.warning(
                    "train.kwargs.dropped_by_signature",
                    dropped=sorted(dropped.keys()),
                    rfdetr_signature=sorted(allowed),
                )
                train_kwargs = {k: v for k, v in train_kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        # If signature inspection itself fails (C-extension, partial), fall
        # through and rely on the TypeError handler below.
        pass

    try:
        model.train(**train_kwargs)
    except TypeError as exc:
        # Surface an actionable message if the installed rfdetr signature differs
        # in a way that signature-filtering didn't catch (e.g. unexpected positional).
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
