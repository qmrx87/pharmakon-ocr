"""Export a trained RF-DETR (medium) checkpoint to ONNX, with a parity check.

``to_onnx`` exports the checkpoint and then verifies that ONNX Runtime reproduces
the torch outputs within tolerance on a real fixture image — raising on mismatch
so a silently-broken export can never reach serving. All heavy libs (torch /
rfdetr / onnx / onnxruntime) are imported lazily inside the function.

Public API (per docs/INTERFACES.md):
    to_onnx(ckpt, out) -> Path

------------------------------------------------------------------------------
TensorRT path (documented; not built here — belongs on the GPU/serving host):

    1. Export ONNX with a FIXED input shape (export.dynamic_batch: false in
       rfdetr_medium.yaml) — TensorRT optimizes a static graph best.
    2. Build the engine on the *target* GPU (engines are not portable across
       GPU arch / TRT version):
           trtexec --onnx=model.onnx \
                   --saveEngine=model_fp16.plan \
                   --fp16 \
                   --memPoolSize=workspace:4096
       (or the Python `tensorrt` Builder API for INT8 calibration).
    3. Validate the .plan with the SAME parity check used here (torch vs TRT on
       the fixture) before promoting it. Keep FP16/INT8 atol looser than the
       FP32 ONNX tolerance, and re-run detection eval to confirm mAP / business-
       critical localization recall did not regress.
    4. Serve via onnxruntime's TensorRT EP or a native TRT runtime; the
       Detector(.onnx) backend already prefers CUDA providers when present.
------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from vignocr.common import (
    get_active_dataset,
    get_classes,
    get_logger,
    load_config,
)

if TYPE_CHECKING:
    from PIL.Image import Image

log = get_logger(__name__)

_ML_HINT = (
    "ONNX export needs the ML extra (torch/rfdetr/onnx/onnxruntime). Run: pip install -e .[ml]"
)


def to_onnx(
    ckpt: Path | str, out: Path | str, cfg_path: str = "detection/rfdetr_medium"
) -> Path:
    """Export ``ckpt`` to ONNX at ``out`` and verify torch<->onnxruntime parity.

    Args:
        ckpt: RF-DETR ``.pth``/``.pt`` checkpoint to export.
        out: destination ``.onnx`` path (parent dirs are created).
        cfg_path: the detection config the checkpoint was TRAINED with (Stage A
            checkpoints need ``detection/rfdetr_vignette``). Determines the
            model variant, resolution, and class schema — previously hardcoded
            to the Stage B config, which mis-sized a Stage A 3-class head as 17.

    Returns:
        The path to the written ``.onnx`` file.

    Raises:
        ImportError: if the ``[ml]`` extra is missing.
        RuntimeError: if the ONNX Runtime outputs diverge from torch beyond the
            configured tolerance (``export.parity_atol`` / ``parity_rtol``).
    """
    ckpt = Path(ckpt)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    cfg = load_config(cfg_path)
    mcfg = cfg.get("model", {})
    ecfg = cfg.get("export", {})
    resolution = int(mcfg.get("resolution", 640))
    opset = int(ecfg.get("opset", 17))
    atol = float(ecfg.get("parity_atol", 1e-3))
    rtol = float(ecfg.get("parity_rtol", 1e-3))

    try:
        # rfdetr requires torch transitively, so this single import fails fast
        # (with the [ml] hint) when the extra is absent. The helpers below import
        # torch directly where they use it.
        from vignocr.detection._resolve import resolve_model_class

        model_cls = resolve_model_class(cfg)
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_ML_HINT) from exc

    # Head width: the class_names.json the trainer persisted BESIDE the ckpt is
    # authoritative (it reflects the COCO the model actually trained on). Fall
    # back to the config-resolved schema, then classes.yaml.
    num_classes = _ckpt_num_classes(ckpt)
    if num_classes is None:
        from vignocr.detection._resolve import resolve_class_schema, resolve_dataset

        try:
            num_classes, _ = resolve_class_schema(cfg, resolve_dataset(cfg))
        except Exception:  # noqa: BLE001 - config best-effort
            num_classes = get_classes().num_classes
    log.info("export.start", ckpt=str(ckpt), out=str(out), num_classes=num_classes, opset=opset)
    model = model_cls(
        pretrain_weights=str(ckpt),
        num_classes=num_classes,
        resolution=resolution,
    )

    # 1) Export via rfdetr's own exporter when available (it knows the graph /
    #    postprocessing). Otherwise fall back to a generic torch.onnx.export.
    exported = _try_rfdetr_export(model, out, opset, resolution, ecfg)
    if not exported:
        _generic_torch_export(model, out, opset, resolution, ecfg)
    if not out.exists():
        raise RuntimeError(f"ONNX export did not produce a file at {out}")
    log.info("export.written", path=str(out), size_bytes=out.stat().st_size)

    # 2) Parity check on a real fixture image (raises on mismatch).
    _parity_check(model, out, resolution, atol, rtol)
    log.info("export.parity_ok", atol=atol, rtol=rtol)
    return out


def _ckpt_num_classes(ckpt: Path) -> int | None:
    """Head width from the ``class_names.json`` persisted beside the checkpoint.

    The trainer writes the authoritative id->name list next to every checkpoint
    (see ``train._persist_label_map``). Returns ``None`` when absent/unreadable.
    """
    import json

    for d in (ckpt.parent, ckpt.parent.parent):
        f = d / "class_names.json"
        if f.is_file():
            try:
                names = json.loads(f.read_text(encoding="utf-8")).get("names")
                if isinstance(names, list) and names:
                    return len(names)
            except (OSError, ValueError):
                continue
    return None


# --------------------------------------------------------------------------- #
# Export backends
# --------------------------------------------------------------------------- #
def _try_rfdetr_export(
    model: Any, out: Path, opset: int, resolution: int, ecfg: dict[str, Any]
) -> bool:
    """Use rfdetr's built-in ONNX exporter if present. Returns True on success.

    TODO(rfdetr-api): rfdetr==1.1.0 ships an `export()` method (it writes an
    ONNX file under an output dir, sometimes named `inference_model.onnx`). The
    exact signature varies; we try it and move the artefact to `out`. If your
    build lacks `export()`, the generic torch exporter below is used instead.
    """
    export_fn = getattr(model, "export", None)
    if not callable(export_fn):
        return False
    try:
        import inspect

        out_dir = out.parent / f"_rfdetr_export_{out.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        params = inspect.signature(export_fn).parameters
        if "output_dir" in params:
            kwargs["output_dir"] = str(out_dir)
        if "opset_version" in params:
            kwargs["opset_version"] = opset
        elif "opset" in params:
            kwargs["opset"] = opset
        if "simplify" in params:
            kwargs["simplify"] = True
        if "dynamic" in params:
            kwargs["dynamic"] = bool(ecfg.get("dynamic_batch", False))
        export_fn(**kwargs)

        produced = sorted(out_dir.rglob("*.onnx"), key=lambda p: p.stat().st_mtime)
        if not produced:
            log.warning(
                "export.rfdetr_no_onnx", note="model.export() produced no .onnx; using fallback"
            )
            return False
        produced[-1].replace(out)
        return True
    except Exception as exc:  # noqa: BLE001 - fall back to generic exporter
        log.warning(
            "export.rfdetr_failed", error=str(exc), note="falling back to torch.onnx.export"
        )
        return False


def _generic_torch_export(
    model: Any, out: Path, opset: int, resolution: int, ecfg: dict[str, Any]
) -> None:
    """Generic ``torch.onnx.export`` of the underlying nn.Module on a dummy input."""
    import torch

    module = _inner_module(model)
    module.eval()
    dummy = torch.randn(1, 3, resolution, resolution)
    dynamic_axes = None
    if bool(ecfg.get("dynamic_batch", False)):
        dynamic_axes = {"images": {0: "batch"}, "boxes": {0: "batch"}, "logits": {0: "batch"}}
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy,
            str(out),
            input_names=["images"],
            output_names=["boxes", "logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
    # Structural validation of the produced graph.
    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(out)))
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_ML_HINT) from exc


# --------------------------------------------------------------------------- #
# Parity check
# --------------------------------------------------------------------------- #
def _parity_check(model: Any, out: Path, resolution: int, atol: float, rtol: float) -> None:
    """Compare torch vs onnxruntime outputs on a fixture image; raise on mismatch."""
    import numpy as np
    import torch

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_ML_HINT) from exc

    pil = _fixture_image(resolution)
    tensor = _preprocess(pil, resolution)  # numpy [1,3,R,R] float32

    # torch forward on the underlying module.
    module = _inner_module(model)
    module.eval()
    with torch.no_grad():
        torch_out = module(torch.from_numpy(tensor))
    torch_arrays = _flatten_to_numpy(torch_out)

    # onnxruntime forward.
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_arrays = [np.asarray(a) for a in sess.run(None, {input_name: tensor})]

    if len(torch_arrays) != len(onnx_arrays):
        raise RuntimeError(
            f"parity check: output count differs (torch={len(torch_arrays)}, "
            f"onnx={len(onnx_arrays)}). Check input/output names in the export."
        )
    # Match outputs by shape (export may reorder them), then compare values.
    onnx_remaining = list(onnx_arrays)
    max_abs = 0.0
    for t in torch_arrays:
        match = _pop_matching_shape(onnx_remaining, t.shape)
        if match is None:
            raise RuntimeError(
                f"parity check: no ONNX output with shape {tuple(t.shape)} "
                f"(remaining ONNX shapes: {[tuple(a.shape) for a in onnx_remaining]})."
            )
        diff = (
            float(np.max(np.abs(t.astype("float64") - match.astype("float64")))) if t.size else 0.0
        )
        max_abs = max(max_abs, diff)
        if not np.allclose(t, match, atol=atol, rtol=rtol):
            raise RuntimeError(
                "ONNX export PARITY CHECK FAILED: torch vs onnxruntime outputs "
                f"diverge beyond tolerance (max|Δ|={diff:.3e} > atol={atol:.1e}, "
                f"rtol={rtol:.1e}) on output shape {tuple(t.shape)}. The export is "
                "unreliable — do not ship it."
            )
    log.info("export.parity_detail", max_abs_diff=max_abs, n_outputs=len(torch_arrays))


# --------------------------------------------------------------------------- #
# Fixture + preprocessing helpers
# --------------------------------------------------------------------------- #
def _fixture_image(resolution: int) -> Image:
    """A real fixture image for parity: the first image of the active dataset.

    Falls back to a deterministic synthetic RGB image only if no dataset image is
    on disk (so the parity check still runs in a bare checkout). This is NOT a
    detector stub — it is only the *input* used to compare two backends.
    """
    from PIL import Image as PILImage

    ds = get_active_dataset()
    root = Path(ds["root"])
    coco_name = ds.get("coco_filename", "_annotations.coco.json")
    for split in ds.get("splits", {}).values():
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        imgs = sorted(
            p
            for p in split_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name != coco_name
        )
        if imgs:
            log.info("export.fixture_image", path=str(imgs[0]))
            return PILImage.open(imgs[0]).convert("RGB")

    log.warning(
        "export.fixture_synthetic", note="no dataset image found; using synthetic parity input"
    )
    import numpy as np

    rng = np.random.default_rng(1337)
    arr = (rng.uniform(0, 255, size=(resolution, resolution, 3))).astype("uint8")
    return PILImage.fromarray(arr, mode="RGB")


def _preprocess(pil: Image, resolution: int) -> Any:
    """Letterbox + ImageNet-normalize to CHW float32 [1,3,R,R] (matches infer.py)."""
    import numpy as np
    from PIL import Image as PILImage

    w, h = pil.size
    scale = min(resolution / w, resolution / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = pil.resize((new_w, new_h), PILImage.BILINEAR)
    canvas = PILImage.new("RGB", (resolution, resolution), (114, 114, 114))
    canvas.paste(resized, ((resolution - new_w) // 2, (resolution - new_h) // 2))
    arr = np.asarray(canvas, dtype="float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(chw, dtype="float32")


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _inner_module(model: Any) -> Any:
    """Return the underlying ``torch.nn.Module`` from an RF-DETR wrapper.

    RF-DETR keeps the nn.Module under ``.model`` (sometimes ``.model.model``);
    fall back to the wrapper itself if it is already a Module.
    """
    inner = getattr(model, "model", None)
    if inner is not None:
        nested = getattr(inner, "model", None)
        if nested is not None and hasattr(nested, "forward"):
            return nested
        if hasattr(inner, "forward"):
            return inner
    return model


def _flatten_to_numpy(obj: Any) -> list[Any]:
    """Flatten a torch output (Tensor / tuple / list / dict) to a list of ndarrays."""
    import numpy as np
    import torch

    arrays: list[Any] = []

    def _walk(x: Any) -> None:
        if isinstance(x, torch.Tensor):
            arrays.append(x.detach().cpu().numpy())
        elif isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, list | tuple):
            for v in x:
                _walk(v)
        elif isinstance(x, np.ndarray):
            arrays.append(x)
        # Ignore non-array leaves (e.g. None, scalars from aux heads).

    _walk(obj)
    return arrays


def _pop_matching_shape(arrays: list[Any], shape: tuple[int, ...]) -> Any | None:
    """Pop and return the first array in ``arrays`` whose shape matches ``shape``."""
    for i, a in enumerate(arrays):
        if tuple(a.shape) == tuple(shape):
            return arrays.pop(i)
    return None
