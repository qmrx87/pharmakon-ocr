"""Resolve dataset + class-schema bindings for a detection config.

Detection configs may now BIND to a specific dataset and class list independently
of the global ``VIGNOCR_DATA_ACTIVE`` selector:

    Stage A (vignette)  rfdetr_vignette.yaml   dataset: vignette  ->  data2/, 3 classes
    Stage B (fields)    rfdetr_medium.yaml     dataset: real      ->  data/,  17 classes

Class names + count come from (first hit wins):
    1. cfg.model.class_names + cfg.model.num_classes  (Stage A: explicit)
    2. the dataset's class_names  (data.yaml: datasets.<name>.class_names)
    3. configs/classes.yaml via get_classes()  (Stage B default)

So one well-named knob — ``dataset:`` — pivots every detection entrypoint to the
right data + head width with zero call-site changes.
"""

from __future__ import annotations

from typing import Any

from vignocr.common import get_active_dataset, get_classes, get_dataset


def resolve_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """The dataset block this config trains/evals against."""
    name = cfg.get("dataset")
    return get_dataset(str(name)) if name else get_active_dataset()


def resolve_class_schema(
    cfg: dict[str, Any], ds: dict[str, Any] | None = None
) -> tuple[int, list[str]]:
    """``(num_classes, class_names)`` for the detector head.

    Sources in order: ``cfg.model`` -> dataset block -> ``classes.yaml``.
    """
    model_cfg = cfg.get("model") or {}
    n_cfg = model_cfg.get("num_classes")
    names_cfg = model_cfg.get("class_names")
    if n_cfg is not None and names_cfg:
        return int(n_cfg), list(names_cfg)

    if ds is None:
        ds = resolve_dataset(cfg)
    ds_names = ds.get("class_names")
    if ds_names:
        return len(ds_names), list(ds_names)

    schema = get_classes()
    return schema.num_classes, list(schema.names)


# cfg.model.name -> rfdetr wrapper class name. Until 2026-06 the `name` knob was
# silently IGNORED (train/infer/export all hardcoded RFDETRMedium), so a config
# could claim `rfdetr_nano` and still train a medium. Resolved lazily so this
# module stays importable without the [ml] extra.
_RFDETR_VARIANTS: dict[str, str] = {
    "rfdetr_nano": "RFDETRNano",
    "rfdetr_small": "RFDETRSmall",
    "rfdetr_medium": "RFDETRMedium",
    "rfdetr_base": "RFDETRBase",
    "rfdetr_large": "RFDETRLarge",
}


def resolve_model_class(cfg: dict[str, Any]) -> Any:
    """Return the rfdetr model class for ``cfg.model.name`` (lazy ML import).

    Falls back to ``RFDETRMedium`` with a clear error if the installed rfdetr
    build lacks the requested variant (older wheels ship fewer sizes).
    """
    name = str((cfg.get("model") or {}).get("name", "rfdetr_medium")).lower()
    cls_name = _RFDETR_VARIANTS.get(name)
    if cls_name is None:
        raise ValueError(
            f"unknown detection model name {name!r} in config; "
            f"expected one of {sorted(_RFDETR_VARIANTS)}"
        )
    import rfdetr  # noqa: PLC0415 - lazy: keeps the core CPU-importable

    cls = getattr(rfdetr, cls_name, None)
    if cls is None:
        raise ImportError(
            f"config requests model {name!r} but the installed rfdetr "
            f"({getattr(rfdetr, '__version__', '?')}) does not export {cls_name}. "
            "Upgrade rfdetr or pick a variant the installed build ships."
        )
    return cls
