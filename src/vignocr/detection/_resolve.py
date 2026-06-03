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
