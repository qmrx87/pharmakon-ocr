"""Config loading + the canonical field schema.

This module is the single entry point to ``configs/``. Nothing else in the
codebase should open a YAML file directly or hardcode a class name, path,
threshold, or regex — they all flow from here.

Repo-root resolution order:
    1. env ``VIGNOCR_REPO_ROOT``
    2. walk up from this file until a dir containing ``configs/classes.yaml`` is found
Config-dir override: env ``VIGNOCR_CONFIG_DIR`` (defaults to ``<repo>/configs``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Absolute path to the pharmakon-ocr repo root."""
    env = os.environ.get("VIGNOCR_REPO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "classes.yaml").exists():
            return parent
    # Fallback: src/vignocr/common/config.py -> parents[3] == repo root
    return here.parents[3]


def config_dir() -> Path:
    env = os.environ.get("VIGNOCR_CONFIG_DIR")
    return Path(env).resolve() if env else repo_root() / "configs"


def resolve_path(p: str | os.PathLike[str]) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    path = Path(p)
    return path if path.is_absolute() else (repo_root() / path)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=32)
def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` (``name`` may be nested, e.g. 'parsing/fields')."""
    rel = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    return load_yaml(config_dir() / rel)


# --------------------------------------------------------------------------- #
# Field schema (single source of truth)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClassSchema:
    """Parsed view of ``configs/classes.yaml``."""

    classes: tuple[dict[str, Any], ...]
    roles: dict[str, Any]
    business_critical_fields: tuple[str, ...]
    reimbursability: dict[str, Any]
    _name2id: dict[str, int] = field(default_factory=dict)
    _id2name: dict[int, str] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        """Class names ordered by training id."""
        return [c["name"] for c in sorted(self.classes, key=lambda c: c["id"])]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def id_of(self, name: str) -> int:
        return self._name2id[name]

    def name_of(self, cid: int) -> str:
        return self._id2name[cid]

    def by_name(self, name: str) -> dict[str, Any]:
        for c in self.classes:
            if c["name"] == name:
                return c
        raise KeyError(f"unknown class: {name!r}")

    def role(self, key: str) -> Any:
        return self.roles.get(key)

    def is_rotated(self, name: str) -> bool:
        return name in set(self.roles.get("rotated_fields", []))


@lru_cache(maxsize=1)
def get_classes() -> ClassSchema:
    cfg = load_config("classes")
    classes = tuple(cfg["classes"])
    name2id = {c["name"]: int(c["id"]) for c in classes}
    id2name = {int(c["id"]): c["name"] for c in classes}
    return ClassSchema(
        classes=classes,
        roles=cfg.get("roles", {}),
        business_critical_fields=tuple(cfg.get("business_critical_fields", [])),
        reimbursability=cfg.get("reimbursability", {}),
        _name2id=name2id,
        _id2name=id2name,
    )


# --------------------------------------------------------------------------- #
# Active dataset (synthetic now / real later — 12-factor overridable)
# --------------------------------------------------------------------------- #


def get_active_dataset() -> dict[str, Any]:
    """Resolve ``data.yaml``'s active dataset with an absolute ``root`` path.

    Override the active dataset at runtime with env ``VIGNOCR_DATA_ACTIVE``.
    """
    cfg = load_config("data")
    active = os.environ.get("VIGNOCR_DATA_ACTIVE", cfg.get("active", "synthetic"))
    ds = dict(cfg["datasets"][active])
    ds["name"] = active
    ds["root"] = str(resolve_path(ds["root"]))
    ds["_nomenclature"] = cfg.get("nomenclature", {})
    ds["_integrity"] = cfg.get("integrity", {})
    return ds
