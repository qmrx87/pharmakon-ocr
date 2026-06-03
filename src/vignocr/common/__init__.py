"""Cross-cutting utilities: config loading, logging, seeding, metrics.

Public API (import from ``vignocr.common``):
    load_config, get_classes, get_active_dataset, repo_root, ClassSchema
    get_logger, configure_logging
    seed_everything
"""

from vignocr.common.config import (
    ClassSchema,
    get_active_dataset,
    get_classes,
    load_config,
    repo_root,
)
from vignocr.common.logging import configure_logging, get_logger
from vignocr.common.schemas import (
    BBox,
    ChecksumReport,
    ExtractionRecord,
    FieldRead,
    NomenclatureReport,
    Reimbursability,
    money_str,
)
from vignocr.common.seeding import seed_everything

__all__ = [
    "ClassSchema",
    "load_config",
    "get_classes",
    "get_active_dataset",
    "repo_root",
    "configure_logging",
    "get_logger",
    "seed_everything",
    "BBox",
    "FieldRead",
    "Reimbursability",
    "ChecksumReport",
    "NomenclatureReport",
    "ExtractionRecord",
    "money_str",
]
