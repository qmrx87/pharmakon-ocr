"""Load the nomenclature register CSV into an in-memory, key-indexed lookup.

The CSV layout (columns, key column, encoding) is read from
``configs/nomenclature/correction.yaml`` — nothing here is hardcoded. The index
key is the **normalized** ``num_enregistrement`` (same normalization the matcher
applies), so an OCR-read code can be looked up exactly or fuzzy-matched against
the same canonical space.

A missing CSV yields an **empty** index (never an exception): the pipeline must
keep running and simply fall back to keeping the OCR values.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vignocr.common import get_logger, load_config
from vignocr.common.config import resolve_path
from vignocr.nomenclature.match import normalize_code

log = get_logger(__name__)


@dataclass(frozen=True)
class NomenclatureIndex:
    """An in-memory view of the nomenclature register.

    Attributes:
        rows: every parsed row, as ``{column: value}`` dicts (order preserved).
        by_key: rows indexed by the **normalized** key column (last row wins on
            duplicate keys — a warning is logged at load time).
        key_column: the column used as the lookup key.
        columns: the configured column list (the schema this index conforms to).
    """

    rows: tuple[dict[str, str], ...] = ()
    by_key: dict[str, dict[str, str]] = field(default_factory=dict)
    key_column: str = "num_enregistrement"
    columns: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        return not self.rows

    def get(self, normalized_key: str) -> dict[str, str] | None:
        """Exact lookup by an already-normalized key (no fuzzy matching)."""
        return self.by_key.get(normalized_key)

    def keys(self) -> list[str]:
        """All normalized keys in the index (the matcher's search space)."""
        return list(self.by_key.keys())


def _config() -> dict[str, Any]:
    return load_config("nomenclature/correction")


def load_csv(path: str | Path | None = None) -> NomenclatureIndex:
    """Load the nomenclature CSV into a :class:`NomenclatureIndex`.

    Args:
        path: CSV path. Defaults to ``csv.path`` from ``correction.yaml`` (resolved
            against the repo root). Relative paths are repo-root-relative.

    Returns:
        A populated index, or an **empty** index if the file does not exist or is
        empty/headerless. Never raises on a missing file.
    """
    cfg = _config()
    csv_cfg = cfg.get("csv", {})
    key_column: str = csv_cfg.get("key_column", "num_enregistrement")
    columns: tuple[str, ...] = tuple(csv_cfg.get("columns", ()))
    encoding: str = csv_cfg.get("encoding", "utf-8")

    raw_path = path if path is not None else csv_cfg.get("path", "fixtures/nomenclature.csv")
    csv_path = resolve_path(raw_path)

    if not csv_path.exists():
        log.warning("nomenclature.csv_missing", path=str(csv_path))
        return NomenclatureIndex(key_column=key_column, columns=columns)

    rows: list[dict[str, str]] = []
    by_key: dict[str, dict[str, str]] = {}
    duplicates = 0

    with open(csv_path, encoding=encoding, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            log.warning("nomenclature.csv_empty", path=str(csv_path))
            return NomenclatureIndex(key_column=key_column, columns=columns)

        if key_column not in reader.fieldnames:
            log.warning(
                "nomenclature.key_column_absent",
                path=str(csv_path),
                key_column=key_column,
                header=list(reader.fieldnames),
            )

        for raw in reader:
            # Keep only the configured columns; coerce None -> "" and strip whitespace.
            row = {col: (raw.get(col) or "").strip() for col in columns}
            rows.append(row)

            raw_key = row.get(key_column, "")
            norm_key = normalize_code(raw_key, cfg)
            if not norm_key:
                continue
            if norm_key in by_key:
                duplicates += 1
            by_key[norm_key] = row

    if duplicates:
        log.warning("nomenclature.duplicate_keys", path=str(csv_path), count=duplicates)
    log.info(
        "nomenclature.loaded",
        path=str(csv_path),
        rows=len(rows),
        unique_keys=len(by_key),
    )
    return NomenclatureIndex(
        rows=tuple(rows),
        by_key=by_key,
        key_column=key_column,
        columns=columns,
    )
