"""``vignocr`` command-line entry point (wired to ``[project.scripts]``).

Three subcommands, all runnable on a CPU-only box today (no ``[ml]`` extra
required — the pipeline falls back to deterministic fixture stubs):

    vignocr extract <image> [--flow selling|receiving]   # -> ExtractionRecord JSON
    vignocr stats                                         # -> dataset summary JSON
    vignocr gen-fixtures                                  # (re)generate the synthetic fixture

``extract`` prints a :class:`~vignocr.common.schemas.ExtractionRecord` as JSON
(money already serialized as Decimal strings). ``stats`` prints the active
dataset's per-class/box summary. ``gen-fixtures`` regenerates the synthetic
dataset from ``configs/data.yaml`` deterministically.

Heavy ML libs are never imported here; the pipeline imports them lazily only when
a real backend actually runs. Output goes to stdout as JSON so the CLI composes
with ``jq`` and friends.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Keep stdout clean for JSON: default the (idempotent, read-once) logging level to
# WARNING *before* importing vignocr, since importing it configures logging on the
# first ``get_logger`` call and the level is then fixed for the process. ``--verbose``
# must therefore be honoured here, pre-import, by raising the level to INFO. A user
# can also set VIGNOCR_LOG_LEVEL explicitly (it wins over this default).
_VERBOSE = any(a in ("--verbose", "-v") for a in sys.argv[1:])
os.environ.setdefault("VIGNOCR_LOG_LEVEL", "INFO" if _VERBOSE else "WARNING")

from vignocr.common import configure_logging, get_logger  # noqa: E402
from vignocr.common.config import get_active_dataset  # noqa: E402
from vignocr.common.schemas import ExtractionRecord, Flow  # noqa: E402

log = get_logger(__name__)

_FLOWS: tuple[str, ...] = ("selling", "receiving")


# --------------------------------------------------------------------------- #
# Subcommand: extract
# --------------------------------------------------------------------------- #


def _cmd_extract(args: argparse.Namespace) -> int:
    """Run one image through the pipeline and print its ExtractionRecord as JSON."""
    image_path = Path(args.image)
    if not image_path.exists():
        _fail(f"image not found: {image_path}")
        return 2

    # Imported here (not at module top) so `vignocr stats|gen-fixtures` never even
    # constructs the pipeline; and the pipeline keeps ML libs lazy regardless.
    from vignocr.pipeline import VignocrPipeline  # noqa: PLC0415

    cfg: dict[str, Any] = {"default_flow": args.flow}
    if args.backend:
        cfg["backend"] = args.backend
    if args.detector:
        cfg["detector_path"] = args.detector
    if args.nomenclature_csv:
        cfg["nomenclature_csv"] = args.nomenclature_csv

    flow: Flow = args.flow  # validated by argparse choices
    pipeline = VignocrPipeline(cfg)
    record: ExtractionRecord = pipeline.extract(str(image_path), flow=flow)

    _print_json(record.model_dump(mode="json"), pretty=not args.compact)
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: stats
# --------------------------------------------------------------------------- #


def _cmd_stats(args: argparse.Namespace) -> int:
    """Print the active dataset's summary (generating the fixture if needed)."""
    from vignocr.data.stats import summarize  # noqa: PLC0415

    ds = get_active_dataset()
    root = Path(ds["root"])

    if not root.exists():
        if ds.get("name") == "synthetic":
            log.info("stats.generating_missing_fixture", root=str(root))
            _generate_fixtures()
        else:
            _fail(f"dataset root does not exist: {root} (active dataset: {ds.get('name')})")
            return 2

    summary = summarize(root)
    _print_json(summary, pretty=not args.compact)
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: gen-fixtures
# --------------------------------------------------------------------------- #


def _cmd_gen_fixtures(args: argparse.Namespace) -> int:
    """(Re)generate the synthetic fixture from configs/data.yaml deterministically."""
    root = _generate_fixtures()
    _print_json(
        {"generated": True, "root": str(root), "dataset": "synthetic"},
        pretty=not args.compact,
    )
    return 0


def _generate_fixtures() -> Path:
    """Regenerate the active synthetic dataset; returns its root path."""
    # Local import keeps Pillow/synthetic out of the `extract`/`stats` fast paths.
    from vignocr.data.synthetic import _generate_from_config  # noqa: PLC0415

    return _generate_from_config()


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    # Global flags live on a shared *parent* parser so they are accepted both
    # before the subcommand (``vignocr --compact extract ...``) and after it
    # (``vignocr extract ... --compact``) — the latter is what users type.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--compact",
        action="store_true",
        help="Emit single-line JSON (default is indented).",
    )
    common.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Emit INFO logs (default suppresses them so stdout stays pure JSON). "
        "For debugging — do not combine with JSON piping.",
    )

    parser = argparse.ArgumentParser(
        prog="vignocr",
        description="VignOCR — OCR extraction for Algerian pharmaceutical vignettes.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # extract -----------------------------------------------------------------
    p_extract = sub.add_parser(
        "extract",
        parents=[common],
        help="Extract structured fields from a vignette image (prints ExtractionRecord JSON).",
    )
    p_extract.add_argument("image", help="Path to the vignette image.")
    p_extract.add_argument(
        "--flow",
        choices=_FLOWS,
        default="selling",
        help="Abstention profile: 'selling' (stricter) or 'receiving' (default: selling).",
    )
    p_extract.add_argument(
        "--backend",
        choices=("auto", "stub", "real"),
        default=None,
        help="Backend selection (default: from config / auto). 'stub' forces the "
        "fixture-backed CPU path; 'real' requires [ml] + weights.",
    )
    p_extract.add_argument(
        "--detector",
        default=None,
        help="Path to detector weights (ONNX/ckpt) for the real backend.",
    )
    p_extract.add_argument(
        "--nomenclature-csv",
        default=None,
        help="Override the nomenclature CSV path (else from config).",
    )
    p_extract.set_defaults(func=_cmd_extract)

    # stats -------------------------------------------------------------------
    p_stats = sub.add_parser(
        "stats",
        parents=[common],
        help="Summarize the active dataset (per-class counts, box sizes, split sizes).",
    )
    p_stats.set_defaults(func=_cmd_stats)

    # gen-fixtures ------------------------------------------------------------
    p_gen = sub.add_parser(
        "gen-fixtures",
        parents=[common],
        help="(Re)generate the synthetic fixture dataset from configs/data.yaml.",
    )
    p_gen.set_defaults(func=_cmd_gen_fixtures)

    return parser


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _print_json(payload: Any, *, pretty: bool) -> None:
    """Write ``payload`` as JSON to stdout (UTF-8, indented unless ``--compact``)."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    print(text)


def _fail(message: str) -> None:
    """Print an error to stderr (stdout stays clean for JSON consumers)."""
    print(f"vignocr: error: {message}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success).

    Wired to ``[project.scripts] vignocr = vignocr.cli:main`` so ``vignocr ...``
    works after ``pip install -e .``.
    """
    # Logging level was fixed at import time from VIGNOCR_LOG_LEVEL (raised to INFO
    # above when --verbose/-v is present on argv), so configure_logging() here is a
    # no-op confirmation rather than a re-config.
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        _fail(str(exc))
        return 2
    except ImportError as exc:
        # Real backend selected without the [ml] extra / weights, or a missing
        # optional dep. Surface the actionable message, not a traceback.
        _fail(str(exc))
        return 3
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _fail("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    raise SystemExit(main())
