"""Structured logging — one consistent strategy for the whole codebase.

Usage::

    from vignocr.common import get_logger, configure_logging
    configure_logging()                # once, at process start (idempotent)
    log = get_logger(__name__)
    log.info("extracted", field="ppa", confidence=0.97)
"""

from __future__ import annotations

import logging
import os

import structlog

_CONFIGURED = False


def configure_logging(level: str | None = None, *, json: bool | None = None) -> None:
    """Configure structlog + stdlib logging. Idempotent.

    Level from arg or env ``VIGNOCR_LOG_LEVEL`` (default INFO).
    JSON renderer when ``json`` is True or env ``VIGNOCR_LOG_JSON=1`` (good for prod).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("VIGNOCR_LOG_LEVEL", "INFO")).upper()
    use_json = json if json is not None else os.environ.get("VIGNOCR_LOG_JSON") == "1"

    logging.basicConfig(format="%(message)s", level=getattr(logging, lvl, logging.INFO))
    renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, lvl, logging.INFO)),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
