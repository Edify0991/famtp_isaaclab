"""Logging helpers."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure package-level logging format and verbosity."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
