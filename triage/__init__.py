"""Triage & SOP recommendation package."""

from triage.mapping import (
    DEFAULT_DISCLAIMER,
    DEFAULT_TRIAGE_TABLE,
    get_disclaimer,
    load_triage_config,
    triage,
)

__all__ = [
    "DEFAULT_DISCLAIMER",
    "DEFAULT_TRIAGE_TABLE",
    "get_disclaimer",
    "load_triage_config",
    "triage",
]
