"""
Backward-compatible alias module for ``triage.mapping``.

The multi-factor severity rules in ``severity.rules`` now handle cluster spread
and dominant area before triage runs, so this module simply exposes ``triage``
and ``get_disclaimer`` from ``triage.mapping`` without secondary DSI thresholds
or override rules (PRD Induk TR-1).
"""

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
