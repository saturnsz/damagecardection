"""
Triage & SOP Recommendation Module
===================================

Maps a DSI (Damage Severity Index) percentage to a severity level,
recommended repair path, SOP description, and time estimate.

This module runs AFTER classification and provides actionable
recommendations based on the computed DSI value.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "override_box_threshold": 3,
}

# ---------------------------------------------------------------------------
# Triage lookup table  (TR-2, TR-3, TR-4)
# ---------------------------------------------------------------------------

TRIAGE_TABLE: List[Dict[str, Any]] = [
    {
        "min_dsi": 0,
        "max_dsi": 15,
        "level": "Minor",
        "level_id": "Ringan",
        "path": "Fast-Track",
        "sop": "Compound dan Spot Repaint",
        "estimate": "1–2 hari kerja",
        "color": "#28a745",   # green
        "icon": "✅",
        "rank": 0,
    },
    {
        "min_dsi": 15,
        "max_dsi": 40,
        "level": "Moderate",
        "level_id": "Sedang",
        "path": "Standard Workshop Review",
        "sop": "Ketok Panel, Pendempulan, dan Full Panel Respray",
        "estimate": "3–5 hari kerja",
        "color": "#ffc107",   # yellow / amber
        "icon": "⚠️",
        "rank": 1,
    },
    {
        "min_dsi": 40,
        "max_dsi": 101,       # exclusive upper bound
        "level": "Severe",
        "level_id": "Parah",
        "path": "Specialist Escalation",
        "sop": "Penggantian Panel Baru",
        "estimate": "> 5 hari kerja",
        "color": "#dc3545",   # red
        "icon": "🔴",
        "rank": 2,
    },
]

# Maximum rank present in the table (used to cap escalation).
_MAX_RANK: int = max(entry["rank"] for entry in TRIAGE_TABLE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lookup_by_dsi(dsi: float) -> Dict[str, Any]:
    """Return the matching triage entry for a given DSI value.

    The lookup uses half-open intervals:
      - DSI < 15        → Minor
      - 15 ≤ DSI < 40   → Moderate
      - DSI ≥ 40        → Severe

    Parameters
    ----------
    dsi : float
        Damage Severity Index (0–100).

    Returns
    -------
    dict
        A *copy* of the matching ``TRIAGE_TABLE`` entry.

    Raises
    ------
    RuntimeError
        If no matching row is found (should never happen with valid DSI).
    """
    for entry in TRIAGE_TABLE:
        if entry["min_dsi"] <= dsi < entry["max_dsi"]:
            return deepcopy(entry)
    raise RuntimeError(f"No triage entry matched DSI={dsi}")


def _lookup_by_rank(rank: int) -> Dict[str, Any]:
    """Return the triage entry with the given rank.

    Parameters
    ----------
    rank : int
        Severity rank (0 = Minor, 1 = Moderate, 2 = Severe).

    Returns
    -------
    dict
        A *copy* of the matching ``TRIAGE_TABLE`` entry.

    Raises
    ------
    KeyError
        If the rank does not exist in the table.
    """
    for entry in TRIAGE_TABLE:
        if entry["rank"] == rank:
            return deepcopy(entry)
    raise KeyError(f"No triage entry with rank={rank}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def triage(
    dsi: float,
    num_boxes: int = 0,
    classes: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map a DSI percentage to a severity level and repair recommendation.

    Implements rules TR-1 through TR-6 from the PRD.

    Parameters
    ----------
    dsi : float
        Damage Severity Index as a percentage (0–100 inclusive).
    num_boxes : int, optional
        Number of detected damage bounding boxes. Used for the override
        rule (TR-6). Defaults to ``0``.
    classes : list of str or None, optional
        Detected damage class labels. Reserved for future override rules.
    config : dict or None, optional
        Runtime configuration overrides. Supported keys:

        - ``override_box_threshold`` (int): If ``num_boxes`` exceeds this
          value the severity is escalated by one level. Default is ``3``.

    Returns
    -------
    dict
        Result dictionary with the following keys:

        - **dsi** (*float*) – The input DSI value.
        - **level** (*str*) – Severity level in English (e.g. ``"Minor"``).
        - **level_id** (*str*) – Severity level in Bahasa Indonesia.
        - **path** (*str*) – Recommended repair path.
        - **sop** (*str*) – Standard operating procedure description.
        - **estimate** (*str*) – Estimated repair duration.
        - **color** (*str*) – Hex colour code for UI display.
        - **icon** (*str*) – Emoji icon for UI display.
        - **override** (*bool*) – Whether an override rule was applied.
        - **override_reason** (*str | None*) – Human-readable reason for
          the override, or ``None`` if no override.
        - **original_level** (*str | None*) – The level before escalation,
          or ``None`` if no override.

    Raises
    ------
    ValueError
        If *dsi* is not in the range 0–100 (inclusive).
    """

    # TR-1: Validate DSI range
    if not (0 <= dsi <= 100):
        raise ValueError(
            f"DSI must be between 0 and 100 (inclusive), got {dsi}"
        )

    # Merge caller config with defaults
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    override_box_threshold: int = cfg["override_box_threshold"]

    # TR-4: Look up base severity from the triage table
    entry = _lookup_by_dsi(dsi)

    # Prepare override tracking
    override: bool = False
    override_reason: Optional[str] = None
    original_level: Optional[str] = None

    # TR-6: Override rule — escalate if num_boxes exceeds threshold
    if num_boxes > override_box_threshold:
        current_rank: int = entry["rank"]
        if current_rank < _MAX_RANK:
            original_level = entry["level"]
            escalated = _lookup_by_rank(current_rank + 1)
            entry = escalated
            override = True
            override_reason = (
                f"Jumlah area kerusakan ({num_boxes}) melebihi "
                f"ambang batas ({override_box_threshold}), "
                f"tingkat keparahan dinaikkan dari {original_level} "
                f"ke {entry['level']}"
            )

    # Build result dict
    result: Dict[str, Any] = {
        "dsi": dsi,
        "level": entry["level"],
        "level_id": entry["level_id"],
        "path": entry["path"],
        "sop": entry["sop"],
        "estimate": entry["estimate"],
        "color": entry["color"],
        "icon": entry["icon"],
        "override": override,
        "override_reason": override_reason,
        "original_level": original_level,
    }

    return result


def get_disclaimer() -> str:
    """Return the standard triage disclaimer text.

    Returns
    -------
    str
        Disclaimer in Bahasa Indonesia stating that the automated
        recommendation does not replace a physical inspection by a
        qualified technician.
    """
    return (
        "Rekomendasi ini merupakan hasil analisis awal sistem otomatis "
        "dan tidak menggantikan pemeriksaan fisik oleh teknisi "
        "berpengalaman. Keputusan akhir penanganan sepenuhnya berada "
        "pada pemeriksa yang berwenang."
    )
