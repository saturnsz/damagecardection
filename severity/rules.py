"""
Multi-Factor Severity Rule Engine
=================================

Evaluates DSI, cluster count, and dominant cluster ratio in strict priority order:
    Rusak Parah -> Rusak Sedang -> Rusak Ringan (first match wins)

Implements requirements SV-1 through SV-7 from the PRD.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Configuration defaults (SV-7)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "dsi_severe": 40.0,
    "dsi_moderate": 15.0,
    "cluster_count_upgrade": 2,
    "dominant_cluster_ratio": 0.25,
    "gap_ratio": 0.05,
}

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "severity.yaml",
)

_SEVERITY_STYLES: Dict[str, Dict[str, str]] = {
    "Rusak Ringan": {
        "level": "Minor",
        "level_id": "Ringan",
        "color": "#28a745",
        "icon": "✅",
    },
    "Rusak Sedang": {
        "level": "Moderate",
        "level_id": "Sedang",
        "color": "#ffc107",
        "icon": "⚠️",
    },
    "Rusak Parah": {
        "level": "Severe",
        "level_id": "Parah",
        "color": "#dc3545",
        "icon": "🔴",
    },
    "Tidak ada kerusakan terdeteksi": {
        "level": "None",
        "level_id": "Tidak Ada Kerusakan",
        "color": "#17a2b8",
        "icon": "ℹ️",
    },
}


def load_severity_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load severity thresholds from ``config/severity.yaml`` merged with defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if os.path.isfile(_CONFIG_PATH):
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            cfg.update(loaded)
    if config:
        cfg.update(config)

    # Normalize ratio if configured as percentage > 1 (e.g. 25 instead of 0.25)
    dom_ratio = float(cfg["dominant_cluster_ratio"])
    if dom_ratio > 1.0:
        cfg["dominant_cluster_ratio"] = dom_ratio / 100.0
    gap = float(cfg["gap_ratio"])
    if gap > 1.0:
        cfg["gap_ratio"] = gap / 100.0

    return cfg


def classify(
    dsi: float,
    n_clusters: int,
    max_cluster_ratio: float,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify overall damage severity using multi-factor rules (SV-1 through SV-7).

    Parameters
    ----------
    dsi : float
        Damage Severity Index in percentage (0.0-100.0).
    n_clusters : int
        Number of damage point clusters.
    max_cluster_ratio : float
        Largest single cluster area divided by vehicle body area (0.0-1.0).
        Values > 1.0 are interpreted as percentages and divided by 100.
    cfg : dict or None, optional
        Optional threshold configuration override.

    Returns
    -------
    dict
        Contains ``label``, ``level``, ``level_id``, ``dsi``, ``n_clusters``,
        ``max_cluster_ratio``, ``reason``, ``color``, ``icon``, ``rule_matched``.
    """
    if isinstance(dsi, bool) or not isinstance(dsi, (int, float)) or not (0.0 <= float(dsi) <= 100.0):
        raise ValueError(f"DSI must be between 0 and 100, got {dsi}")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 0:
        raise ValueError(f"n_clusters must be a non-negative integer, got {n_clusters}")

    eff_cfg = load_severity_config(cfg)
    dsi_severe = float(eff_cfg["dsi_severe"])
    dsi_moderate = float(eff_cfg["dsi_moderate"])
    cluster_upgrade = int(eff_cfg["cluster_count_upgrade"])
    dominant_thresh = float(eff_cfg["dominant_cluster_ratio"])

    dsi_val = float(dsi)
    dom_ratio = float(max_cluster_ratio)
    if dom_ratio > 1.0:
        dom_ratio = dom_ratio / 100.0

    # DS-5: No damage boxes detected
    if n_clusters == 0:
        label = "Tidak ada kerusakan terdeteksi"
        reason = "Tidak ada kotak kerusakan terdeteksi pada kendaraan (DSI 0,0%, 0 titik)."
        rule_matched = "no_damage"
    # SV-1 / SV-2 / SV-5: Rusak Parah evaluated first
    elif dsi_val >= dsi_severe or dom_ratio >= dominant_thresh:
        label = "Rusak Parah"
        dsi_hit = dsi_val >= dsi_severe
        dom_hit = dom_ratio >= dominant_thresh
        if dsi_hit and dom_hit:
            reason = (
                f"DSI {dsi_val:.1f}% (>= {dsi_severe:.0f}%) dan klaster terbesar "
                f"{dom_ratio * 100:.1f}% bodi (>= {dominant_thresh * 100:.0f}%) -> Rusak Parah"
            )
            rule_matched = "severe_dsi_and_dominant_cluster"
        elif dsi_hit:
            reason = (
                f"DSI {dsi_val:.1f}% (>= {dsi_severe:.0f}%) pada {n_clusters} titik kerusakan -> Rusak Parah"
            )
            rule_matched = "severe_dsi"
        else:
            reason = (
                f"DSI {dsi_val:.1f}% tetapi satu klaster dominan mencapai {dom_ratio * 100:.1f}% bodi "
                f"(>= {dominant_thresh * 100:.0f}%) -> Rusak Parah"
            )
            rule_matched = "severe_dominant_cluster"
    # SV-3 / SV-5: Rusak Sedang evaluated second
    elif dsi_val >= dsi_moderate or n_clusters > cluster_upgrade:
        label = "Rusak Sedang"
        if dsi_val >= dsi_moderate:
            reason = (
                f"DSI {dsi_val:.1f}% berada di rentang sedang ({dsi_moderate:.0f}% hingga "
                f"<{dsi_severe:.0f}%) dengan {n_clusters} titik kerusakan -> Rusak Sedang"
            )
            rule_matched = "moderate_dsi"
        else:
            reason = (
                f"DSI {dsi_val:.1f}% (< {dsi_moderate:.0f}%) tetapi terdapat {n_clusters} titik "
                f"tersebar (> {cluster_upgrade} titik) -> Rusak Sedang"
            )
            rule_matched = "moderate_spread_clusters"
    # SV-4: Rusak Ringan
    else:
        label = "Rusak Ringan"
        reason = (
            f"DSI {dsi_val:.1f}% (< {dsi_moderate:.0f}%) dengan {n_clusters} titik kerusakan "
            f"(<= {cluster_upgrade} titik) dan klaster terbesar {dom_ratio * 100:.1f}% -> Rusak Ringan"
        )
        rule_matched = "minor_default"

    style = _SEVERITY_STYLES[label]
    return {
        "label": label,
        "level": style["level"],
        "level_id": style["level_id"],
        "dsi": round(dsi_val, 1),
        "n_clusters": n_clusters,
        "max_cluster_ratio": dom_ratio,
        "max_cluster_pct": round(dom_ratio * 100.0, 1),
        "reason": reason,
        "color": style["color"],
        "icon": style["icon"],
        "rule_matched": rule_matched,
    }
