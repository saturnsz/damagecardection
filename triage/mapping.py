"""
Triage & SOP Mapping Module
===========================

Maps a severity **label** produced by ``severity.rules.classify()`` to a
recommended claim handling path, workshop SOP, and estimated duration using
the configurable table in ``config/triage.yaml``.

Implements requirements TR-1 through TR-6 from the PRD Induk:
- No secondary DSI thresholding or box-count override logic lives here.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import yaml

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "triage.yaml",
)

DEFAULT_TRIAGE_TABLE: Dict[str, Dict[str, str]] = {
    "Rusak Ringan": {
        "level": "Minor",
        "level_id": "Ringan",
        "path": "Fast-Track",
        "sop": "Compound dan Spot Repaint",
        "estimate": "1-2 hari kerja",
        "color": "#28a745",
        "icon": "✅",
    },
    "Rusak Sedang": {
        "level": "Moderate",
        "level_id": "Sedang",
        "path": "Standard Workshop Review",
        "sop": "Ketok Panel, Pendempulan, dan Full Panel Respray",
        "estimate": "3-5 hari kerja",
        "color": "#ffc107",
        "icon": "⚠️",
    },
    "Rusak Parah": {
        "level": "Severe",
        "level_id": "Parah",
        "path": "Specialist Escalation",
        "sop": "Penggantian Panel Baru",
        "estimate": "> 5 hari kerja",
        "color": "#dc3545",
        "icon": "🔴",
    },
    "Tidak ada kerusakan terdeteksi": {
        "level": "None",
        "level_id": "Tidak Ada Kerusakan",
        "path": "Tidak Diperlukan Perbaikan",
        "sop": "Tidak ada tindakan perbaikan yang diperlukan",
        "estimate": "0 hari kerja",
        "color": "#17a2b8",
        "icon": "ℹ️",
    },
}

DEFAULT_DISCLAIMER: str = (
    "Rekomendasi ini merupakan hasil analisis awal sistem otomatis berbasis luas area, "
    "tidak mengukur kedalaman atau kerusakan struktural tersembunyi, dan tidak "
    "menggantikan pemeriksaan fisik oleh teknisi berpengalaman. Keputusan akhir "
    "penanganan sepenuhnya berada pada pemeriksa yang berwenang."
)

_LABEL_ALIASES: Dict[str, str] = {
    "ringan": "Rusak Ringan",
    "minor": "Rusak Ringan",
    "rusak ringan": "Rusak Ringan",
    "sedang": "Rusak Sedang",
    "moderate": "Rusak Sedang",
    "rusak sedang": "Rusak Sedang",
    "parah": "Rusak Parah",
    "severe": "Rusak Parah",
    "rusak parah": "Rusak Parah",
    "tidak ada kerusakan terdeteksi": "Tidak ada kerusakan terdeteksi",
    "tidak ada kerusakan": "Tidak ada kerusakan terdeteksi",
    "none": "Tidak ada kerusakan terdeteksi",
}


def load_triage_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load triage lookup table and disclaimer from ``config/triage.yaml``."""
    cfg: Dict[str, Any] = {
        "mapping": deepcopy(DEFAULT_TRIAGE_TABLE),
        "disclaimer": DEFAULT_DISCLAIMER,
    }
    if os.path.isfile(_CONFIG_PATH):
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            if isinstance(loaded.get("mapping"), dict):
                merged_map = deepcopy(DEFAULT_TRIAGE_TABLE)
                for key, val in loaded["mapping"].items():
                    if isinstance(val, dict):
                        base = dict(merged_map.get(key, {}))
                        base.update(val)
                        merged_map[key] = base
                cfg["mapping"] = merged_map
            if isinstance(loaded.get("disclaimer"), str) and loaded["disclaimer"].strip():
                cfg["disclaimer"] = loaded["disclaimer"].strip()
    if config:
        if isinstance(config.get("mapping"), dict):
            for key, val in config["mapping"].items():
                if isinstance(val, dict):
                    base = dict(cfg["mapping"].get(key, {}))
                    base.update(val)
                    cfg["mapping"][key] = base
        if isinstance(config.get("disclaimer"), str) and config["disclaimer"].strip():
            cfg["disclaimer"] = config["disclaimer"].strip()
    return cfg


def triage(label: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map a severity label from feature 5.2 to triage path, SOP, and duration (TR-1..TR-6).

    Parameters
    ----------
    label : str
        Severity label (e.g. ``"Rusak Ringan"``, ``"Rusak Sedang"``,
        ``"Rusak Parah"``, or ``"Tidak ada kerusakan terdeteksi"``).
    config : dict or None, optional
        Optional override mapping/disclaimer dict.

    Returns
    -------
    dict
        ``label``, ``level``, ``level_id``, ``path``, ``sop``, ``estimate``,
        ``color``, ``icon``.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"Severity label must be a non-empty string, got {label!r}")

    cfg = load_triage_config(config)
    mapping: Dict[str, Dict[str, str]] = cfg["mapping"]

    raw_key = label.strip()
    canonical_key = raw_key if raw_key in mapping else _LABEL_ALIASES.get(raw_key.lower())

    if canonical_key is None or canonical_key not in mapping:
        valid = ", ".join(mapping.keys())
        raise ValueError(f"Unknown severity label {label!r}. Expected one of: {valid}")

    entry = deepcopy(mapping[canonical_key])
    entry["label"] = canonical_key
    return entry


def get_disclaimer(config: Optional[Dict[str, Any]] = None) -> str:
    """Return the configured triage disclaimer text (TR-4, PD-3)."""
    return load_triage_config(config)["disclaimer"]
