"""Multi-factor severity assessment package (DSI, clustering, rule engine)."""

from severity.clustering import cluster_boxes
from severity.dsi import clip_box, compute_dsi, select_body_box, union_area
from severity.rules import DEFAULT_CONFIG, classify, load_severity_config

__all__ = [
    "DEFAULT_CONFIG",
    "classify",
    "clip_box",
    "cluster_boxes",
    "compute_dsi",
    "load_severity_config",
    "select_body_box",
    "union_area",
]
