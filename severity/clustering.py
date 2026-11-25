"""
Damage Point Clustering Module
==============================

Groups overlapping or nearby damage bounding boxes into connected clusters
based on a configurable fraction of the image diagonal.

Implements requirements CL-1 through CL-5 from the PRD.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from severity.dsi import Box, clip_box, union_area
from severity.rules import load_severity_config


def _rect_distance(a: Box, b: Box) -> float:
    """Return Euclidean gap distance between two axis-aligned boxes (0 if touching/overlapping)."""
    dx = max(0, b[0] - a[2], a[0] - b[2])
    dy = max(0, b[1] - a[3], a[1] - b[3])
    return math.hypot(dx, dy)


def _boxes_connected(a: Box, b: Box, threshold_px: float) -> bool:
    """Return True if boxes overlap/touch or lie within *threshold_px* of each other."""
    if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
        return True
    return _rect_distance(a, b) < threshold_px


def cluster_boxes(
    boxes: Sequence[Sequence[float]],
    gap_ratio: Optional[float] = None,
    shape: Optional[Sequence[int]] = None,
    body_area: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Group damage bounding boxes into spatial clusters (CL-1 through CL-5).

    Parameters
    ----------
    boxes : sequence of (x1, y1, x2, y2)
        Damage bounding boxes in image pixel coordinates.
    gap_ratio : float or None, optional
        Fraction of image diagonal used as merge distance threshold (default from config: 0.05).
    shape : sequence of int or None, optional
        Image shape ``(H, W)`` or ``(H, W, C)``. If omitted, inferred from boxes.
    body_area : float or None, optional
        Vehicle body area in pixels for computing each cluster's ratio (CL-4).
        Falls back to image area if omitted or non-positive.
    config : dict or None, optional
        Optional severity config dictionary override.

    Returns
    -------
    list of dict
        Sorted list of cluster dicts, each containing:
        - ``cluster_id`` (int, 1-based)
        - ``indices`` (list[int], 0-based indices into input ``boxes``)
        - ``boxes`` (list[Box])
        - ``union_area`` (int)
        - ``ratio`` (float, 0.0-1.0 vs body area)
        - ``ratio_pct`` (float, percentage 0.0-100.0)
        - ``bbox`` (Box enclosing all boxes in the cluster)
    """
    box_list = list(boxes or [])
    if not box_list:
        return []

    cfg = load_severity_config(config)
    eff_gap_ratio = float(cfg["gap_ratio"] if gap_ratio is None else gap_ratio)

    if shape is not None:
        h, w = int(shape[0]), int(shape[1])
    else:
        w = max(1, max(int(round(b[2])) for b in box_list))
        h = max(1, max(int(round(b[3])) for b in box_list))

    valid_items: List[Tuple[int, Box]] = []
    for idx, b in enumerate(box_list):
        clipped = clip_box(b, (h, w))
        if clipped is not None:
            valid_items.append((idx, clipped))

    if not valid_items:
        return []

    diagonal = math.hypot(w, h)
    threshold_px = max(0.0, eff_gap_ratio) * diagonal
    eff_body_area = float(body_area) if body_area is not None and body_area > 0 else float(w * h)

    n = len(valid_items)
    visited = [False] * n
    raw_components: List[List[Tuple[int, Box]]] = []

    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        component: List[Tuple[int, Box]] = []

        while queue:
            cur = queue.popleft()
            component.append(valid_items[cur])
            cur_box = valid_items[cur][1]
            for nxt in range(n):
                if not visited[nxt] and _boxes_connected(cur_box, valid_items[nxt][1], threshold_px):
                    visited[nxt] = True
                    queue.append(nxt)

        raw_components.append(component)

    # Sort deterministically top-to-bottom, left-to-right by enclosing bbox
    def _enclosing(comp: List[Tuple[int, Box]]) -> Box:
        xs1 = min(item[1][0] for item in comp)
        ys1 = min(item[1][1] for item in comp)
        xs2 = max(item[1][2] for item in comp)
        ys2 = max(item[1][3] for item in comp)
        return (xs1, ys1, xs2, ys2)

    raw_components.sort(key=lambda comp: (_enclosing(comp)[1], _enclosing(comp)[0], comp[0][0]))

    clusters: List[Dict[str, Any]] = []
    for cluster_idx, comp in enumerate(raw_components, start=1):
        comp_indices = [item[0] for item in comp]
        comp_boxes = [item[1] for item in comp]
        area_px = union_area(comp_boxes, (h, w))
        ratio = min(1.0, max(0.0, area_px / eff_body_area)) if eff_body_area > 0 else 0.0
        clusters.append(
            {
                "cluster_id": cluster_idx,
                "indices": comp_indices,
                "boxes": comp_boxes,
                "union_area": int(area_px),
                "ratio": ratio,
                "ratio_pct": round(ratio * 100.0, 1),
                "bbox": _enclosing(comp),
            }
        )

    return clusters
