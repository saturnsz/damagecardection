"""
Geometric DSI (Damage Severity Index) Module
==========================================

Computes DSI geometrically from bounding boxes using a binary mask union:
    DSI = (union area of damage boxes / vehicle body area) * 100

Implements requirements DS-1 through DS-6 from the PRD.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]


def _normalize_shape(shape: Sequence[int]) -> Tuple[int, int]:
    """Extract (height, width) from an image shape tuple."""
    if len(shape) < 2:
        raise ValueError(f"Expected shape with at least 2 dimensions, got {shape}")
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(h, w)}")
    return h, w


def _infer_shape(
    boxes: Iterable[Sequence[float]],
    body_box: Optional[Sequence[float]] = None,
) -> Tuple[int, int]:
    """Infer minimal (height, width) enclosing all boxes when shape is omitted."""
    max_x, max_y = 1, 1
    candidates = list(boxes)
    if body_box is not None:
        candidates.append(body_box)
    for box in candidates:
        if len(box) >= 4:
            max_x = max(max_x, int(round(box[2])))
            max_y = max(max_y, int(round(box[3])))
    return max_y, max_x


def clip_box(box: Sequence[float], shape: Sequence[int]) -> Optional[Box]:
    """Clip (x1, y1, x2, y2) coordinates to image bounds (DS-3).

    Returns ``None`` if the clipped box has zero or negative area.
    """
    h, w = _normalize_shape(shape)
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def union_area(boxes: Iterable[Sequence[float]], shape: Sequence[int]) -> int:
    """Compute exact union pixel area of bounding boxes via binary mask (DS-2, DS-3).

    Overlapping regions are counted only once.
    """
    h, w = _normalize_shape(shape)
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        clipped = clip_box(box, (h, w))
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        mask[y1:y2, x1:x2] = 1
    return int(np.count_nonzero(mask))


def _box_intersection_area(box_a: Box, box_b: Box) -> int:
    """Compute pixel intersection area between two boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def select_body_box(
    vehicle_boxes: Sequence[Sequence[float]],
    damage_boxes: Optional[Sequence[Sequence[float]]] = None,
    shape: Optional[Sequence[int]] = None,
) -> Optional[Box]:
    """Select the primary vehicle bounding box for DSI denominator.

    When multiple vehicles appear in the frame (e.g. background traffic), selects
    the vehicle box having the highest intersection with detected damage boxes.
    Falls back to the largest vehicle bounding box by area when no overlap exists
    or when ``damage_boxes`` is empty.
    """
    if not vehicle_boxes:
        return None

    inferred_shape = shape or _infer_shape(list(vehicle_boxes) + list(damage_boxes or []))
    clipped_vehicles = [
        cb for b in vehicle_boxes if (cb := clip_box(b, inferred_shape)) is not None
    ]
    if not clipped_vehicles:
        return None

    clipped_damages = [
        cb for b in (damage_boxes or []) if (cb := clip_box(b, inferred_shape)) is not None
    ]

    def _area(b: Box) -> int:
        return (b[2] - b[0]) * (b[3] - b[1])

    if clipped_damages:
        best_box = max(
            clipped_vehicles,
            key=lambda vb: (
                sum(_box_intersection_area(vb, db) for db in clipped_damages),
                _area(vb),
            ),
        )
        if sum(_box_intersection_area(best_box, db) for db in clipped_damages) > 0:
            return best_box

    return max(clipped_vehicles, key=_area)


def compute_dsi(
    boxes: Sequence[Sequence[float]],
    body_area: Optional[float] = None,
    shape: Optional[Sequence[int]] = None,
    body_box: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute geometric Damage Severity Index (DS-1 through DS-6).

    Parameters
    ----------
    boxes : sequence of (x1, y1, x2, y2)
        Damage bounding boxes in pixel coordinates.
    body_area : float or None, optional
        Vehicle body area in pixels. Ignored if ``body_box`` is provided and valid.
    shape : sequence of int or None, optional
        Image shape ``(H, W)`` or ``(H, W, C)`` used for clipping and mask sizing.
    body_box : (x1, y1, x2, y2) or None, optional
        Vehicle bounding box from detector. Preferred denominator source (DS-4).

    Returns
    -------
    dict
        ``dsi`` (float, 0.0-100.0 rounded to 1 decimal), ``damage_area`` (int),
        ``body_area`` (int), ``body_box`` (tuple | None), ``used_fallback`` (bool).
    """
    box_list = list(boxes or [])
    eff_shape = _normalize_shape(shape) if shape is not None else _infer_shape(box_list, body_box)
    img_h, img_w = eff_shape
    image_area = img_h * img_w

    clipped_body_box: Optional[Box] = None
    used_fallback = False

    if body_box is not None:
        clipped_body_box = clip_box(body_box, eff_shape)

    if clipped_body_box is not None:
        eff_body_area = (clipped_body_box[2] - clipped_body_box[0]) * (
            clipped_body_box[3] - clipped_body_box[1]
        )
    elif body_area is not None and body_area > 0:
        eff_body_area = int(round(body_area))
    else:
        eff_body_area = image_area
        used_fallback = True

    if not box_list:
        return {
            "dsi": 0.0,
            "damage_area": 0,
            "body_area": int(eff_body_area),
            "body_box": clipped_body_box,
            "used_fallback": used_fallback,
        }

    damage_px = union_area(box_list, eff_shape)
    if eff_body_area <= 0 or damage_px <= 0:
        dsi_val = 0.0
    else:
        raw = (damage_px / float(eff_body_area)) * 100.0
        dsi_val = round(min(100.0, max(0.0, raw)), 1)

    return {
        "dsi": dsi_val,
        "damage_area": int(damage_px),
        "body_area": int(eff_body_area),
        "body_box": clipped_body_box,
        "used_fallback": used_fallback,
    }
