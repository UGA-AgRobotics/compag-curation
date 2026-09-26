"""Import-light geometry shared by canonical inference and point evaluation."""

from __future__ import annotations

from typing import Any


def bbox_iou(left: Any, right: Any) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    intersection = width * height
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def nms_xyxy(boxes: Any, scores: Any, iou_threshold: float) -> list[int]:
    """Run the source-backed deterministic greedy full-image NMS."""

    box_rows = list(boxes)
    score_rows = [float(value) for value in scores]
    if not box_rows:
        return []
    if len(box_rows) != len(score_rows):
        raise ValueError("NMS box and score counts differ")
    order = sorted(range(len(score_rows)), key=lambda index: (-score_rows[index], index))
    keep: list[int] = []
    while order:
        index = order[0]
        keep.append(index)
        if len(order) == 1:
            break
        order = [
            other
            for other in order[1:]
            if bbox_iou(box_rows[index], box_rows[other]) <= iou_threshold
        ]
    return keep


__all__ = ["bbox_iou", "nms_xyxy"]
