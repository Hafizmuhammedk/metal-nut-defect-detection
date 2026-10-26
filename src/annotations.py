"""Convert MVTec binary masks to single-class detection labels."""

import cv2
import numpy as np


def read_mask(path):
    """Decode directly: some YOLO versions patch cv2.imread to return HxWx1."""
    contents = np.fromfile(path, dtype=np.uint8)
    if not contents.size:
        raise ValueError(f"Empty mask file: {path}")
    mask = cv2.imdecode(contents, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2:
        raise ValueError(f"Unreadable grayscale mask: {path}")
    return mask


def mask_to_boxes(mask, min_area=9):
    """Return an exclusive xyxy box for each retained 8-connected region.

    The filter uses foreground pixel area, not bounding rectangle area. Nine
    pixels is a fixed annotation cleanup rule; do not tune it on the test set.
    """
    if mask.ndim != 2 or not np.isfinite(min_area) or min_area < 1:
        raise ValueError("Expected a grayscale mask and a positive minimum area")
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    boxes = []
    for x, y, width, height, area in stats[1:]:
        if area >= min_area:
            boxes.append([int(x), int(y), int(x + width), int(y + height)])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def boxes_to_yolo(boxes, width, height):
    """Convert pixel xyxy boxes into normalized center-x, center-y, w, h."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if len(boxes) and (
        not np.isfinite(boxes).all()
        or (boxes[:, :2] < 0).any()
        or (boxes[:, 2] > width).any()
        or (boxes[:, 3] > height).any()
        or (boxes[:, 2:] <= boxes[:, :2]).any()
    ):
        raise ValueError("Invalid or out-of-bounds xyxy box")
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    return np.column_stack((centers, sizes)) / [width, height, width, height]


def yolo_to_boxes(labels, width, height):
    """Convert normalized center/size labels back to pixel xyxy boxes."""
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 4)
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if (
        not np.isfinite(labels).all()
        or (labels < 0).any()
        or (labels > 1).any()
        or (labels[:, 2:] <= 0).any()
    ):
        raise ValueError("YOLO labels need normalized coordinates and positive sizes")
    corners = np.column_stack(
        (labels[:, :2] - labels[:, 2:] / 2, labels[:, :2] + labels[:, 2:] / 2)
    )
    # Text labels are rounded, so allow a small error at the image boundary.
    if (corners < -1e-6).any() or (corners > 1 + 1e-6).any():
        raise ValueError("YOLO box extends beyond the image")
    pixels = labels * [width, height, width, height]
    return np.column_stack(
        (pixels[:, :2] - pixels[:, 2:] / 2, pixels[:, :2] + pixels[:, 2:] / 2)
    ).astype(np.float32)
