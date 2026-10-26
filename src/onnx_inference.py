"""FP32, batch-one, single-class YOLOv8 ONNX preprocessing and decoding."""

import cv2
import numpy as np


def preprocess(image, image_size=640):
    """Letterbox an RGB uint8 image with the same padding convention as YOLO."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image with three channels")
    if not isinstance(image_size, (int, np.integer)) or image_size < 1:
        raise ValueError("Image size must be a positive integer")
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Image cannot be empty")
    ratio = min(image_size / height, image_size / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    if resized_width == 0 or resized_height == 0:
        raise ValueError("Image aspect ratio is too extreme for the requested size")
    left = round((image_size - resized_width) / 2 - 0.1)
    top = round((image_size - resized_height) / 2 - 0.1)
    resized = cv2.resize(image, (resized_width, resized_height))
    padded = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    padded[top : top + resized_height, left : left + resized_width] = resized
    tensor = padded.transpose(2, 0, 1)[None]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255
    return tensor, (ratio, left, top, height, width)


def decode(output, transform, confidence=0.25, nms_iou=0.7, max_det=300):
    """YOLOv8 raw output is [1, 4+classes, anchors]; no objectness channel."""
    if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("Confidence and NMS IoU must be between 0 and 1")
    if not isinstance(max_det, (int, np.integer)) or max_det < 1:
        raise ValueError("Maximum detections must be a positive integer")
    ratio, left, top, height, width = transform
    if not np.isfinite(transform).all() or min(ratio, height, width) <= 0:
        raise ValueError("Invalid letterbox scale or image dimensions")
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 5:
        raise ValueError(f"Expected one-class YOLOv8 [1,5,N], got {output.shape}")
    candidates = output[0].T
    valid = np.isfinite(candidates).all(axis=1)
    valid &= (candidates[:, 4] > confidence) & (candidates[:, 4] <= 1)
    valid &= (candidates[:, 2:4] > 0).all(axis=1)
    candidates = candidates[valid]
    if not len(candidates):
        return np.empty((0, 5), dtype=np.float32)
    boxes = np.column_stack(
        (
            candidates[:, :2] - candidates[:, 2:4] / 2,
            candidates[:, :2] + candidates[:, 2:4] / 2,
        )
    )
    scores = candidates[:, 4]
    xywh = np.column_stack((boxes[:, :2], boxes[:, 2:] - boxes[:, :2]))
    # Run NMS before clipping: clipping first can change pairwise overlap.
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), confidence, nms_iou)
    keep = np.asarray(keep).reshape(-1).astype(int)[:max_det]
    boxes, scores = boxes[keep], scores[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    visible = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
    return np.column_stack((boxes[visible], scores[visible])).astype(np.float32)


def predict_onnx(
    session, image, image_size=640, confidence=0.25, nms_iou=0.7, max_det=300
):
    """Run the exported single-class detector on one RGB image."""
    tensor, transform = preprocess(image, image_size)
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return decode(output, transform, confidence, nms_iou, max_det)
