"""Confidence-ordered one-to-one IoU matching at a fixed operating point."""

import numpy as np


def box_iou(boxes_a, boxes_b):
    """Return the pairwise IoU matrix for two sets of pixel xyxy boxes."""
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(
        area_a[:, None] + area_b[None, :] - intersection, 1e-9
    )


def match_detections(truth, predictions, iou_threshold=0.5):
    """Predictions are Nx5 xyxy+score. Each truth box can match only once."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("Matching IoU must be greater than 0 and at most 1")
    truth = np.asarray(truth, dtype=np.float32).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 5)
    for boxes in (truth, predictions[:, :4]):
        if not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError("Detection boxes must be finite with positive sizes")
    scores = predictions[:, 4]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction scores must be between 0 and 1")

    overlaps = box_iou(predictions[:, :4], truth)
    matched_truth = set()
    matched_predictions = set()
    for prediction_index in np.argsort(-scores, kind="stable"):
        available = [index for index in range(len(truth)) if index not in matched_truth]
        if not available:
            break
        best_truth = max(available, key=lambda index: overlaps[prediction_index, index])
        if overlaps[prediction_index, best_truth] >= iou_threshold:
            matched_truth.add(best_truth)
            matched_predictions.add(int(prediction_index))

    false_predictions = sorted(set(range(len(predictions))) - matched_predictions)
    missed_truth = sorted(set(range(len(truth))) - matched_truth)
    return {
        "tp": len(matched_truth),
        "fp": len(false_predictions),
        "fn": len(missed_truth),
        "false_predictions": false_predictions,
        "missed_truth": missed_truth,
    }
