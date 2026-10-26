"""Image loading, drawing, and timing shared by notebook sections."""

import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_boxes(axis, image, boxes, color, scores=None):
    """Draw pixel xyxy boxes and optional confidence scores on an image."""
    axis.imshow(image)
    boxes = np.asarray(boxes).reshape(-1, 4)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        axis.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5
            )
        )
        if scores is not None:
            axis.text(
                x1,
                max(0, y1 - 3),
                f"{scores[index]:.2f}",
                color=color,
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.5},
            )
    axis.axis("off")


def save_figure(figure, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")


def measure_latency(function, warmup=10, runs=100):
    """Time a synchronous callable; loading and disk I/O belong outside it."""
    if warmup < 0 or runs < 1:
        raise ValueError("Invalid benchmark repetition counts")
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    values = np.asarray(samples)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps": float(1000 / values.mean()),
        "warmup": warmup,
        "runs": runs,
    }


def save_json(data, path):
    """Write a readable JSON result using the same encoding on every platform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
