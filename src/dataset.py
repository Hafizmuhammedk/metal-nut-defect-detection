"""Inspect the original files, split once, and write YOLO annotations."""

import hashlib
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import structlog
import yaml
from sklearn.model_selection import train_test_split

from src.annotations import boxes_to_yolo, mask_to_boxes, read_mask

log = structlog.get_logger()
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_dataset(root, min_area=9):
    """Read every image and its exact category/stem mask before splitting."""
    root = Path(root).resolve()
    for directory in ("train", "test", "ground_truth"):
        if not (root / directory).is_dir():
            raise FileNotFoundError(
                f"Missing {root / directory}. Extract the official metal_nut "
                "category; see data/README.md."
            )
    rows, used_masks = [], set()
    for original_split in ("train", "test"):
        files = sorted(
            p
            for p in (root / original_split).rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in files:
            relative = path.relative_to(root)
            if len(relative.parts) != 3:
                raise ValueError(f"Unexpected image layout: {relative}")
            category = path.parent.name
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable image: {path}")
            height, width = image.shape[:2]
            mask_path = None
            boxes = np.empty((0, 4), dtype=np.float32)
            if category != "good":
                mask_path = root / "ground_truth" / category / f"{path.stem}_mask.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(f"No matching mask: {mask_path}")
                mask = read_mask(mask_path)
                if mask.shape != (height, width):
                    raise ValueError(f"Image and mask are misaligned: {mask_path}")
                boxes = mask_to_boxes(mask, min_area)
                if not len(boxes):
                    raise ValueError(f"Defective image has no retained regions: {path}")
                used_masks.add(mask_path.resolve())
            rows.append(
                {
                    "image_id": "__".join(relative.with_suffix("").parts),
                    "image_path": relative.as_posix(),
                    "mask_path": mask_path.relative_to(root).as_posix()
                    if mask_path
                    else "",
                    "original_split": original_split,
                    "category": category,
                    "is_defective": category != "good",
                    "width": width,
                    "height": height,
                    "sha256": file_hash(path),
                    "mask_sha256": file_hash(mask_path) if mask_path else "",
                    "boxes": boxes.tolist(),
                }
            )
    if not rows:
        raise ValueError(f"No images found in {root}")
    unused = {
        p.resolve()
        for p in (root / "ground_truth").rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    } - used_masks
    if unused:
        raise ValueError(
            f"Unmatched ground-truth masks: {sorted(map(str, unused))[:5]}"
        )
    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if frame.image_id.duplicated().any() or frame.sha256.duplicated().any():
        raise ValueError(
            "Duplicate image IDs or identical files; resolve before splitting"
        )
    log.info(
        "dataset_inspected",
        images=len(frame),
        categories=frame.category.value_counts().to_dict(),
    )
    return frame


def _strata(frame, test_size):
    test_count = math.ceil(len(frame) * test_size)
    for column in ("category", "is_defective"):
        counts = frame[column].value_counts()
        smallest_split = min(test_count, len(frame) - test_count)
        if counts.min() >= 2 and smallest_split >= len(counts):
            if column != "category":
                log.warning("stratification_fallback", column=column)
            return frame[column]
    log.warning("stratification_unavailable", images=len(frame))
    return None


def split_dataset(frame, test_ratio=0.20, val_ratio=0.125, seed=42):
    """80% development, 20% test; validation is 12.5% of development (~10% total)."""
    if not 0 < test_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("Test and validation ratios must be between 0 and 1")
    if (
        frame.empty
        or frame.image_id.duplicated().any()
        or frame.sha256.duplicated().any()
    ):
        raise ValueError("Split requires a nonempty dataset with unique images")
    frame = frame.reset_index(drop=True)
    development, test = train_test_split(
        frame,
        test_size=test_ratio,
        random_state=seed,
        stratify=_strata(frame, test_ratio),
    )
    train, validation = train_test_split(
        development,
        test_size=val_ratio,
        random_state=seed,
        stratify=_strata(development, val_ratio),
    )
    result = frame.copy()
    result["split"] = ""
    for name, part in (("train", train), ("val", validation), ("test", test)):
        result.loc[part.index, "split"] = name
    if result.groupby("sha256").split.nunique().max() != 1:
        raise ValueError("An image appears in more than one split")
    return result


def split_summary(manifest):
    rows = []
    for label, parts in (
        ("Training/development (80%)", ["train", "val"]),
        ("Testing (20%)", ["test"]),
        ("Training only (within development)", ["train"]),
        ("Validation (within development)", ["val"]),
    ):
        part = manifest[manifest.split.isin(parts)]
        defects = int(part.is_defective.sum())
        rows.append(
            {
                "Dataset": label,
                "Images": len(part),
                "Good": len(part) - defects,
                "Defective": defects,
            }
        )
    return pd.DataFrame(rows)


def prepare_yolo(manifest, source_root, output_root, yaml_path, min_area=9):
    """Use a content-specific output directory so stale files cannot leak in."""
    if manifest.empty or set(manifest.split) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain train, val and test images")
    if manifest.image_id.duplicated().any() or manifest.sha256.duplicated().any():
        raise ValueError("Manifest contains duplicate images")
    identity = manifest.assign(boxes=manifest.boxes.map(json.dumps)).to_csv(index=False)
    identity += str(min_area)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    destination = Path(output_root) / digest
    source_root = Path(source_root).resolve()
    files = []
    expected = set()
    for row in manifest.itertuples():
        if Path(row.image_id).name != row.image_id or row.image_id in {".", ".."}:
            raise ValueError(f"Invalid image ID: {row.image_id}")
        source = (source_root / row.image_path).resolve()
        if not source.is_relative_to(source_root):
            raise ValueError(
                f"Image path leaves the dataset directory: {row.image_path}"
            )
        image_name = f"{row.image_id}{source.suffix.lower()}"
        image_path = destination / "images" / row.split / image_name
        label_path = destination / "labels" / row.split / f"{row.image_id}.txt"
        if file_hash(source) != row.sha256:
            raise ValueError(f"Image changed since inspection: {source}")
        if row.mask_path and file_hash(source_root / row.mask_path) != row.mask_sha256:
            raise ValueError(f"Mask changed since inspection: {row.mask_path}")
        labels = boxes_to_yolo(row.boxes, row.width, row.height)
        if row.is_defective != bool(len(labels)):
            raise ValueError(f"Defect status and boxes disagree: {row.image_id}")
        files.append((source, image_path, label_path, labels))
        expected.update([image_path.resolve(), label_path.resolve()])

    existing = {
        path.resolve()
        for path in destination.rglob("*")
        if path.is_file() and path.suffix != ".cache"
    }
    unexpected = existing - expected
    if unexpected:
        raise ValueError(f"Unexpected files in prepared dataset: {unexpected}")

    # Check the whole manifest before changing an existing prepared dataset.
    for source, image_path, label_path, labels in files:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_path)
        lines = ["0 " + " ".join(f"{value:.8f}" for value in box) for box in labels]
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
    actual = {
        p.resolve()
        for p in destination.rglob("*")
        if p.is_file() and p.suffix != ".cache"
    }
    if actual != expected:
        raise ValueError(f"Unexpected files in prepared dataset: {actual - expected}")
    config = {
        "path": destination.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "defect"},
    }
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    Path(yaml_path).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    log.info("yolo_dataset_written", directory=str(destination), images=len(manifest))
    return destination
