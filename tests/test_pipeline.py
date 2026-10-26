"""Synthetic correctness checks; these are not model performance results."""

import json

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest
import yaml
from onnx import TensorProto, helper

from src.annotations import boxes_to_yolo, mask_to_boxes, read_mask, yolo_to_boxes
from src.dataset import inspect_dataset, prepare_yolo, split_dataset
from src.evaluation import box_iou, match_detections
from src.onnx_inference import decode, predict_onnx, preprocess


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "metal_nut"
    rng = np.random.default_rng(42)
    for category in ("good", "bent", "color", "flip", "scratch"):
        for index in range(20):
            split = "train" if category == "good" and index < 15 else "test"
            image_path = root / split / category / f"{index:03}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(image_path), rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
            )
            if category != "good":
                path = root / "ground_truth" / category / f"{index:03}_mask.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((40, 60), dtype=np.uint8)
                mask[4:10, 8:15] = 255
                cv2.imwrite(str(path), mask)
    return root


def test_mask_components_and_coordinate_roundtrip():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[0:3, 0:3] = 255
    mask[15:20, 25:30] = 255
    mask[8, 10] = 255
    boxes = mask_to_boxes(mask)
    np.testing.assert_array_equal(boxes, [[0, 0, 3, 3], [25, 15, 30, 20]])
    labels = boxes_to_yolo(boxes, 30, 20)
    np.testing.assert_allclose(yolo_to_boxes(labels, 30, 20), boxes, atol=1e-5)
    assert mask_to_boxes(np.zeros_like(mask)).shape == (0, 4)
    with pytest.raises(ValueError):
        boxes_to_yolo([[0, 0, 31, 1]], 30, 20)


def test_split_and_written_annotations(dataset, tmp_path):
    frame = inspect_dataset(dataset)
    manifest = split_dataset(frame)
    pd.testing.assert_frame_equal(manifest, split_dataset(frame))
    assert manifest.split.value_counts().to_dict() == {
        "train": 70,
        "test": 20,
        "val": 10,
    }
    for part in ("train", "val", "test"):
        assert set(manifest[manifest.split == part].category) == set(frame.category)
    yaml_path = tmp_path / "data.yaml"
    prepared = prepare_yolo(manifest, dataset, tmp_path / "prepared", yaml_path)
    assert yaml.safe_load(yaml_path.read_text())["path"] == prepared.as_posix()
    for row in manifest.itertuples():
        path = prepared / "labels" / row.split / f"{row.image_id}.txt"
        text = path.read_text()
        if row.category == "good":
            assert text == ""
        else:
            values = np.loadtxt(path, ndmin=2)
            assert (values[:, 0] == 0).all()
            np.testing.assert_allclose(
                yolo_to_boxes(values[:, 1:], row.width, row.height),
                row.boxes,
                atol=1e-5,
            )
    manifest.drop(columns="boxes").to_csv(tmp_path / "manifest.csv", index=False)
    (prepared / "images" / "test" / "stale.png").write_bytes(b"stale")
    with pytest.raises(ValueError, match="Unexpected files"):
        prepare_yolo(manifest, dataset, tmp_path / "prepared", yaml_path)


def test_missing_misaligned_and_orphan_masks(dataset):
    path = dataset / "ground_truth" / "bent" / "000_mask.png"
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(FileNotFoundError, match="matching mask"):
        inspect_dataset(dataset)
    cv2.imwrite(str(path), np.ones((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="misaligned"):
        inspect_dataset(dataset)
    path.write_bytes(original)
    (path.parent / "orphan_mask.png").write_bytes(original)
    with pytest.raises(ValueError, match="Unmatched"):
        inspect_dataset(dataset)


def test_duplicate_source_images_are_rejected(dataset):
    files = sorted((dataset / "train" / "good").glob("*.png"))
    files[1].write_bytes(files[0].read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        inspect_dataset(dataset)


def test_mask_reader_after_yolo_patches_opencv(dataset):
    from ultralytics import YOLO

    assert YOLO is not None
    frame = inspect_dataset(dataset)
    assert len(frame) == 100
    assert len(frame[frame.is_defective].iloc[0].boxes) == 1


def test_one_to_one_matching_empty_and_duplicate_predictions():
    truth = [[0, 0, 10, 10], [20, 20, 30, 30]]
    predictions = [[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8], [40, 40, 50, 50, 0.7]]
    result = match_detections(truth, predictions)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
    assert match_detections([], predictions)["fp"] == 3
    assert match_detections(truth, [])["fn"] == 2
    assert match_detections([], [])["tp"] == 0
    np.testing.assert_allclose(box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]]), [[1 / 3]])


def test_matching_uses_best_unmatched_truth():
    truth = [[0, 0, 10, 10], [2, 0, 12, 10]]
    predictions = [[0, 0, 10, 10, 0.9], [1, 0, 11, 10, 0.8]]
    assert match_detections(truth, predictions)["tp"] == 2


def test_letterbox_decode_nms_and_runtime(tmp_path):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, transform = preprocess(image, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert transform == (3.2, 0, 160, 100, 200)
    # Two overlapping boxes plus one low-confidence box in letterboxed coordinates.
    raw = np.array(
        [
            [
                [320, 321, 20],
                [320, 320, 20],
                [320, 320, 10],
                [160, 160, 10],
                [0.9, 0.8, 0.1],
            ]
        ],
        dtype=np.float32,
    )
    result = decode(raw, transform)
    np.testing.assert_allclose(result, [[50, 25, 150, 75, 0.9]], atol=1e-5)
    assert decode(raw, transform, confidence=0.99).shape == (0, 5)
    with pytest.raises(ValueError, match="one-class"):
        decode(np.zeros((1, 84, 8400)), transform)
    node = helper.make_node(
        "Constant",
        [],
        ["output"],
        value=helper.make_tensor(
            "values", TensorProto.FLOAT, raw.shape, raw.flatten().tolist()
        ),
    )
    graph = helper.make_graph(
        [node],
        "synthetic-decoder-test",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(raw.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "synthetic.onnx"
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(predict_onnx(session, image), result)


def test_notebook_has_complete_unexecuted_sections():
    from pathlib import Path

    notebook = json.loads(
        Path("notebooks/metal_nut_detection.ipynb").read_text(encoding="utf-8")
    )
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in code:
        compile("".join(cell["source"]), "notebook-cell", "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
    headings = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ]
    assert len(headings) == 16


@pytest.mark.parametrize(
    "label",
    [
        [0.5, 0.5, 0, 0.2],
        [0.5, 0.5, -0.1, 0.2],
        [float("nan"), 0.5, 0.2, 0.2],
        [0.99, 0.5, 0.2, 0.2],
    ],
)
def test_invalid_yolo_annotations_are_rejected(label):
    with pytest.raises(ValueError):
        yolo_to_boxes([label], 100, 100)


def test_empty_mask_has_a_clear_error(tmp_path):
    path = tmp_path / "empty_mask.png"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Empty mask"):
        read_mask(path)


def test_split_does_not_depend_on_dataframe_index(dataset):
    frame = inspect_dataset(dataset)
    expected = split_dataset(frame)
    frame.index = [0] * len(frame)
    pd.testing.assert_frame_equal(split_dataset(frame), expected)
    with pytest.raises(ValueError, match="unique images"):
        split_dataset(pd.concat([frame, frame]))


def test_preparation_checks_files_before_overwriting(dataset, tmp_path):
    manifest = split_dataset(inspect_dataset(dataset))
    config = tmp_path / "data.yaml"
    prepared = prepare_yolo(manifest, dataset, tmp_path / "prepared", config)
    label = next((prepared / "labels" / "train").glob("*.txt"))
    label.write_text("review marker", encoding="utf-8")
    (prepared / "images" / "test" / "unexpected.png").write_bytes(b"stale")
    with pytest.raises(ValueError, match="Unexpected files"):
        prepare_yolo(manifest, dataset, tmp_path / "prepared", config)
    assert label.read_text(encoding="utf-8") == "review marker"


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, float("nan")])
def test_invalid_matching_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="Matching IoU"):
        match_detections([], [], threshold)


def test_invalid_onnx_inputs_and_clipped_boxes():
    with pytest.raises(ValueError, match="RGB uint8"):
        preprocess(np.zeros((20, 20), dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB uint8"):
        preprocess(np.zeros((20, 20, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="empty"):
        preprocess(np.zeros((0, 20, 3), dtype=np.uint8))
    transform = (1, 0, 0, 100, 100)
    output = np.array([[[200], [200], [20], [20], [0.9]]], dtype=np.float32)
    assert decode(output, transform).shape == (0, 5)
    with pytest.raises(ValueError, match="Maximum detections"):
        decode(output, transform, max_det=-1)
    with pytest.raises(ValueError, match="Confidence"):
        decode(output, transform, confidence=1.1)
