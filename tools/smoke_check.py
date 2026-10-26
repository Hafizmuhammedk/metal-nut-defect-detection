"""Execute the whole notebook on disposable synthetic data, never real metrics."""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import nbformat
import numpy as np
from nbclient import NotebookClient
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="metal-nut-synthetic-check-") as directory:
        project = Path(directory)
        shutil.copytree(
            ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__")
        )
        (project / "models").mkdir()
        rng = np.random.default_rng(42)
        dataset_root = project / "data" / "metal_nut"
        for category in ("good", "bent", "color", "flip", "scratch"):
            for index in range(10):
                split = "train" if category == "good" and index < 7 else "test"
                image_path = dataset_root / split / category / f"{index:03}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = rng.integers(30, 220, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(image_path), image)
                if category != "good":
                    mask_path = (
                        dataset_root
                        / "ground_truth"
                        / category
                        / f"{index:03}_mask.png"
                    )
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    mask = np.zeros((64, 64), dtype=np.uint8)
                    mask[27:33, 28:36] = 255
                    cv2.imwrite(str(mask_path), mask)
        # Random initialization avoids downloading or representing a trained detector.
        YOLO("yolov8n.yaml").save(project / "models" / "yolov8n.pt")
        notebook = nbformat.read(
            ROOT / "notebooks" / "metal_nut_detection.ipynb", as_version=4
        )
        nbformat.validate(notebook)
        config_cell = next(
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and "EPOCHS = 50" in cell.source
        )
        for old, new in (
            ("EPOCHS = 50", "EPOCHS = 1"),
            ("IMAGE_SIZE = 640", "IMAGE_SIZE = 64"),
            ("WARMUP_RUNS = 10", "WARMUP_RUNS = 1"),
            ("MEASURED_RUNS = 100", "MEASURED_RUNS = 3"),
            ('DEVICE = 0 if torch.cuda.is_available() else "cpu"', 'DEVICE = "cpu"'),
            ("CPU_THREADS = min(4, os.cpu_count() or 1)", "CPU_THREADS = 2"),
        ):
            if old not in config_cell.source:
                raise ValueError(
                    f"Smoke-check setting no longer matches the notebook: {old}"
                )
            config_cell.source = config_cell.source.replace(old, new)
        # Suppress inline rendering in the disposable execution; savefig still runs.
        imports = next(
            cell
            for cell in notebook.cells
            if cell.cell_type == "code"
            and "from ultralytics import YOLO" in cell.source
        )
        imports.source = "import matplotlib\nmatplotlib.use('Agg')\n" + imports.source
        imports.source += (
            f"\nif Path(sys.executable).resolve() != Path({sys.executable!r}).resolve():\n"
            "    raise RuntimeError('Select the project virtual environment for the metal-nut kernel')\n"
        )

        def on_cell_start(cell, cell_index, **kwargs):
            if cell.cell_type == "markdown":
                print(cell.source.splitlines()[0], flush=True)

        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="metal-nut",
            resources={"metadata": {"path": str(project)}},
            on_cell_start=on_cell_start,
        )
        try:
            client.execute()
        except Exception:
            # Save diagnostics without presenting synthetic metrics as assignment results.
            diagnostic = ROOT / ".smoke-failure.ipynb"
            nbformat.write(notebook, diagnostic)
            print(f"Synthetic execution failed; diagnostics: {diagnostic}", flush=True)
            raise
        status = json.loads((project / "results" / "run_status.json").read_text())
        assert status["status"] == "complete"
        assert status["dataset_images"] == 50
        assert status["test_images"] == 10
        assert (project / "models" / "metal_nut_detector.onnx").is_file()
        for name in (
            "metrics.csv",
            "parity.csv",
            "cpu_benchmark.csv",
            "final_results.md",
        ):
            assert (project / "results" / name).is_file(), name
        print(
            "All 16 sections executed on synthetic data. Temporary artifacts discarded."
        )
    result = {
        "check": "full_notebook_synthetic_execution",
        "status": "passed",
        "sections": 16,
        "fixture_images": 50,
        "fixture_test_images": 10,
        "epochs_per_experiment": 1,
        "input_size": 64,
        "initialization": "random YOLOv8n, not pretrained",
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "scope": "synthetic execution only; real-data experiments are separate",
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "smoke-check.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
