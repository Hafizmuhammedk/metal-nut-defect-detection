"""Execute the real notebook, keeping a separate executed copy after each cell."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "notebooks" / "metal_nut_detection.ipynb"
    output = source.with_name("metal_nut_detection.executed.ipynb")
    notebook = nbformat.read(source, as_version=4)

    def on_cell_start(cell, cell_index, **kwargs):
        if cell.cell_type == "markdown":
            print(cell.source.splitlines()[0], flush=True)

    def on_cell_executed(cell, cell_index, **kwargs):
        nbformat.write(notebook, output)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="metal-nut",
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_start=on_cell_start,
        on_cell_executed=on_cell_executed,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
