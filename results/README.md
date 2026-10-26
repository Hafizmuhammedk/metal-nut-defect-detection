# Results status

**Real-data training, testing, export, and benchmarking: completed.**
Both models trained for 50 epochs on the 234-image training subset. Validation
uses 34 images; the held-out test set contains 67. The selected augmentation
model has precision 0.9167, recall 0.4583 and mAP@0.5 0.6516. It misses 13 of
24 labeled regions at confidence 0.25. These limits remain visible in the report.

Open [final_results.md](final_results.md) for actual numbers, benchmark scopes
and five genuine failures from 10 failing test images. The notebook with all
outputs is `notebooks/metal_nut_detection.executed.ipynb`.
`verification.json` separates synthetic correctness checks from real-data results.

Running the notebook with the official dataset creates:

- `dataset_inventory.csv`, `split_manifest.csv`, `split_summary.csv`
- `experiment_config.json`, `environment.json`, `environment-lock.txt`
- `selection.json` (frozen before opening test results)
- `runs/baseline/` and `runs/augmented/` (weights, arguments, logs, plots)
- `training_curves/`, `predictions/`, `failure_cases/`
- `metrics.csv`, per-image test CSVs and prediction JSONs
- `parity.csv`, `cpu_benchmark.csv`, `final_results.md`
- `run_status.json` (complete only when the final section finishes)

The split manifest includes original category, good/defective status, relative
image/mask paths, SHA-256 hashes, original dimensions, boxes and assigned split.
Preserve the results folder alongside any submitted executed notebook.
