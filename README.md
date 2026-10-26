# Metal nut defect detection

A straightforward YOLOv8n project for the InSiSo Technologies computer vision
internship assignment. Binary MVTec AD masks become bounding boxes for one
detection class, `defect`. Original categories remain available for analysis.
The notebook covers preparation, controlled training, held-out evaluation,
failure analysis, ONNX inference, and CPU benchmarking.

**Status: complete; both real-data experiments executed for 50 epochs.** The
user supplied a downloaded MVTec AD `metal_nut` mirror. All 335 images and 93
masks passed inspection and were copied into `data/metal_nut`. The machine has
an NVIDIA RTX 2050 (4 GiB); CUDA-enabled PyTorch 2.7.1 is installed in `.venv`.
The executed notebook, trained checkpoints, actual test results, five genuine
failure visualizations, validated ONNX detector and CPU benchmarks are saved.
The selected model's recall is **45.8% at confidence 0.25**: it still misses
defects and needs further development before customer deployment. Engineering
verification is reported separately from detector performance.

## Files

```text
notebooks/metal_nut_detection.ipynb  Complete experiment, 16 numbered sections
notebooks/metal_nut_detection.executed.ipynb  Executed workflow with actual outputs
src/annotations.py                Mask components and coordinate conversion
src/dataset.py                    Inventory, stratified splits, YOLO preparation
src/evaluation.py                 One-to-one detection matching
src/onnx_inference.py             Letterboxing, raw YOLOv8 decoding, NMS
src/utils.py                      Drawing and repeated latency measurement
configs/data.yaml                 Template; regenerated for the actual dataset
data/README.md                    Download and extraction instructions
models/README.md                  Expected trained artifacts
results/README.md                 Generated output inventory and status
tests/test_pipeline.py            Synthetic correctness checks
tools/smoke_check.py              Isolated synthetic execution of all notebook sections
tools/run_notebook.py             Execute real workflow and save results after each cell
```

## Setup in VS Code

Use Python 3.10 or later (the local verification environment uses Python 3.11).
Open this folder in VS Code, install the Python and Jupyter extensions, and run
these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name metal-nut --display-name "Python (metal-nut)"
```

Select `.venv` as interpreter and **Python (metal-nut)** as notebook kernel.
You can also run `.\.venv\Scripts\python.exe -m jupyterlab`.
For CUDA, install a matching PyTorch/torchvision build using the
[official selector](https://pytorch.org/get-started/locally/) before installing
the remaining requirements. Do not assume a CUDA device exists.

This machine's CUDA setup used the official PyTorch CUDA 12.8 wheels:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

The working `.venv` inherits existing packages on this computer and overrides
PyTorch/torchvision locally. The setup commands above create a clean environment
for reproduction; its exact full training run has not been independently checked.

Dependencies have bounded versions for installation across supported Python
versions. The notebook records exact package versions, host information, training
arguments, initial weight SHA-256, and split manifest for the executed run.
`results/verified-environment.txt` describes this delivery's engineering-check
environment; it is not a claim of fresh-environment training verification.

## Dataset and annotations

Download **metal_nut** from the original
[MVTec AD dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
Follow [data/README.md](data/README.md) and extract into `data/metal_nut/`, or
change `DATASET_DIR` in notebook section 3. There is no automatic redistribution
or download of the data. MVTec AD uses **CC BY-NC-SA 4.0**, including a restriction
on commercial use. Check the dataset's terms before using it for a customer.
Ultralytics also has its own [software licensing terms](https://www.ultralytics.com/license).

The inspector reads the actual source tree before preparing anything. A defective
image must match its category and stem-specific mask. Mismatched dimensions,
missing/orphan masks, unreadable images, and identical image files raise errors.
The mask-to-box function finds 8-connected foreground regions and keeps those
with at least **9 foreground pixels** at source resolution. This fixed cleanup
rule can remove real tiny anomalies, so inspect the section 5 plots. Do not tune
it on test results. Good images receive truly empty `.txt` label files.

Coordinates use exclusive pixel edges, then normalized YOLO center/size format.
Saved labels are loaded and checked against the source boxes in section 7.
Content-specific prepared directories and an exact file inventory guard against
stale image/label files. Files are copied, leaving the original dataset intact.

## Split and reproducibility

The required split is a **custom 80% development / 20% test split**, created with
`train_test_split(test_size=0.20, random_state=42)` on original images **before
augmentation**. The test count is `ceil(N * 0.20)` because images are indivisible.
Stratification uses category (`good`, `bent`, `color`, `flip`, `scratch`) where
counts permit, then falls back to good/defective status with a logged warning.

Validation uses 12.5% of development, approximately 10% of all images. The
remaining approximately 70% trains the detector. Both good and defective
examples are included in the populations. A count table and category crosstab
are calculated from actual files, never hard-coded.

MVTec's original training directory mainly contains normal examples. We pool its
original train/test populations to construct this supervised split; the original
test anomalies can therefore enter our development portion. **These results
must not be reported as the official MVTec anomaly-detection benchmark.**

`results/split_manifest.csv` preserves paths, provenance, categories, hashes,
dimensions, boxes, and assignments. Rerunning with changed source files or split
settings refuses to replace an existing incompatible manifest. Archive the whole
results folder before intentionally beginning a different dataset experiment.
File hashes catch identical copies, not near-duplicates, shared lots or temporal
correlation; use grouped holdouts for future factory data.

Verified local split (calculated from the supplied files):

| Dataset | Images | Good | Defective |
| --- | --- | --- | --- |
| Development (80%) | 268 | 194 | 74 |
| Test (20%) | 67 | 48 | 19 |
| Training within development | 234 | 169 | 65 |
| Validation within development | 34 | 25 | 9 |

All four defect categories occur in training, validation and test. The source
mirror revision and integrity-check method are documented in
[data/README.md](data/README.md). All 428 image/mask PNGs matched the downloader's
cached SHA-256 etags; this is not an independent comparison to the original archive.

## Run the notebook

Open [metal_nut_detection.ipynb](notebooks/metal_nut_detection.ipynb) and execute
top to bottom. Section 4 stops with an actionable error if the real dataset is
missing. Training is intentionally not replaced with a fake demonstration.

For this completed run, open the
[executed notebook](notebooks/metal_nut_detection.executed.ipynb) or
[actual results report](results/final_results.md). To execute another exact
reproduction in the configured environment, run:

```powershell
.\.venv\Scripts\python.exe tools/run_notebook.py
```

This creates a new timestamped training directory and overwrites the executed
notebook and top-level result summaries. Archive the existing results first if
you want to retain them. Do not change settings based on this exposed test set.

Defaults: `yolov8n.pt`, 640 pixels, 50 epochs, seed 42, AdamW with initial learning
rate 0.001, nominal batch 16. CPU uses batch 4; CUDA with less than 6 GiB uses 8.
Device selection is automatic; Windows notebook workers default to 0. If memory
requires a smaller batch, choose it before running either experiment. First
loading the pretrained checkpoint requires internet access.

Section 9 trains **both** experiments and selects the model before section 10
opens held-out results. Every run starts from the same checkpoint and seed.
Training uses only `train`; best-epoch/model decisions use only `val`.
Checkpoint files, logs, arguments, and native loss/validation plots live under
`results/runs/<run timestamp>/`. The final selected checkpoint is copied to
`models/metal_nut_detector.pt`. Both models run the same 50-epoch budget.

The augmentation experiment changes only horizontal reflection probability from
0 to 0.5. Reflection retains every defect pixel, its color and the visible face
of the nut, including face-flip anomalies. Rotation was excluded because some
training defects touch image boundaries. Color jitter, mosaic, scaling and mixup
are disabled. Optional Albumentations defaults are also disabled. Review the
training reflection previews before proceeding. An augmentation run may perform
worse; the report uses its actual outcome.

For a headless run after registering the kernel:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/metal_nut_detection.ipynb --output metal_nut_detection.executed.ipynb --ExecutePreprocessor.timeout=-1
```

Preserve the executed notebook and all results together. Fixed seeds improve
repeatability but do not guarantee identical results across devices or library
versions. Use the saved environment lock and manifest to repeat a completed run.

To reproduce this run's core package versions, use Python 3.11, install the CUDA
wheels shown above, then run `python -m pip install -r requirements.txt` and
`python -m pip install -r results/environment-lock.txt`. Keep the saved split
manifest and original dataset unchanged. The recorded result is one automated
run, not a claim of independent reproducibility across hardware.

## Evaluation and current results

Every image in the held-out 20%, including good images, is evaluated. Standard
Ultralytics box AP uses a 0.001 confidence floor, IoU 0.5 and IoU 0.5:0.95.
Precision/recall and TP/FP/FN use a separate fixed operating point: confidence
0.25, matching IoU 0.5, and one-to-one matching in descending score order.
Each truth box can match only once; duplicate detections are false positives.
NMS IoU is 0.7, independent of matching IoU. Empty-label good images contribute
false positives when a defect is predicted.

All models use the same image size, square letterboxing, FP32, confidence floor,
NMS, max detections and matching policy. There is no test-time augmentation.
The test set never selects thresholds, epochs, architecture, or final model.
Library max-F1 precision/recall are deliberately not used as fixed-threshold P/R.

| Metric | Baseline | Augmented experiment |
| --- | --- | --- |
| Precision at confidence 0.25 | 0.8462 | 0.9167 |
| Recall at confidence 0.25 | 0.4583 | 0.4583 |
| mAP@0.5 | 0.6466 | 0.6516 |
| mAP@0.5:0.95 | 0.4791 | 0.4419 |
| TP / FP / FN | 11 / 2 / 13 | 11 / 1 / 13 |
| PyTorch CPU end-to-end mean, ms | 74.71 | 65.32 |

Both models were evaluated on all **67 held-out images**, containing 24
mask-derived regions across 19 defective images and 48 good images. Values are
fractions, not percentages. Exact values are in [metrics.csv](results/metrics.csv).
The **augmented** checkpoint was selected before test evaluation because its
validation mAP@0.5:0.95 was 0.6283, versus baseline 0.4302.

The augmentation result is mixed: precision improved and mAP@0.5 rose by only
0.0050, but recall stayed unchanged and mAP@0.5:0.95 fell by 0.0372. It is not a
general accuracy improvement. The final choice remains the validation selection;
we did not switch models after inspecting test scores. Both runs missed 13 of
24 regions at the fixed operating point. Further threshold or model development
must use validation data and a new independent final holdout. Different CPU
timings for identical architectures are observations from this run, not proof
that augmentation accelerates a model.

The selected model has **10 failing test images**, of which five are shown below.
Panels are original / ground truth (green) / prediction (red). The explanations
are visual-review hypotheses; they do not establish a physical root cause.

**Bent 010 — three missed regions.** The large damaged edge and two smaller
disconnected mask regions were all missed. Small regions and similarity to
normal reflective edges are plausible difficulties.

![Bent 010: three missed regions](results/failure_cases/01_test__bent__010.png)

**Color 021 — incorrect localization.** A 0.34-confidence prediction covers only
part of the long discoloration. It fails IoU 0.5 and contributes one FP and one
FN, rather than a correct detection of the entire region.

![Color 021: partially localized discoloration](results/failure_cases/02_test__color__021.png)

**Scratch 005 — two missed regions.** Fine scratches lie across the upper face
and near a notch, where normal machining lines create a similar texture.

![Scratch 005: two missed regions](results/failure_cases/03_test__scratch__005.png)

**Bent 006 — missed edge deformation.** The annotated lower-left contour damage
is relatively dark and resembles the background/normal outline.

![Bent 006: missed contour defect](results/failure_cases/04_test__bent__006.png)

**Bent 007 — missed small edge region.** The small annotation on the right edge
has weak separation from the surrounding reflection and contour.

![Bent 007: missed small edge defect](results/failure_cases/05_test__bent__007.png)

All test predictions and per-image TP/FP/FN records are saved in `results/`.
The complete split, metrics and timings are in
[final_results.md](results/final_results.md).

## ONNX and benchmarking

Section 14 exports the selected trained model to
`models/metal_nut_detector.onnx`: FP32, opset 17, static batch 1, 640 × 640,
external NMS, no simplification dependency. It runs ONNX checker, loads a CPU
session, compares raw tensors against PyTorch, then displays decoded predictions
on test examples. Raw YOLOv8 single-class output has shape `[1, 5, N]`; it has
class confidence and **no separate objectness score**.

Section 15 benchmarks `CPUExecutionProvider` with 10 warm-up runs and 100 measured
runs using `time.perf_counter()`. It records actual CPU model, provider, thread
counts, image size, runtime version, mean/median/p95 latency, and approximate
serial FPS. Model-only timing uses an already preprocessed tensor. Total timing
includes RGB preprocessing, inference, decoding and NMS on preloaded images.
Disk reads, camera acquisition, drawing and session startup are excluded.
The two scopes are saved separately in `results/cpu_benchmark.csv`. Actual host:
**AMD Ryzen 7 7435HS**, ONNX Runtime 1.31.0, four intra-op threads, one inter-op
thread, batch 1, FP32, 640 input. Both scopes use 10 warm-ups and 100 measured
runs over the same 10 preloaded images:

| ONNX CPU scope | Mean, ms | Median, ms | p95, ms | Approx. FPS |
| --- | --- | --- | --- | --- |
| Model only | 60.33 | 61.02 | 64.20 | 16.58 |
| Preprocess + model + postprocess | 63.31 | 63.80 | 66.31 | 15.80 |

The exported file passes ONNX checker and CPU loading. Raw-output parity against
FP32 PyTorch passes `rtol=1e-3, atol=1e-3` on five category-representative test
images. The two samples with detections have mean best-box IoU effectively 1.0;
empty predictions are recorded as empty, with no invented overlap scores.
See [parity.csv](results/parity.csv). No Raspberry Pi 5 performance is claimed.

Reference behavior was checked against the
[Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/),
[export documentation](https://docs.ultralytics.com/modes/export/), and
[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).

## Engineering checks

The project has been reviewed and simplified. See [CODE_REVIEW.md](CODE_REVIEW.md)
for the fixes and regression checks. Edit the source notebook directly; it is
the single source of the experimental workflow. The executed notebook preserves
the original training run and its measured outputs.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/smoke_check.py
```

Tests cover mask filtering and coordinate round trips, saved-label alignment,
deterministic split proportions/strata, normal empty labels, duplicate leakage,
missing/misaligned/orphan masks, stale prepared files, IoU matching with duplicate
and empty detections, letterbox recovery, confidence filtering, NMS and an actual
CPU ONNX Runtime call on a synthetic graph. They also compile notebook cells.
The smoke check runs every notebook section in a temporary project with synthetic
images, one-epoch random-initialized YOLOv8n models and reduced input size. It
checks execution and export plumbing, **not trained detector quality**. Synthetic
weights, figures, and detection metrics are discarded and never presented as
assignment results.

## Written assignment answers

### A — Only 30 defective images and 3,000 good images

I would first inspect what those 30 images represent. Thirty independent parts
covering several defect mechanisms are different from thirty frames of one
damaged part. I would record defect category, production lot, camera, date, and
severity, remove duplicates, and ask the customer which defects must never pass.
The first holdout should contain independent parts or lots. With so few defects,
a single random split gives an unstable recall estimate; I would use grouped
cross-validation within development data and keep a small, untouched final audit
set. All augmented copies must stay in their source image's training fold.

My first technical baseline would be anomaly detection trained mainly on the
good population, for example pretrained patch features with a nearest-neighbor
normal reference. This uses the available information instead of trying to learn
every defect from thirty examples. I would ensure the good reference covers
legitimate variation such as surface finish and orientation. Contaminated good
data or an overly narrow normal reference can make anomaly detection unreliable.
An anomaly score can produce a heatmap and candidate regions, but image-level
rejection and localized defect detection must be evaluated separately.

In parallel I would fine-tune a small pretrained detector if useful boxes can be
annotated. Transfer learning is preferable to training from scratch. I would
oversample defect-bearing training images enough that they appear regularly in
batches, while retaining diverse good examples to learn false-positive control.
I would not discard most of the normal set or duplicate defects into validation.
Augmentations should reflect the camera: modest in-plane rotation, plausible
exposure variation, or sensor noise when appropriate. Color changes are unsafe
for a discoloration defect, and aggressive downscaling may erase fine scratches.
Synthetic anomalies can supplement training, but cannot substitute for real
defects in the audit set.

The operating threshold comes from development/validation data and the cost of
escapes versus false rejects. I would report defect recall by category and size,
false rejects on good parts, box localization quality, and uncertainty caused by
the small sample. Accuracy is misleading when good parts dominate. A practical
launch could route uncertain parts to manual inspection and collect reviewed
false rejects and escapes. Labeling the next diverse failures often improves
reliability more than adding complexity to a model trained on thirty examples.

### B — Good test scores, missed defects after two weeks

I would first preserve recent missed-defect frames, nearby normal frames, raw
camera images, predictions and software/configuration versions. If escapes are
costly, I would enable the agreed manual-inspection fallback while investigating.
The first comparison is a known reference part captured today versus its
commissioning image. Check lens contamination, focus, working distance, camera
movement, vibration, exposure, gain, white balance and lighting position or aging.
Metal surfaces are reflective: a small light-angle change can erase a scratch
without an obvious software failure. Check full-resolution frames, not only a
resized dashboard thumbnail.

Next I would ask production what changed: supplier, alloy or coating, machining
tool wear, lubricant, surface finish, part orientation, conveyor speed, or new
defect types. Compare recent images and defect sizes against development data by
lot and time. A new defect mechanism is not solved simply by lowering confidence.
Review a labeled sample from both accepted and rejected parts; inspecting only
rejected images cannot estimate the escaped-defect rate.

Then I would reproduce a missed frame offline using the deployed checkpoint and
the original validated runtime. Verify model hash, class map, image resolution,
RGB/BGR ordering, normalization, letterboxing, ROI cropping, inverse coordinates,
confidence threshold and NMS settings. Check for a configuration update two weeks
ago. Compare raw outputs with the service output to distinguish weak model scores
from a decoding, threshold or controller-integration problem. A threshold change
requires evidence from labeled validation data and an explicit false-reject budget.

I would also inspect runtime behavior under load: dropped or stale frames,
incorrect image/part association, timeouts, thermal throttling, power limits,
memory pressure, device/provider fallback, and any quantized model mismatch.
A perfectly accurate prediction on the previous part is still a production
failure. Record capture-to-decision latency and identifiers through the controller.
Once the cause is isolated, validate the correction on representative recent lots
and a protected holdout. Add drift monitoring for image statistics, score
distributions and reviewed escape/false-reject rates, with a clear rollback path.
Do not continuously retrain on unreviewed model predictions.

### C — A 10 ms budget but a 40 ms model

I would clarify whether the 10 ms limit is model execution or the complete
capture-to-controller decision. Then I would profile acquisition, resize/copy,
inference, decoding/NMS and communication separately on the actual controller,
with batch 1, warm-up and repeated p95 measurements. A 40 ms mean measured on a
laptop is not a deployment specification. Check which execution provider is
actually active and whether unexpected copies or CPU fallback dominate.

For a compute-bound model, I would first benchmark a nano architecture or a
lighter backbone and reduce the input resolution in controlled steps. A tightly
defined product ROI can reduce background compute without shrinking scratches
as much as resizing the entire frame. However, ROI detection must be reliable
and included in the latency budget. I would compare recall by defect size and
type at every resolution; a faster detector that misses the customer's smallest
critical scratch is not an acceptable improvement.

Next I would export to the runtime best supported by the controller: ONNX
Runtime or OpenVINO on a suitable CPU, or an appropriate accelerator runtime.
Use static shapes, reuse sessions and buffers, choose sensible thread counts,
and avoid Python or image-format conversions in the hot path. Check graph
optimization and end-to-end output parity before accepting a speed claim. INT8
quantization is worth testing on representative **training/development** calibration
images; evaluate thin or low-contrast defects carefully because their signals may
be disproportionately affected. FP16 helps only where the hardware and runtime
support it efficiently. Quantization-aware training is an option if simple
post-training quantization loses too much recall.

If these changes cannot meet 10 ms at the required recall, I would evaluate an
NPU, GPU, or other supported accelerator, including transfer time, power and cost.
Batching may improve throughput but usually conflicts with a strict per-part
deadline. Pipelining helps only if the allowed decision delay permits it. A
two-stage cascade can save average compute, but its first stage must not reject
critical defects and its worst case still needs to meet the contract. I would
present measured latency/recall trade-offs and recommend a hardware or process
change when the physical requirement cannot be met honestly.

## AI assistance disclosure

This project was generated and reviewed with OpenAI Codex assistance, including
the notebook, helper modules, documentation, and synthetic verification code.
Automated local checks verify the behaviors recorded in
`results/verification.json`: 21 passing tests, valid notebook structure, full
synthetic execution, and the completed real-data workflow on this machine.
The real models trained for 50 epochs each on the RTX 2050, followed by held-out
evaluation, ONNX validation and CPU measurement. These are observed automated
results, not independent human validation or an independently reproduced study.
No production validation or Raspberry Pi benchmark is claimed. Before submission,
the applicant should review the executed notebook, annotations and failures,
repeat the experiment where practical, and explain every design decision.
