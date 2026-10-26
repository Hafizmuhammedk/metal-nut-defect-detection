# Model artifacts

Both real-data models completed 50 epochs. The selected model is **augmented**,
chosen using validation mAP@0.5:0.95 before the held-out test evaluation.

- `yolov8n.pt`: original pretrained Ultralytics checkpoint.
- `metal_nut_detector.pt`: selected fine-tuned detector (6,249,514 bytes).
- `metal_nut_detector.onnx`: checked FP32 export (12,238,511 bytes).

ONNX checker, CPU Runtime loading and raw-output parity checks passed. Test
precision is 0.9167, recall 0.4583 and mAP@0.5 0.6516. Low recall is a material
limitation: this experiment is not a production acceptance test.

Notebook section 9 saves both runs' best checkpoints beneath `results/runs/`.
The validation-selected checkpoint is copied to `models/metal_nut_detector.pt`;
section 14 creates and validates `models/metal_nut_detector.onnx`. It is a
single-class YOLOv8n FP32 model with batch 1, static 640 x 640 input, and external
NMS. Do not substitute a COCO checkpoint and describe it as a trained nut model.
