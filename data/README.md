# Dataset installation

Download the **metal_nut** category of the original MVTec AD dataset from
[MVTec](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), using their
download form. Extract it locally (do not use MVTec AD 2).

Expected root, verified against your actual files by notebook section 4:

```text
data/metal_nut/
  train/good/*.png
  test/good/*.png
  test/bent/*.png
  test/color/*.png
  test/flip/*.png
  test/scratch/*.png
  ground_truth/bent/*_mask.png
  ground_truth/color/*_mask.png
  ground_truth/flip/*_mask.png
  ground_truth/scratch/*_mask.png
```

If your archive has an outer directory, move `metal_nut` here or set
`DATASET_DIR` in notebook section 3. The supplied local copy has now been
inspected: 335 images (242 good, 93 defective) and 93 masks, all at 700 x 700.
It was copied from the user's downloader folder into the ignored `data/metal_nut`.
The source is the `MSherbinii/mvtec-ad-metal-nut` Hugging Face mirror at revision
`8db7371c35a94709cbb3e69ac69f80d2716e2f8a`. SHA-256 hashes of all 428 PNG files
match the downloader's cached LFS etags. This verifies the downloaded mirror,
not its equivalence to an independently obtained original MVTec archive.
Provenance is saved in `data/source.json`. No dataset is committed or distributed.
The inspector prints the actual inventory, rejects missing/orphan masks, checks
image/mask dimensions and refuses identical image files before writing labels.

All nonzero mask pixels are foreground. Use one box per 8-connected component
with **at least 9 foreground pixels**, measured at original image resolution.
The fixed rule removes isolated specks, but may also remove real tiny defects;
review the mask/box plots before training. Do not adjust it using held-out test
results. A defective image with no retained region raises an error.

The notebook combines original train and test images to create its custom
supervised 80/20 split. Results are **not official MVTec benchmark scores**.
All images are split before any training augmentation. The test portion is never
used for training, validation, calibration, or model selection.

MVTec AD is licensed **CC BY-NC-SA 4.0**, including a noncommercial restriction.
See the source page and included dataset license for attribution and terms.
Keep all raw images, masks, and prepared copies out of version control.
