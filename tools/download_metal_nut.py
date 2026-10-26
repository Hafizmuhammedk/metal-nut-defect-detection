"""Download and verify the MVTec AD metal_nut dataset for this project.

The dataset is downloaded locally and must not be committed to GitHub.
Original MVTec AD license: CC BY-NC-SA 4.0 (non-commercial).
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image


REPO_ID = "MSherbinii/mvtec-ad-metal-nut"
REVISION = "8db7371c35a94709cbb3e69ac69f80d2716e2f8a"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
CATEGORY_DIR = DATA_DIR / "metal_nut"


def image_files(directory):
    return sorted(directory.glob("*.png")) if directory.is_dir() else []


def check_image(path):
    with Image.open(path) as image:
        size = image.size
        image.verify()
    return size


def verify_dataset():
    """Check required folders, original images, and image-to-mask pairs."""
    train_dir = CATEGORY_DIR / "train" / "good"
    test_dir = CATEGORY_DIR / "test"
    masks_dir = CATEGORY_DIR / "ground_truth"
    for directory in (train_dir, test_dir, masks_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing required directory: {directory}")

    normal_train = image_files(train_dir)
    normal_test = image_files(test_dir / "good")
    if not normal_train or not normal_test:
        raise RuntimeError("Missing normal images in train or test")

    categories = sorted(
        path.name for path in test_dir.iterdir()
        if path.is_dir() and path.name != "good"
    )
    if not categories:
        raise RuntimeError("Missing defective image categories")

    for path in normal_train + normal_test:
        check_image(path)

    defect_count = 0
    checked_masks = set()
    for category in categories:
        images = image_files(test_dir / category)
        if not images:
            raise RuntimeError(f"No defective images in category {category}")

        for image_path in images:
            mask_path = masks_dir / category / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask: {mask_path}")
            if check_image(image_path) != check_image(mask_path):
                raise ValueError(f"Image and mask sizes differ: {image_path.name}")
            checked_masks.add(mask_path.resolve())
        defect_count += len(images)
        print(f"  {category}: {len(images)} defective images")

    available_masks = {path.resolve() for path in masks_dir.rglob("*.png")}
    if available_masks != checked_masks:
        raise RuntimeError("Found orphan masks or unexpected ground-truth files")

    total = len(normal_train) + len(normal_test) + defect_count
    print(f"  good: {len(normal_train) + len(normal_test)} normal images")
    print(f"  total: {total} inspection images, {len(checked_masks)} masks")
    print(f"Dataset ready at: {CATEGORY_DIR}")
    print("The notebook makes its custom 80/20 split after inspection.")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading MVTec AD metal_nut from the public Hugging Face mirror.")
    print(f"Source: https://huggingface.co/datasets/{REPO_ID}")
    print(f"Revision: {REVISION}")
    print(f"Destination: {CATEGORY_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=["metal_nut/**"],
        local_dir=DATA_DIR,
        max_workers=4,
    )
    print("\nVerifying images and corresponding defect masks...")
    verify_dataset()


if __name__ == "__main__":
    main()
