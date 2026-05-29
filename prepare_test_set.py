"""
prepare_test_set.py
--------------------
Carves out a test split from the val folder.

Strategy
--------
- Reads  : Plant dataset/val/<class>/
- Moves  : 50 % of each class's images → Plant dataset/test/<class>/
- Result : val → ~10 % of full data,  test → ~10 % of full data

Run ONCE before training:
    python prepare_test_set.py

The script is idempotent: if Plant dataset/test/ already
contains images for a class it will skip that class.
"""

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent / "Plant dataset"
VAL_DIR       = BASE_DIR / "val"
TEST_DIR      = BASE_DIR / "test"
TEST_FRACTION = 0.50          # move 50 % of val images to test
RANDOM_SEED   = 42
# ────────────────────────────────────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_images(folder: Path) -> list:
    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]


def prepare_test_split():
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Val directory not found: {VAL_DIR}")

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in VAL_DIR.iterdir() if d.is_dir()])
    print(f"\n📂 Found {len(classes)} classes in val folder.")
    print(f"📁 Test split will be saved to: {TEST_DIR}\n")

    random.seed(RANDOM_SEED)

    summary = []
    for cls in tqdm(classes, desc="Creating test split", unit="class"):
        src_cls_dir = VAL_DIR / cls
        dst_cls_dir = TEST_DIR / cls
        dst_cls_dir.mkdir(parents=True, exist_ok=True)

        # Skip if test class already has images (idempotent)
        existing_test_images = get_images(dst_cls_dir)
        if existing_test_images:
            summary.append((cls, 0, len(existing_test_images), "skipped (already exists)"))
            continue

        images = get_images(src_cls_dir)
        if not images:
            summary.append((cls, 0, 0, "⚠️ no images in val"))
            continue

        random.shuffle(images)
        n_move = max(1, int(len(images) * TEST_FRACTION))
        to_move = images[:n_move]

        for img_path in to_move:
            shutil.move(str(img_path), str(dst_cls_dir / img_path.name))

        summary.append((cls, n_move, len(images) - n_move, "✅"))

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  {'Class':<50} {'Moved→Test':>10} {'Left in Val':>11} Status")
    print(f"  {'-'*50} {'-'*10} {'-'*11} ------")
    total_moved = 0
    for cls, moved, remaining, status in summary:
        print(f"  {cls:<50} {moved:>10} {remaining:>11}  {status}")
        total_moved += moved
    print(f"{'='*72}")
    print(f"\n✅ Done!  {total_moved} images moved to test split.")
    print(f"   Train : {BASE_DIR / 'train'}  (unchanged)")
    print(f"   Val   : {VAL_DIR}")
    print(f"   Test  : {TEST_DIR}\n")


if __name__ == "__main__":
    prepare_test_split()
