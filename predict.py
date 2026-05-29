"""
predict.py
-----------
Load the saved model and run predictions on sample images.

This script verifies the full pipeline end-to-end:
  saved model → load → preprocess → predict → display result

Usage
-----
    # Auto mode: picks 5 random images from Plant dataset/test/
    python predict.py

    # Specific images:
    python predict.py path/to/image1.jpg path/to/image2.jpg
"""

import sys
import json
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
MODEL_DIR        = BASE_DIR / "models"
TEST_DIR         = BASE_DIR / "Plant dataset" / "test"

BEST_MODEL_PATH  = str(MODEL_DIR / "best_model.keras")
CLASS_INDEX_PATH = str(MODEL_DIR / "class_indices.json")

IMG_SIZE         = None     # Detected from model below
NUM_SAMPLES      = 5        # images to sample when no path is given
# ─────────────────────────────────────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_model_and_classes():
    """Load model and return (model, idx_to_class)."""
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    
    # Detect IMG_SIZE from model
    global IMG_SIZE
    IMG_SIZE = (model.input_shape[1], model.input_shape[2])

    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)          # {class_name: int}
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


def preprocess_image(image_path: str) -> np.ndarray:
    """Load, resize, and normalise a single image → (1, H, W, 3)."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")
    # Use the same preprocessing as training
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_image(model, image_path: str, idx_to_class: dict) -> dict:
    """Return prediction dict for one image."""
    img_tensor = preprocess_image(image_path)
    probs      = model.predict(img_tensor, verbose=0)[0]
    top_idx    = int(np.argmax(probs))
    return {
        "path":       str(image_path),
        "predicted":  idx_to_class[top_idx],
        "confidence": float(probs[top_idx]),
        "top3": [
            {"class": idx_to_class[i], "confidence": float(probs[i])}
            for i in np.argsort(probs)[::-1][:3]
        ],
    }


def find_sample_images(test_dir: Path, n: int) -> list:
    """Pick n random images spread across test class folders."""
    all_images = []
    for cls_dir in sorted(test_dir.iterdir()):
        if cls_dir.is_dir():
            imgs = [
                f for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]
            if imgs:
                all_images.append(random.choice(imgs))

    random.seed(42)
    random.shuffle(all_images)
    return all_images[:n]


def print_result(result: dict, index: int):
    """Pretty-print a single prediction result."""
    confidence_bar = "#" * int(result["confidence"] * 20)
    print(f"\n  [{index}] {Path(result['path']).name}")
    print(f"       Path        : {result['path']}")
    print(f"       Prediction  : {result['predicted']}")
    print(f"       Confidence  : {result['confidence']*100:.2f}%  {confidence_bar}")
    print(f"       Top-3:")
    for r in result["top3"]:
        bar = "*" * int(r["confidence"] * 20)
        print(f"           {r['class']:<55}  {r['confidence']*100:5.2f}%  {bar}")


def main():
    print("=" * 60)
    print("  Plant Disease Detection — Prediction Verification")
    print("=" * 60)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n  Loading model from: {BEST_MODEL_PATH}")
    model, idx_to_class = load_model_and_classes()
    print(f"  Model loaded successfully ({len(idx_to_class)} classes)")

    # ── Pick images ─────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        image_paths = [Path(p) for p in sys.argv[1:]]
        print(f"\n  Running prediction on {len(image_paths)} provided image(s)...")
    else:
        print(f"\n  Auto-selecting {NUM_SAMPLES} sample images from test split...")
        image_paths = find_sample_images(TEST_DIR, NUM_SAMPLES)
        if not image_paths:
            print(f"  Warning: No images found in {TEST_DIR}.")
            print("  Run prepare_test_set.py first, or provide image paths as arguments.")
            return

    # ── Predict ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    correct = 0
    for i, img_path in enumerate(image_paths, 1):
        result = predict_image(model, str(img_path), idx_to_class)
        print_result(result, i)

        # Auto-check: class name is the parent folder name in test/
        true_label = img_path.parent.name
        if true_label == result["predicted"]:
            correct += 1
            print(f"       Correct! (true label: {true_label})")
        else:
            print(f"       Incorrect.  True label: {true_label}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE: Predictions complete!")
    print(f"  Correct: {correct}/{len(image_paths)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
