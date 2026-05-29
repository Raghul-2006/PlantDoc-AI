"""
evaluate.py
------------
Final evaluation on the held-out TEST set.

Run AFTER training is complete (train.py must have finished).

Usage
-----
    python evaluate.py

Outputs (written to outputs/)
-------------------------------
  confusion_matrix.png
  classification_report.txt
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent / "Plant dataset"
TEST_DIR         = str(BASE_DIR / "test")
MODEL_DIR        = Path(__file__).resolve().parent / "models"
OUTPUT_DIR       = Path(__file__).resolve().parent / "outputs"

BEST_MODEL_PATH  = str(MODEL_DIR / "best_model.keras")
CLASS_INDEX_PATH = str(MODEL_DIR / "class_indices.json")

# We'll detect this from the model
IMG_SIZE         = None 
BATCH_SIZE       = 32
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_classes():
    """Load saved model and invert class_indices → {index: class_name}."""
    print(f"  Loading model from: {BEST_MODEL_PATH}")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    # Detect IMG_SIZE from model
    global IMG_SIZE
    IMG_SIZE = (model.input_shape[1], model.input_shape[2])

    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)           # {class_name: index}

    # Invert for index → class name lookup
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, class_indices, idx_to_class


def build_test_generator(class_indices: dict):
    """Build a test generator aligned with train class order."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=sorted(class_indices.keys()),
    )
    return test_gen


def evaluate_model(model, test_gen, idx_to_class: dict):
    """Run inference, compute metrics, return true/pred labels."""
    print(f"\n  Predicting {test_gen.samples} test images...")
    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = test_gen.classes

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    acc  = accuracy_score(y_true, y_pred)
    loss = tf.keras.losses.CategoricalCrossentropy()(
        tf.keras.utils.to_categorical(y_true, num_classes=len(class_names)),
        y_pred_prob,
    ).numpy()

    return y_true, y_pred, class_names, acc, loss


def save_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    """Plot and save a labelled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    # Normalise for readability
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm_norm,
        annot=False,            # too many classes for text annotations
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Normalised) — PlantVillage 38-Class", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0,  fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


def save_classification_report(y_true, y_pred, class_names: list, save_path: str):
    """Save sklearn classification report to a text file."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(save_path, "w") as f:
        f.write("Plant Disease Detection — Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    print(f"  Classification report saved -> {save_path}")
    return report


def main():
    print("=" * 60)
    print("  Plant Disease Detection — Evaluation on Test Set")
    print("=" * 60)

    # ── Load model & class map ──────────────────────────────────────────────
    print("\n[1/4] Loading model and class indices...")
    model, class_indices, idx_to_class = load_model_and_classes()
    print(f"      Detected IMG_SIZE : {IMG_SIZE}")
    print(f"      Model input shape : {model.input_shape}")
    print(f"      Num classes       : {len(class_indices)}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n[2/4] Preparing test data generator...")
    test_gen = build_test_generator(class_indices)
    print(f"      Test images  : {test_gen.samples}")
    print(f"      Test batches : {len(test_gen)}")

    # ── Evaluate ────────────────────────────────────────────────────────────
    print("\n[3/4] Running evaluation...")
    y_true, y_pred, class_names, acc, loss = evaluate_model(
        model, test_gen, idx_to_class
    )

    # ── Reports ─────────────────────────────────────────────────────────────
    print("\n[4/4] Generating reports...")
    cm_path      = str(OUTPUT_DIR / "confusion_matrix.png")
    report_path  = str(OUTPUT_DIR / "classification_report.txt")

    save_confusion_matrix(y_true, y_pred, class_names, cm_path)
    report = save_classification_report(y_true, y_pred, class_names, report_path)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE: Evaluation Complete!")
    print(f"  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"{'='*60}\n")
    print("Per-class summary (first 10 lines):")
    print("\n".join(report.splitlines()[:14]))
    print("  ...")
    print(f"\n  Full report -> {report_path}")
    print(f"  Confusion matrix -> {cm_path}\n")


if __name__ == "__main__":
    main()
