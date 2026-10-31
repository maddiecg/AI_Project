"""Evaluate a trained TensorFlow/Keras Designer AI clothing classifier.

Run from anywhere: python Notebooks/model_evaluation.py
Requires: pandas, numpy, pillow, matplotlib, scikit-learn, tensorflow

Assumptions: model expects RGB images scaled to [0, 1] and outputs four
class scores. Change PREPROCESSING if training used a different method.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"

TEST_CSV = DATA_DIR / "prepared" / "test.csv"
OUTPUT_DIR = DATA_DIR / "evaluation"
MODEL_CANDIDATES = [
    PROJECT_DIR / "Backend" / "clothing_model.keras",
    PROJECT_DIR / "Backend" / "clothing_model.h5",
    PROJECT_DIR / "models" / "clothing_model.keras",
    PROJECT_DIR / "models" / "clothing_model.h5",
]

# IMPORTANT: Match this order to the class indices used during TRAINING.
# Alphabetical ordering is common but not guaranteed.
CLASS_NAMES = ["bottoms", "dresses", "jackets", "tops"]
BATCH_SIZE = 32
# Set to "0_255" if model training used raw image pixel values.
PREPROCESSING = "0_1"


def find_model():
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No trained Keras model found. Save clothing_model.keras in Backend/ "
        "(or update MODEL_CANDIDATES to your model's path)."
    )


def load_image(path, image_size):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize(image_size)
            array = np.asarray(img, dtype=np.float32)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"Cannot read image: {path}") from exc
    if PREPROCESSING == "0_1":
        array /= 255.0
    elif PREPROCESSING != "0_255":
        raise ValueError("PREPROCESSING must be '0_1' or '0_255'")
    return array


def main():
    if not TEST_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {TEST_CSV}. Run Notebooks/prepare_data.py first."
        )

    test = pd.read_csv(TEST_CSV, dtype={"id": str})
    required = {"id", "image_path", "category"}
    if not required.issubset(test.columns):
        raise ValueError(f"Test CSV is missing columns: {required - set(test.columns)}")
    test = test.dropna(subset=["image_path", "category"]).copy()
    test["category"] = test["category"].astype(str).str.strip()
    unknown = set(test["category"]) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown test labels: {sorted(unknown)}")
    if test.empty:
        raise ValueError("The test CSV has no usable rows.")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Install TensorFlow: pip install tensorflow") from exc

    model_path = find_model()
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        raise ValueError("This script expects a model with one image input.")
    if len(input_shape) != 4 or input_shape[-1] != 3:
        raise ValueError(f"Expected an RGB image model, got input shape {input_shape}")
    if input_shape[1] is None or input_shape[2] is None:
        raise ValueError("Model image dimensions are dynamic. Set a fixed size in the script.")
    image_size = (int(input_shape[2]), int(input_shape[1]))  # PIL uses width, height

    correct_labels = []
    predicted_labels = []
    rows = []
    skipped = []

    for start in range(0, len(test), BATCH_SIZE):
        chunk = test.iloc[start : start + BATCH_SIZE]
        images = []
        accepted = []
        for _, row in chunk.iterrows():
            raw_path = Path(str(row["image_path"]))
            path = raw_path if raw_path.is_absolute() else PROJECT_DIR / raw_path
            try:
                images.append(load_image(path, image_size))
                accepted.append(row)
            except ValueError as exc:
                skipped.append(str(exc))
        if not images:
            continue
        scores = np.asarray(model.predict(np.stack(images), verbose=0))
        if scores.ndim != 2 or scores.shape != (len(images), len(CLASS_NAMES)):
            raise ValueError(
                f"Expected class scores of shape (batch, {len(CLASS_NAMES)}), "
                f"but model returned {scores.shape}."
            )
        indexes = np.argmax(scores, axis=1)
        for row, index in zip(accepted, indexes):
            actual = str(row["category"])
            predicted = CLASS_NAMES[int(index)]
            correct_labels.append(actual)
            predicted_labels.append(predicted)
            rows.append({
                "id": str(row["id"]),
                "image_path": str(row["image_path"]),
                "actual": actual,
                "predicted": predicted,
                "correct": actual == predicted,
            })

    if not rows:
        raise ValueError("No readable test images were found. Check image_path values.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    accuracy = float(accuracy_score(correct_labels, predicted_labels))
    report = classification_report(
        correct_labels, predicted_labels,
        labels=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(correct_labels, predicted_labels, labels=CLASS_NAMES)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )
    with (OUTPUT_DIR / "evaluation_report.json").open("w", encoding="utf-8") as file:
        json.dump({
            "model": str(model_path.relative_to(PROJECT_DIR)),
            "class_order": CLASS_NAMES,
            "evaluated_images": len(rows),
            "skipped_images": len(skipped),
            "accuracy": accuracy,
            "classification_report": report,
        }, file, indent=2)
    if skipped:
        (OUTPUT_DIR / "skipped_images.txt").write_text(
            "\n".join(skipped), encoding="utf-8"
        )

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    display.plot(cmap="Blues", values_format="d", xticks_rotation=30)
    plt.title("Designer AI: Test-set confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print(f"Model: {model_path}")
    print(f"Evaluated images: {len(rows)} (skipped: {len(skipped)})")
    print(f"Test accuracy: {accuracy:.2%}")
    print(classification_report(
        correct_labels, predicted_labels, labels=CLASS_NAMES,
        zero_division=0,
    ))
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Make sure CLASS_NAMES and PREPROCESSING match your training code.")


if __name__ == "__main__":
    main()
