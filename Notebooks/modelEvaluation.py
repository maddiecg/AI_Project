"""Designer AI: evaluate the trained clothing and footwear classifier.

Run AFTER train_model.py:
    python Notebooks/modelEvaluation.py

Saves a report, predictions, and confusion matrix in Data/evaluation/.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")  # Save the confusion matrix to a PNG.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

# Find the existing project folders. Leave this script in Notebooks.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"

TEST_FILE = DATA_DIR / "prepared" / "test.csv"
MODEL_FILE = PROJECT_DIR / "Backend" / "clothing_model.keras"
CLASS_FILE = PROJECT_DIR / "Backend" / "class_names.json"
OUTPUT_DIR = DATA_DIR / "evaluation"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# All four expected categories. Training saves their exact numeric order.
EXPECTED_CLASSES = {"bottoms", "dresses", "shoes", "tops"}


def load_image(path):
    """Open a clothing image using the SAME preprocessing as training."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        pixels = np.asarray(image, dtype=np.float32)
    return pixels / 255.0  # Our model expects pixels from 0 to 1.


def main():
    print("\nDESIGNER AI - MODEL EVALUATION")

    # Make sure preparation and training have both finished.
    for path in [TEST_FILE, MODEL_FILE, CLASS_FILE]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Prepare data and train again first.")

    # Read the class order that train_model.py saved alongside the model.
    # This prevents accidentally treating a shoe prediction as a jacket.
    class_names = json.loads(CLASS_FILE.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or set(class_names) != EXPECTED_CLASSES:
        raise ValueError("class_names.json does not contain the new four categories.")

    test = pd.read_csv(TEST_FILE, dtype={"id": str})
    if not {"image_path", "category"}.issubset(test.columns):
        raise ValueError("test.csv needs image_path and category columns.")
    test = test.dropna(subset=["image_path", "category"]).copy()
    if test.empty:
        raise ValueError("There are no test images to evaluate.")
    if not set(test["category"]).issubset(set(class_names)):
        raise ValueError("Test labels do not match the trained model's categories.")

    # Load the model trained on the new categories.
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    print(f"Testing images: {len(test)}")
    print(f"Class names: {class_names}")

    actual_labels = []
    predicted_labels = []
    results = []
    skipped = 0

    # Evaluate in small batches so we don't load all images at once.
    for start in range(0, len(test), BATCH_SIZE):
        batch = test.iloc[start:start + BATCH_SIZE]
        images = []
        valid_rows = []

        for _, item in batch.iterrows():
            # The CSV paths are relative to the main AI_Project folder.
            path = PROJECT_DIR / str(item["image_path"])
            try:
                images.append(load_image(path))
                valid_rows.append(item)
            except (OSError, ValueError):
                skipped += 1  # Skip corrupted or missing images.

        if not images:
            continue

        # Predict one of the four clothing categories for every image.
        predictions = model.predict(np.stack(images), verbose=0)
        if predictions.shape != (len(images), len(class_names)):
            raise ValueError("Model output does not match the four clothing categories.")

        for item, prediction in zip(valid_rows, predictions):
            actual = str(item["category"])
            predicted = class_names[int(np.argmax(prediction))]
            actual_labels.append(actual)
            predicted_labels.append(predicted)
            results.append({
                "image_path": item["image_path"],
                "actual": actual,
                "predicted": predicted,
                "correct": actual == predicted,
            })

    if not results:
        raise ValueError("No test images could be evaluated.")

    # Overall accuracy is useful, but ALSO check each category's recall.
    accuracy = accuracy_score(actual_labels, predicted_labels)
    report = classification_report(
        actual_labels, predicted_labels,
        labels=class_names, zero_division=0, output_dict=False
    )
    # Pylance can otherwise think classification_report returned a dictionary.
    assert isinstance(report, str)

    print(f"\nImages evaluated: {len(results)}")
    print(f"Skipped images: {skipped}")
    print(f"Test accuracy: {accuracy:.2%}\n")
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save per-image predictions and the readable classification report.
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    (OUTPUT_DIR / "classification_report.txt").write_text(
        f"Designer AI Model Evaluation\nAccuracy: {accuracy:.2%}\n\n{report}",
        encoding="utf-8"
    )

    # A confusion matrix shows the categories the model mixes up.
    matrix = confusion_matrix(actual_labels, predicted_labels, labels=class_names)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d", xticks_rotation=30)
    plt.title("Designer AI: Clothing and Footwear")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
