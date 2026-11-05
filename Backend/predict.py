
"""
Designer AI - Clothing Image Prediction

This file uses our trained AI model to identify clothing
from a new image.

Our model recognizes four categories:
- Tops
- Bottoms
- Dresses
- Shoes (including sandals, boots, and flip-flops)

How it works:
1. Loads our saved MobileNetV2 clothing model.
2. Opens a clothing image.
3. Resizes the image to 224 x 224 pixels.
4. Converts the image into numbers the model can read.
5. Predicts the clothing category.
6. Displays the predicted category and confidence.

Run from the main AI_Project folder:

python Backend/predict.py "path/to/image.jpg"
"""

# ------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------

# Path helps us locate our files.
from pathlib import Path

# sys allows us to read the image path
# entered in the terminal.
import sys

# json allows us to read the class names.
import json

# NumPy converts images into arrays of numbers.
import numpy as np

# Pillow opens and resizes images.
from PIL import Image

# TensorFlow loads our trained AI model.
import tensorflow as tf


# ------------------------------------------
# STEP 2: SET UP FILE PATHS
# ------------------------------------------

# Find the Backend folder.
# This script is saved inside Backend.
BACKEND_DIR = Path(__file__).resolve().parent

# Find the saved model created by train_model.py.
MODEL_PATH = BACKEND_DIR / "clothing_model.keras"

# This file tells us the category order.
CLASS_PATH = BACKEND_DIR / "class_names.json"

# Our model was trained with 224 x 224 images.
IMAGE_SIZE = (224, 224)


# ------------------------------------------
# STEP 3: LOAD THE TRAINED MODEL
# ------------------------------------------

def load_clothing_model():
    """
    Load our trained AI model and clothing categories.

    We use the model that was already trained,
    so we do not need to train it again.
    """

    # Make sure the model file exists.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Trained model not found. "
            "Check Backend/clothing_model.keras."
        )

    # Make sure the class names file exists.
    if not CLASS_PATH.is_file():
        raise FileNotFoundError(
            "Class names not found. "
            "Check Backend/class_names.json."
        )

    # Load our saved Keras model.
    # compile=False is sufficient for making predictions.
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Read the categories saved during training.
    with CLASS_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        class_names = json.load(file)

    # Check that our model uses the new categories.
    expected = {"bottoms", "dresses", "shoes", "tops"}

    if (
        not isinstance(class_names, list)
        or set(class_names) != expected
    ):
        raise ValueError(
            "Class names do not match our four categories."
        )

    return model, class_names


# ------------------------------------------
# STEP 4: PREPARE THE IMAGE
# ------------------------------------------

def prepare_image(image_path):
    """
    Convert a clothing image into the format
    expected by our trained AI model.
    """

    # Make sure the image exists.
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Open the clothing image.
    with Image.open(image_path) as image:

        # Convert the image to RGB.
        # RGB images have three color channels.
        image = image.convert("RGB")

        # Resize the image to 224 x 224 pixels.
        image = image.resize(IMAGE_SIZE)

        # Convert the picture into a NumPy array.
        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    # Normalize pixel values from 0-255 to 0-1.
    # This must match train_model.py.
    image_array = image_array / 255.0

    # Add a batch dimension.
    # The model expects a group of images,
    # even when we only predict one image.
    #
    # Before: (224, 224, 3)
    # After:  (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ------------------------------------------
# STEP 5: MAKE A PREDICTION
# ------------------------------------------

def predict_clothing(image_path):
    """
    Predict the clothing category of a new image.

    Returns:
    - Predicted category
    - Confidence percentage
    """

    # Load our trained model and class names.
    model, class_names = load_clothing_model()

    # Prepare the image for the model.
    image_array = prepare_image(image_path)

    # Ask the AI model to classify the image.
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Check that the model returned one score
    # for each of our four categories.
    if predictions.shape != (1, len(class_names)):
        raise ValueError(
            "Model output does not match the category list."
        )

    # Find the category with the highest score.
    predicted_index = int(
        np.argmax(predictions[0])
    )

    # Convert the category index into its name.
    predicted_category = class_names[predicted_index]

    # Get the model's confidence score.
    confidence = float(
        predictions[0][predicted_index]
    ) * 100

    return predicted_category, confidence


# ------------------------------------------
# STEP 6: RUN THE PROGRAM
# ------------------------------------------

def main():
    """Run a clothing prediction from the terminal."""

    print("\nDESIGNER AI - CLOTHING PREDICTION")
    print("--------------------------------")

    # The user must provide an image path.
    if len(sys.argv) < 2:
        print(
            'Usage: python Backend/predict.py "image.jpg"'
        )
        return

    # Get the image path from the terminal.
    image_path = sys.argv[1]

    # Make a clothing prediction.
    category, confidence = predict_clothing(
        image_path
    )

    # Display the result.
    print(f"\nImage: {Path(image_path).name}")

    print(f"Predicted category: {category}")

    print(f"Confidence: {confidence:.2f}%")

    # Reminder about what this model does.
    print("\nPrediction complete!")


# Run main() when this file is executed directly.
if __name__ == "__main__":
    main()
