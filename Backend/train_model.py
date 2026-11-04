"""Designer AI: train a MobileNetV2 model to classify clothing and footwear.

Run AFTER prepare_data.py:
    python Backend/train_model.py

Recognizes: bottoms, dresses, shoes, tops.
"""

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
# TensorFlow is used to train our clothing classification model.
import tensorflow as tf

# Keras provides the tools for building neural networks.
keras = tf.keras

# Layers are the building blocks of our AI model.
layers = tf.keras.layers

# Our model files are saved in Backend; our CSV files live in Data/prepared.
PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data"
if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"
PREPARED_DIR = DATA_DIR / "prepared"

MODEL_FILE = BACKEND_DIR / "clothing_model.keras"
CLASS_FILE = BACKEND_DIR / "class_names.json"
HISTORY_FILE = BACKEND_DIR / "training_history.csv"

# IMPORTANT: The order here determines the model's numeric class labels.
CLASS_NAMES = ["bottoms", "dresses", "shoes", "tops"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

# Fixed seeds make results more consistent between runs.
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_csv(filename):
    """Load labels and turn relative image paths into complete paths."""
    path = PREPARED_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run prepare_data.py first.")

    data = pd.read_csv(path)
    if not {"image_path", "category"}.issubset(data.columns):
        raise ValueError(f"{filename} needs image_path and category columns.")
    data = data.dropna(subset=["image_path", "category"]).copy()
    if data.empty:
        raise ValueError(f"{filename} has no usable rows.")

    # Every category in the CSV must be one of the four expected classes.
    unexpected = set(data["category"]) - set(CLASS_NAMES)
    if unexpected:
        raise ValueError(f"Unexpected categories in {filename}: {unexpected}")

    data["full_path"] = data["image_path"].apply(
        lambda p: str(PROJECT_DIR / str(p))
    )
    missing = [p for p in data["full_path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images. Example: {missing[0]}")

    # Convert category names to numbers: bottoms=0, dresses=1, shoes=2, tops=3.
    label_numbers = {name: index for index, name in enumerate(CLASS_NAMES)}
    data["label"] = data["category"].map(label_numbers).astype("int32")
    return data


def read_image(path, label):
    """Read and resize a photo; return RGB pixels between 0 and 1."""
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(data, training=False):
    """Build batches of images for TensorFlow."""
    dataset = tf.data.Dataset.from_tensor_slices((
        data["full_path"].tolist(), data["label"].tolist()
    ))
    if training:
        # Shuffle only the training data, never the validation data.
        dataset = dataset.shuffle(len(data), seed=SEED)
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    """Use a pretrained MobileNetV2 to learn our four categories."""
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # Train just the new category layer to start.

    inputs = keras.Input(shape=(224, 224, 3))
    # Random changes help the model handle slightly different photos.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    # MobileNetV2 expects -1 to 1; our input images are 0 to 1.
    x = layers.Rescaling(scale=2.0, offset=-1.0)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="DesignerAI_ClothingClassifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("\nDESIGNER AI - TRAINING TOPS, BOTTOMS, DRESSES, AND SHOES")
    train = load_csv("train.csv")
    validation = load_csv("validation.csv")
    print(f"Training images: {len(train)}")
    print(f"Validation images: {len(validation)}")
    print("\nTraining category counts:\n", train["category"].value_counts().to_string())

    train_data = make_dataset(train, training=True)
    validation_data = make_dataset(validation)
    model = build_model()

    # Save whichever epoch has the highest validation accuracy.
    # This replaces the OLD jackets model at the same path.
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_FILE), monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]

    print("\nStarting training...")
    history = model.fit(
        train_data, validation_data=validation_data,
        epochs=EPOCHS, callbacks=callbacks
    )

    # Reload the best model, not just the last training epoch.
    best_model = keras.models.load_model(MODEL_FILE)
    loss, accuracy = best_model.evaluate(validation_data, verbose=0)

    # The evaluation file reads this JSON to avoid category-order mistakes.
    CLASS_FILE.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    pd.DataFrame(history.history).to_csv(HISTORY_FILE, index=False)

    print("\nTRAINING COMPLETE")
    print(f"Best validation accuracy: {accuracy:.2%}")
    print(f"Validation loss: {loss:.4f}")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Class names saved to: {CLASS_FILE}")
    print("Next: python Notebooks/modelEvaluation.py")


if __name__ == "__main__":
    main()
