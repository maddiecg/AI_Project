
"""
Designer AI - Clothing Classification Model Training

Trains a MobileNetV2 image classifier using transfer learning.

Categories:
    bottoms, dresses, jackets, tops

Input:
    Data/prepared/train.csv
    Data/prepared/validation.csv

Output:
    Backend/clothing_model.keras
    Backend/class_names.json
    Backend/training_history.csv

Run from the main project folder:
    python Backend/train_model.py
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -------------------------------------------------
# PROJECT SETTINGS
# -------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "Data"

if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"

PREPARED_DIR = DATA_DIR / "prepared"

TRAIN_CSV = PREPARED_DIR / "train.csv"
VAL_CSV = PREPARED_DIR / "validation.csv"

MODEL_PATH = BACKEND_DIR / "clothing_model.keras"
CLASSES_PATH = BACKEND_DIR / "class_names.json"
HISTORY_PATH = BACKEND_DIR / "training_history.csv"

# Must match the evaluation script.
IMG_SIZE = (224, 224)

BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
LEARNING_RATE = 0.001

CLASS_NAMES = [
    "bottoms",
    "dresses",
    "jackets",
    "tops",
]

NUM_CLASSES = len(CLASS_NAMES)

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# -------------------------------------------------
# LOAD PREPARED DATA
# -------------------------------------------------

def load_data(csv_path):
    """Read the CSV and verify image paths."""

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing dataset file: {csv_path}\n"
            "Run Notebooks/prepare_data.py first."
        )

    df = pd.read_csv(
        csv_path,
        dtype={"id": str}
    )

    required = {"image_path", "category"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path.name} must contain "
            "image_path and category columns."
        )

    df = df.dropna(
        subset=["image_path", "category"]
    ).copy()

    df["category"] = (
        df["category"].astype(str).str.strip()
    )

    unknown = set(df["category"]) - set(CLASS_NAMES)

    if unknown:
        raise ValueError(
            f"Unexpected categories: {unknown}"
        )

    def resolve_image_path(value):
        path = Path(str(value).replace("\\", "/"))

        if path.is_absolute():
            return path

        return PROJECT_DIR / path

    df["full_path"] = df["image_path"].apply(
        resolve_image_path
    )

    missing = [
        str(path)
        for path in df["full_path"]
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images could not be found.\n"
            f"Example: {missing[0]}\n"
            "Check your extracted Data/images folder."
        )

    category_to_index = {
        name: index
        for index, name in enumerate(CLASS_NAMES)
    }

    df["label"] = (
        df["category"].map(category_to_index)
    )

    if df.empty:
        raise ValueError(
            f"No valid records in {csv_path.name}"
        )

    return df


# -------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------

def process_image(path, label):
    """Load an image and scale its pixels to 0-1."""

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def make_dataset(df, training=False):
    """Build a TensorFlow image dataset."""

    paths = df["full_path"].astype(str).tolist()
    labels = df["label"].astype("int32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# -------------------------------------------------
# BUILD MODEL
# -------------------------------------------------

def build_model():
    """Create a MobileNetV2 transfer-learning model."""

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers initially.
    base_model.trainable = False

    inputs = keras.Input(
        shape=(224, 224, 3),
        name="clothing_image"
    )

    # Basic augmentation helps the model
    # recognize clothing in different positions.
    x = layers.RandomFlip("horizontal")(inputs)

    x = layers.RandomRotation(0.05)(x)

    x = layers.RandomZoom(0.1)(x)

    # Input images are 0-1.
    # MobileNetV2 internally needs -1 to 1.
    x = layers.Rescaling(
        scale=2.0,
        offset=-1.0
    )(x)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="clothing_category"
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="DesignerAI_ClothingClassifier"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

def main():

    print("\nDESIGNER AI - MODEL TRAINING")
    print("=" * 40)

    train_df = load_data(TRAIN_CSV)
    val_df = load_data(VAL_CSV)

    # Check the train and validation splits.
    train_ids = set(train_df["full_path"].astype(str))
    val_ids = set(val_df["full_path"].astype(str))

    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise ValueError(
            "Training and validation contain "
            "some of the same images."
        )

    for name, df in [
        ("Training", train_df),
        ("Validation", val_df)
    ]:
        missing_categories = (
            set(CLASS_NAMES) - set(df["category"])
        )

        if missing_categories:
            raise ValueError(
                f"{name} is missing categories: "
                f"{missing_categories}"
            )

    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    print("\nTraining category counts:")
    print(
        train_df["category"]
        .value_counts()
        .to_string()
    )

    train_dataset = make_dataset(
        train_df,
        training=True
    )

    val_dataset = make_dataset(
        val_df,
        training=False
    )

    print("\nBuilding MobileNetV2 model...")

    model = build_model()

    model.summary()

    # Save the model with the highest
    # validation accuracy.
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=False,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.000001,
            verbose=1
        )
    ]

    print("\nStarting training...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Reload the best saved model.
    best_model = keras.models.load_model(
        MODEL_PATH
    )

    # Evaluate on validation data only.
    # Keep the test dataset for final evaluation.
    loss, accuracy = best_model.evaluate(
        val_dataset,
        verbose=1
    )

    print("\nTRAINING COMPLETE")
    print("=" * 40)

    print(
        f"Best validation accuracy: {accuracy:.2%}"
    )

    print(
        f"Validation loss: {loss:.4f}"
    )

    # Save the class order for prediction
    # and evaluation code.
    with CLASSES_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            CLASS_NAMES,
            file,
            indent=4
        )

    # Save learning history.
    pd.DataFrame(history.history).to_csv(
        HISTORY_PATH,
        index=False
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Class names saved to: {CLASSES_PATH}")
    print(f"History saved to: {HISTORY_PATH}")

    print(
        "\nNext: Run model_evaluation.py "
        "against the test dataset."
    )


if __name__ == "__main__":
    main()
