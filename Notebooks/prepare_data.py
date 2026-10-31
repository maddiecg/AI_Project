
# Prepare clothing data for the Designer AI project.
# Verifies images, maps categories, and creates train/validation/test sets.

from pathlib import Path
import csv

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


# PROJECT PATHS
# This file stays inside the Notebooks folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Find the Data folder regardless of capitalization.
DATA_DIR = PROJECT_DIR / "Data"

if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"

CSV_PATH = DATA_DIR / "styles.csv"
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_DIR = DATA_DIR / "prepared"

SEED = 42

# Map clothing article types to our four categories.
CATEGORY_MAP = {
    "Tshirts": "tops",
    "Shirts": "tops",
    "Tops": "tops",
    "Sweatshirts": "tops",
    "Sweaters": "tops",

    "Jeans": "bottoms",
    "Shorts": "bottoms",
    "Trousers": "bottoms",
    "Track Pants": "bottoms",
    "Leggings": "bottoms",
    "Capris": "bottoms",
    "Skirts": "bottoms",
    "Jeggings": "bottoms",

    "Jackets": "jackets",
    "Rain Jacket": "jackets",
    "Blazers": "jackets",
    "Nehru Jackets": "jackets",

    "Dresses": "dresses",
}

CATEGORIES = ["tops", "bottoms", "jackets", "dresses"]


def find_dataset():
    """Locate the CSV and image folder inside Data."""

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}"
        )

    csv_files = sorted(DATA_DIR.rglob("styles.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "styles.csv was not found in Data. "
            "Make sure the CSV ZIP file is extracted."
        )

    csv_path = csv_files[0]

    image_folders = [
        folder
        for folder in DATA_DIR.rglob("*")
        if folder.is_dir() and folder.name.lower() == "images"
    ]

    if IMAGE_DIR.is_dir():
        image_folder = IMAGE_DIR
    elif image_folders:
        image_folder = image_folders[0]
    else:
        raise FileNotFoundError(
            "Images folder not found in Data. "
            "Make sure the image ZIP file is extracted."
        )

    return csv_path, image_folder


def main():

    print("Preparing Designer AI dataset...")

    csv_path, image_dir = find_dataset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CSV: {csv_path}")
    print(f"Images: {image_dir}")

    # Find all clothing images.
    image_paths = {}

    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }:
            if path.stem not in image_paths:
                image_paths[path.stem] = path

    print(f"Found {len(image_paths)} images.")

    records = []
    malformed = 0
    missing = 0
    unreadable = 0
    seen_ids = set()

    # Read the dataset.
    with csv_path.open(
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "id", "articleType", "baseColour", "usage"
        }

        if not required_columns.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "styles.csv is missing required columns."
            )

        for row in reader:

            # Skip malformed rows.
            if None in row or any(
                value is None for value in row.values()
            ):
                malformed += 1
                continue

            article_type = row["articleType"].strip()
            category = CATEGORY_MAP.get(article_type)

            # Only keep the four project categories.
            if category is None:
                continue

            item_id = row["id"].strip()

            if not item_id or item_id in seen_ids:
                continue

            seen_ids.add(item_id)

            # Match image to product ID.
            image_path = image_paths.get(item_id)

            if image_path is None:
                missing += 1
                continue

            # Check that the image can be opened.
            try:
                with Image.open(image_path) as image:
                    image.verify()

            except (OSError, ValueError, SyntaxError):
                unreadable += 1
                continue

            records.append({
                "id": item_id,
                "image_path": image_path.relative_to(
                    PROJECT_DIR
                ).as_posix(),
                "category": category,
                "article_type": article_type,
                "color": row["baseColour"].strip(),
                "usage": row["usage"].strip(),
            })

    if not records:
        raise ValueError(
            "No matching clothing images were found."
        )

    # Convert records into a DataFrame.
    data = pd.DataFrame(records)

    print(f"\nUsable clothing images: {len(data)}")

    counts = data["category"].value_counts()

    print("\nCategory counts:")
    print(counts.to_string())

    # Ensure all categories have enough examples.
    if (
        set(counts.index) != set(CATEGORIES)
        or counts.min() < 10
    ):
        raise ValueError(
            "Need at least 10 images in each category.\n"
            f"{counts}"
        )

    # Split dataset:
    # 70% training
    # 15% validation
    # 15% testing

    train, remaining = train_test_split(
        data,
        test_size=0.30,
        stratify=data["category"],
        random_state=SEED,
    )

    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        stratify=remaining["category"],
        random_state=SEED,
    )

    # Save the prepared datasets.
    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        output_path = OUTPUT_DIR / f"{name}.csv"

        split.to_csv(output_path, index=False)

        print(f"\n{name.upper()}: {len(split)} images")
        print(split["category"].value_counts().to_string())

    # Print processing summary.
    print("\nDATA PREPARATION COMPLETE")

    print(f"Skipped malformed rows: {malformed}")
    print(f"Missing images: {missing}")
    print(f"Unreadable images: {unreadable}")

    print(f"\nSaved prepared files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
