# Prepare the dataset by verifying images, mapping categories, and splitting into train, validation, and test sets.

from pathlib import Path
import csv

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


# Paths are relative to this Python file.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CSV_PATH = DATA_DIR / "styles.csv"
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_DIR = DATA_DIR / "prepared"

SEED = 42

# Map the dataset's clothing types to our project categories.
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


def main():
    if not CSV_PATH.is_file() or not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            "Place styles.csv and the images folder inside data/."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find images, including images inside subfolders.
    image_paths = {}
    for path in sorted(IMAGE_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }:
            if path.stem in image_paths:
                raise ValueError(f"Duplicate image ID: {path.stem}")
            image_paths[path.stem] = path

    records = []
    malformed = 0
    missing = 0
    unreadable = 0
    seen_ids = set()

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Skip malformed CSV rows for now.
            if None in row or any(value is None for value in row.values()):
                malformed += 1
                continue

            category = CATEGORY_MAP.get(row["articleType"])
            if category is None:
                continue

            item_id = row["id"]
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            image_path = image_paths.get(item_id)
            if image_path is None:
                missing += 1
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError, SyntaxError):
                unreadable += 1
                continue

            records.append({
                "id": item_id,
                "image_path": image_path.relative_to(PROJECT_DIR).as_posix(),
                "category": category,
                "article_type": row["articleType"],
                "color": row["baseColour"],
                "usage": row["usage"],
            })

    if not records:
        raise ValueError("No matching clothing images were found.")

    data = pd.DataFrame(records)
    counts = data["category"].value_counts()

    # Check that every category has enough examples to split.
    if len(counts) != 4 or counts.min() < 10:
        raise ValueError(
            f"Need at least 10 usable images per category.\n{counts}"
        )

    # 70% training, 15% validation, 15% testing.
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

    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        split.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(f"\n{name.upper()}: {len(split)} images")
        print(split["category"].value_counts().to_string())

    print(f"\nSkipped malformed rows: {malformed}")
    print(f"Missing images: {missing}")
    print(f"Unreadable images: {unreadable}")
    print(f"\nSaved split files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
