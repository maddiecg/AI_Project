"""Designer AI: match images to labels and prepare the dataset for training.

Run from the main project folder:
    python Notebooks/prepare_data.py

Creates Data/prepared/train.csv, validation.csv, and test.csv.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Keep this script inside Notebooks. No folder changes are needed.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = DATA_DIR / "prepared"
SEED = 42  # Using the same seed gives the same split when the data is unchanged.

# Clothing categories recognized by our model.
CATEGORIES = ["bottoms", "dresses", "shoes", "tops"]

# All footwear is included, including sandals and flip-flops.
# These keys are lowercase so capitalization in styles.csv does not matter.
CATEGORY_MAP = {
    "tshirts": "tops", "shirts": "tops", "tops": "tops",
    "sweatshirts": "tops", "sweaters": "tops", "kurtas": "tops",
    "tunics": "tops", "blouses": "tops", "hoodies": "tops",
    "jeans": "bottoms", "shorts": "bottoms", "trousers": "bottoms",
    "track pants": "bottoms", "leggings": "bottoms", "capris": "bottoms",
    "skirts": "bottoms", "jeggings": "bottoms",
    "dresses": "dresses", "dress": "dresses",
    "casual shoes": "shoes", "sports shoes": "shoes",
    "formal shoes": "shoes", "sneakers": "shoes",
    "heels": "shoes", "flats": "shoes", "boots": "shoes",
    "sandals": "shoes", "sports sandals": "shoes",
    "flip flops": "shoes", "slippers": "shoes",
    "loafers": "shoes", "wedges": "shoes", "booties": "shoes",
}


def find_files():
    """Find styles.csv and the image folder, including extra extracted folders."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    # Prefer Data/styles.csv, but allow a nested extracted directory.
    csv_files = list(DATA_DIR.rglob("styles.csv"))
    if not csv_files:
        raise FileNotFoundError("No styles.csv found. Extract the Kaggle dataset first.")
    csv_file = DATA_DIR / "styles.csv"
    if not csv_file.is_file():
        csv_file = sorted(csv_files)[0]

    # Look for extracted folders named 'images'.
    image_folders = [p for p in DATA_DIR.rglob("*")
                     if p.is_dir() and p.name.lower() == "images"]
    if not image_folders:
        raise FileNotFoundError("No images folder found. Extract the image ZIP first.")
    return csv_file, image_folders


def main():
    print("Preparing Designer AI dataset...")
    csv_file, image_folders = find_files()

    # Load product IDs as strings, because filenames are also ID strings.
    data = pd.read_csv(csv_file, dtype={"id": str}, on_bad_lines="skip")
    print(f"Original clothing items: {len(data)}")

    # The CSV must include these columns. Color and usage are optional.
    if not {"id", "articleType"}.issubset(data.columns):
        raise ValueError("styles.csv must contain 'id' and 'articleType'.")
    for column in ["baseColour", "usage"]:
        if column not in data.columns:
            data[column] = ""

    # Remove blank IDs and unknown clothing types.
    data = data.dropna(subset=["id", "articleType"]).copy()
    data["id"] = data["id"].str.strip()
    data = data[data["id"] != ""].drop_duplicates(subset="id")
    data["category"] = (
        data["articleType"].astype(str).str.strip().str.lower().map(CATEGORY_MAP)
    )
    data = data.dropna(subset="category").copy()
    print(f"Items in our four categories: {len(data)}")

    # Create a lookup table from image filename (without .jpg) to its path.
    # Search ALL extracted 'images' folders to avoid missing any clothing photos.
    images = {}
    for folder in image_folders:
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                images.setdefault(path.stem, path)
    print(f"Images found: {len(images)}")

    records = []
    missing = 0
    unreadable = 0

    # Match every labeled item to an image and verify that it opens.
    for _, item in data.iterrows():
        item_id = str(item["id"])
        path = images.get(item_id)
        if path is None:
            missing += 1
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError):
            unreadable += 1
            continue

        # These column names match train_model.py and modelEvaluation.py.
        records.append({
            "id": item_id,
            "image_path": path.relative_to(PROJECT_DIR).as_posix(),
            "category": item["category"],
            "article_type": item["articleType"],
            "color": "" if pd.isna(item["baseColour"]) else str(item["baseColour"]),
            "usage": "" if pd.isna(item["usage"]) else str(item["usage"]),
        })

    prepared = pd.DataFrame(records)
    if prepared.empty:
        raise ValueError("No matching images found. Check Data/images and styles.csv.")

    # Display how many items we have for each of the four categories.
    counts = prepared["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    print(f"\nUsable clothing images: {len(prepared)}")
    print("\nImages by category:")
    print(counts.to_string())

    # Stratified splitting needs enough examples of every category.
    if (counts < 10).any():
        raise ValueError("Each category needs at least 10 usable images to split.")

    # 70% training / 15% validation / 15% testing.
    # 'stratify' preserves approximately the same category proportions.
    train, remaining = train_test_split(
        prepared, test_size=0.30, random_state=SEED,
        stratify=prepared["category"]
    )
    validation, test = train_test_split(
        remaining, test_size=0.50, random_state=SEED,
        stratify=remaining["category"]
    )

    # Save three CSV files for training and evaluation.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        split.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(f"{name.capitalize()} images: {len(split)}")

    print(f"\nMissing images: {missing}")
    print(f"Unreadable images: {unreadable}")
    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
