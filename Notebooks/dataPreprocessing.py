"""Designer AI: clean the clothing information in styles.csv.

Run from the main project folder:
    python Notebooks/data_processing.py

This is an OPTIONAL reporting step. The training scripts use prepare_data.py.
"""

from pathlib import Path
import pandas as pd

# This file stays in Notebooks. The parent of Notebooks is AI_Project.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"

# Input and output file locations.
CSV_FILE = DATA_DIR / "styles.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "clothing_data.csv"

# Translate the Kaggle articleType names into our FOUR AI categories.
# Shoes includes all footwear: sneakers, boots, sandals, and flip-flops.
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


def main():
    # The CSV must be extracted from the Kaggle ZIP first.
    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"Cannot find {CSV_FILE}. Extract styles.csv into Data/.")

    # Read IDs as text to avoid changing filenames (such as 12345.jpg).
    data = pd.read_csv(CSV_FILE, dtype={"id": str}, on_bad_lines="skip")
    print(f"Original records: {len(data)}")

    # These two columns are necessary to classify each item.
    if not {"id", "articleType"}.issubset(data.columns):
        raise ValueError("styles.csv must contain 'id' and 'articleType'.")

    # Remove rows without an ID or clothing type, and remove duplicates.
    data = data.dropna(subset=["id", "articleType"]).copy()
    data["id"] = data["id"].str.strip()
    data = data[data["id"] != ""]
    data = data.drop_duplicates(subset="id")

    # Map each original article type to one of our four classes.
    data["category"] = (
        data["articleType"].astype(str).str.strip().str.lower().map(CATEGORY_MAP)
    )
    data = data.dropna(subset="category").copy()

    # Keep relevant descriptive columns if they exist in the dataset.
    columns = ["id", "articleType", "category", "baseColour",
               "season", "usage", "productDisplayName"]
    data = data[[name for name in columns if name in data.columns]]

    # Save the cleaned metadata. This does NOT create train/test splits.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"\nCleaned records: {len(data)}")
    print(data["category"].value_counts().to_string())
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
