"""Explore the prepared Designer AI clothing dataset.

Location: AI_Project/Notebooks/datasetexploration.py
Input:    AI_Project/Data/prepared/{train,validation,test}.csv
Output:   AI_Project/Data/exploration/ (charts and summary)

Run prepare_data.py first to generate the input CSV files.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without requiring a desktop window.
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"
PREPARED_DIR = DATA_DIR / "prepared"
OUTPUT_DIR = DATA_DIR / "exploration"
CATEGORIES = ("tops", "bottoms", "jackets", "dresses")
SPLITS = ("train", "validation", "test")
SEED = 42


def load_data():
    """Read the prepared splits and check for common dataset mistakes."""
    parts = []
    for split in SPLITS:
        path = PREPARED_DIR / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run Notebooks/prepare_data.py first."
            )
        frame = pd.read_csv(path, dtype={"id": str}).fillna("")
        required = {"id", "image_path", "category", "article_type", "color", "usage"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame["split"] = split
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    if data.empty:
        raise ValueError("The prepared dataset is empty.")
    return data


def save_bar_chart(counts, title, xlabel, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel="Number of images")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close(fig)


def make_sample_grid(data, per_category=4):
    """Save example images from every category for visual inspection."""
    fig, axes = plt.subplots(len(CATEGORIES), per_category, figsize=(12, 11))
    for row, category in enumerate(CATEGORIES):
        subset = data[data["category"] == category]
        samples = subset.sample(n=min(per_category, len(subset)), random_state=SEED)
        for col in range(per_category):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(samples):
                continue
            record = samples.iloc[col]
            image_file = PROJECT_DIR / str(record["image_path"])
            try:
                with Image.open(image_file) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    ax.imshow(image)
            except (OSError, ValueError, UnidentifiedImageError):
                ax.text(.5, .5, "Image unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{category}: {record['id']}", fontsize=9)
    fig.suptitle("Sample clothing images", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_images.png", dpi=160)
    plt.close(fig)


def main():
    data = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Total prepared images: {len(data):,}")
    print("\nImages by dataset split:")
    print(data["split"].value_counts().reindex(SPLITS, fill_value=0).to_string())
    print("\nImages by clothing category:")
    category_counts = data["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    print(category_counts.to_string())

    by_split = pd.crosstab(data["split"], data["category"]).reindex(
        index=SPLITS, columns=CATEGORIES, fill_value=0
    )
    print("\nCategory counts by split:")
    print(by_split.to_string())

    print("\nMissing or blank values by column:")
    print(data.apply(lambda col: col.astype(str).str.strip().eq("").sum()).to_string())
    duplicate_ids = int(data["id"].duplicated().sum())
    print(f"\nDuplicate product IDs across splits: {duplicate_ids}")
    unexpected = sorted(set(data["category"]) - set(CATEGORIES))
    if unexpected:
        print(f"WARNING: Unexpected categories: {unexpected}")
    if duplicate_ids:
        print("WARNING: The same product ID occurs more than once. Check your train/test split.")

    # Check that image paths still point to accessible files.
    missing_files = sum(not (PROJECT_DIR / str(path)).is_file() for path in data["image_path"])
    print(f"Missing image files at recorded paths: {missing_files:,}")

    save_bar_chart(category_counts, "Clothing category distribution", "Category", "category_distribution.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    by_split.plot(kind="bar", ax=ax)
    ax.set(title="Categories within each dataset split", xlabel="Split", ylabel="Images")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "split_distribution.png", dpi=160)
    plt.close(fig)

    for column, filename, title in (
        ("article_type", "article_types.png", "Most common clothing types"),
        ("color", "colors.png", "Most common clothing colors"),
        ("usage", "usage.png", "Most common usage labels"),
    ):
        counts = data[column].replace("", "Unknown").value_counts().head(12)
        if not counts.empty:
            save_bar_chart(counts, title, column.replace("_", " ").title(), filename)

    make_sample_grid(data)
    by_split.to_csv(OUTPUT_DIR / "category_counts_by_split.csv")
    summary = [
        f"Total images: {len(data)}",
        f"Duplicate IDs: {duplicate_ids}",
        f"Missing image files: {missing_files}",
        "",
        "Images per split:",
        data["split"].value_counts().reindex(SPLITS, fill_value=0).to_string(),
        "",
        "Images per category:",
        category_counts.to_string(),
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"\nCharts and summary saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
