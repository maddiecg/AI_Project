
"""
Designer AI - Dataset Exploration

This file explores our clothing dataset before model training.

It will:
1. Load training, validation, and testing data.
2. Count items in each clothing category.
3. Check for duplicate IDs and missing images.
4. Create graphs showing our clothing data.
5. Save sample images from each category.
6. Create a summary report.

Run prepare_data.py before running this file.
"""

# ------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------

# csv allows us to read CSV files.
import csv

# Counter helps count clothing categories.
from collections import Counter

# Path helps us find files and folders.
from pathlib import Path

# Matplotlib creates our graphs.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Pillow opens clothing images.
from PIL import Image


# ------------------------------------------
# STEP 2: SET UP FILE PATHS
# ------------------------------------------

# Find the main AI_Project folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Find our Data folder.
DATA_DIR = PROJECT_DIR / "Data"

if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"

# Prepared datasets created by prepare_data.py.
PREPARED_DIR = DATA_DIR / "prepared"

# Folder where we save charts and results.
OUTPUT_DIR = DATA_DIR / "exploration"

# These categories match our updated AI model.
CATEGORIES = ["bottoms", "dresses", "shoes", "tops"]

# Our three dataset splits.
SPLITS = ["train", "validation", "test"]

# Columns needed from our CSV files.
COLUMNS = [
    "id",
    "image_path",
    "category",
    "article_type",
    "color",
    "usage"
]


# ------------------------------------------
# STEP 3: LOAD OUR DATA
# ------------------------------------------

def load_data():
    """Load our three prepared CSV files."""

    records = []

    # Read training, validation, and testing files.
    for split in SPLITS:

        csv_file = PREPARED_DIR / f"{split}.csv"

        # Check that the CSV exists.
        if not csv_file.is_file():
            raise FileNotFoundError(
                f"Cannot find {csv_file}. "
                "Run prepare_data.py first."
            )

        # Open the CSV file.
        with csv_file.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            # Check for required columns.
            headers = reader.fieldnames or []

            missing_columns = [
                name for name in COLUMNS
                if name not in headers
            ]

            if missing_columns:
                raise ValueError(
                    f"{csv_file.name} is missing: "
                    f"{missing_columns}"
                )

            # Read each clothing item.
            for row in reader:

                # Skip malformed rows.
                if None in row or any(
                    row.get(name) is None
                    for name in COLUMNS
                ):
                    continue

                # Store useful information.
                item = {
                    name: (row.get(name) or "").strip()
                    for name in COLUMNS
                }

                # Remember the dataset split.
                item["split"] = split

                records.append(item)

    # Make sure we found usable data.
    if not records:
        raise ValueError(
            "No prepared clothing data was found."
        )

    return records


# ------------------------------------------
# STEP 4: CREATE GRAPHS
# ------------------------------------------

def save_chart(counts, title, filename):
    """Create and save a bar chart."""

    # Get labels and their corresponding counts.
    labels = list(counts.keys())
    values = list(counts.values())

    # Create the graph.
    plt.figure(figsize=(10, 5))

    # Draw the bars.
    plt.bar(labels, values)

    # Add titles and labels.
    plt.title(title)
    plt.ylabel("Number of images")

    # Rotate labels so they fit.
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    # Save the graph.
    plt.savefig(OUTPUT_DIR / filename)

    # Close it to free memory.
    plt.close()


# ------------------------------------------
# STEP 5: CREATE SAMPLE IMAGES
# ------------------------------------------

def save_sample_images(records):
    """Save four example images from each category."""

    # Create a grid with four rows and columns.
    fig, axes = plt.subplots(
        4, 4,
        figsize=(12, 12)
    )

    # Each row represents a clothing category.
    for row_number, category in enumerate(CATEGORIES):

        # Find images belonging to this category.
        examples = [
            item for item in records
            if item["category"] == category
        ][:4]

        # Display up to four examples.
        for column_number in range(4):

            ax = axes[row_number, column_number]

            # Hide the axes.
            ax.axis("off")

            # Skip if there are fewer than four images.
            if column_number >= len(examples):
                continue

            item = examples[column_number]

            # Find the clothing image.
            image_path = (
                PROJECT_DIR / item["image_path"]
            )

            try:
                # Open and display the image.
                with Image.open(image_path) as image:
                    ax.imshow(image.convert("RGB"))

            except (OSError, ValueError):
                # Show a message if the image fails.
                ax.text(
                    0.5, 0.5,
                    "Image unavailable",
                    ha="center",
                    va="center"
                )

            # Label each image.
            ax.set_title(
                f"{category}: {item['id']}",
                fontsize=9
            )

    fig.suptitle(
        "Designer AI - Clothing and Footwear"
    )

    fig.tight_layout()

    # Save the sample image grid.
    fig.savefig(
        OUTPUT_DIR / "sample_images.png"
    )

    plt.close(fig)


# ------------------------------------------
# STEP 6: EXPLORE OUR DATASET
# ------------------------------------------

def main():

    print("\nDESIGNER AI - DATASET EXPLORATION")

    # Load our prepared data.
    records = load_data()

    # Create the output folder.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------
    # COUNT CLOTHING CATEGORIES
    # --------------------------------------

    # Count how many images belong to each class.
    counts = Counter(
        item["category"] for item in records
    )

    category_counts = {
        name: counts[name]
        for name in CATEGORIES
    }

    print(f"\nTotal images: {len(records)}")

    print("\nImages by category:")

    for name, count in category_counts.items():
        print(f"{name}: {count}")

    # --------------------------------------
    # COUNT DATASET SPLITS
    # --------------------------------------

    split_counter = Counter(
        item["split"] for item in records
    )

    split_counts = {
        name: split_counter[name]
        for name in SPLITS
    }

    print("\nImages by split:")

    for name, count in split_counts.items():
        print(f"{name}: {count}")

    # --------------------------------------
    # CHECK DUPLICATE IDS
    # --------------------------------------

    # Collect all product IDs.
    ids = [item["id"] for item in records]

    # Compare total IDs with unique IDs.
    duplicate_count = len(ids) - len(set(ids))

    print(
        f"\nDuplicate product IDs: {duplicate_count}"
    )

    # --------------------------------------
    # CHECK MISSING IMAGES
    # --------------------------------------

    missing_images = 0

    # Check whether every image exists.
    for item in records:

        image_path = (
            PROJECT_DIR / item["image_path"]
        )

        if not image_path.is_file():
            missing_images += 1

    print(f"Missing image files: {missing_images}")

    # --------------------------------------
    # CHECK MISSING VALUES
    # --------------------------------------

    # Count empty cells in each important column.
    missing_values = {}

    for name in COLUMNS + ["split"]:

        count = sum(
            not item[name]
            for item in records
        )

        missing_values[name] = count

    print("\nMissing values:")

    for name, count in missing_values.items():
        print(f"{name}: {count}")

    # --------------------------------------
    # CHECK FOR UNEXPECTED CATEGORIES
    # --------------------------------------

    unexpected = (
        set(item["category"] for item in records)
        - set(CATEGORIES)
    )

    if unexpected:
        print(
            f"WARNING: Unexpected categories: "
            f"{sorted(unexpected)}"
        )

    # --------------------------------------
    # CREATE OUR GRAPHS
    # --------------------------------------

    # Graph 1: Clothing categories.
    save_chart(
        category_counts,
        "Clothing Categories",
        "category_distribution.png"
    )

    # Graph 2: Dataset splits.
    save_chart(
        split_counts,
        "Training, Validation, and Testing",
        "split_distribution.png"
    )

    # Graph 3: Most common colors.
    colors = Counter(
        item["color"] or "Unknown"
        for item in records
    )

    save_chart(
        dict(colors.most_common(10)),
        "Most Common Colors",
        "color_distribution.png"
    )

    # Graph 4: Most common clothing types.
    types = Counter(
        item["article_type"] or "Unknown"
        for item in records
    )

    save_chart(
        dict(types.most_common(10)),
        "Most Common Clothing Types",
        "clothing_types.png"
    )

    # Graph 5: Example clothing and footwear.
    save_sample_images(records)

    # --------------------------------------
    # SAVE SUMMARY REPORT
    # --------------------------------------

    summary = [
        f"Total images: {len(records)}",
        f"Duplicate product IDs: {duplicate_count}",
        f"Missing image files: {missing_images}",
        "",
        "Images by category:"
    ]

    # Add category counts to the report.
    for name, count in category_counts.items():
        summary.append(f"{name}: {count}")

    summary.append("")
    summary.append("Images by split:")

    # Add dataset split counts.
    for name, count in split_counts.items():
        summary.append(f"{name}: {count}")

    summary.append("")
    summary.append("Missing values:")

    # Add missing value counts.
    for name, count in missing_values.items():
        summary.append(f"{name}: {count}")

    # Save the summary.
    summary_file = OUTPUT_DIR / "summary.txt"

    summary_file.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("\nDataset exploration complete!")
    print(f"Results saved to: {OUTPUT_DIR}")


# ------------------------------------------
# STEP 7: RUN THE PROGRAM
# ------------------------------------------

if __name__ == "__main__":
    main()
