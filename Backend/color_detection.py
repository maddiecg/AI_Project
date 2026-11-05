"""
Designer AI - Automatic Color Detection

Estimate the dominant color in a clothing photograph.

How it works:
1. Open an image and correct its orientation.
2. Convert to RGB, resize it, and focus on the center.
3. Ignore almost-white pixels when enough other pixels remain.
4. Assign simple color names to pixels and choose the most common.

This is a rule-based estimate, not a clothing segmentation model.
White clothing, busy backgrounds, skin, or patterned items may be misclassified.

Run from the AI_Project-main folder:
    python Backend/color_detection.py "Data/images/12345.jpg"
"""

# ------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------

# Path locates the image file on your computer.
from pathlib import Path

# sys lets us read the image filename entered in the terminal.
import sys

# Counter counts how many pixels have each color name.
from collections import Counter

# colorsys converts RGB color values to HSV values.
import colorsys

# NumPy turns image pixels into a regular numerical array.
import numpy as np

# Pillow opens, rotates, crops, and resizes images.
from PIL import Image, ImageOps


# ------------------------------------------
# STEP 2: GIVE A PIXEL A COLOR NAME
# ------------------------------------------

def identify_color(red: int, green: int, blue: int) -> str:
    """Convert one RGB pixel to an approximate, everyday color name."""

    # RGB channels range from 0 to 255; colorsys expects 0 to 1.
    r = red / 255.0
    g = green / 255.0
    b = blue / 255.0

    # HSV = hue (color), saturation (intensity), value (brightness).
    hue, saturation, brightness = colorsys.rgb_to_hsv(r, g, b)
    hue *= 360.0  # Convert hue to degrees (0 through 360).

    # Detect dark and low-saturation (neutral) colors first.
    if brightness < 0.16:
        return "black"

    if saturation < 0.13:
        if brightness > 0.85:
            return "white"
        if brightness < 0.30:
            return "black"
        return "grey"

    # Distinguish common warm neutral colors.
    if 15 <= hue < 50 and saturation < 0.55:
        return "beige" if brightness > 0.60 else "brown"

    # Assign the remaining pixels to approximate color families.
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange" if brightness > 0.65 else "brown"
    if hue < 70:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 255:
        return "navy" if brightness < 0.45 else "blue"
    if hue < 290:
        return "purple"
    if hue < 345:
        return "pink"

    return "red"


# ------------------------------------------
# STEP 3: FIND THE IMAGE'S MOST COMMON COLOR
# ------------------------------------------

def detect_color(image_path: str | Path) -> str:
    """Estimate the main color of the garment in a given image."""

    # Convert the supplied filename to a Path and check it exists.
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as original:
        # Correct rotation for images taken on phones.
        oriented = ImageOps.exif_transpose(original)

        # Force RGB so that EVERY pixel has red, green, and blue channels.
        image = oriented.convert("RGB")

        # Small images are faster to process than large photographs.
        image.thumbnail((150, 150))
        width, height = image.size

        # Clothing is often centered in product photos, so crop the
        # outer 10 percent on each side to reduce background influence.
        left = int(width * 0.10)
        top = int(height * 0.10)
        right = max(left + 1, int(width * 0.90))
        bottom = max(top + 1, int(height * 0.90))
        image = image.crop((left, top, right, bottom))

        # The RGB image becomes an array shaped (height, width, 3).
        image_array = np.asarray(image, dtype=np.uint8)

        # Flatten the image into rows, with three values per pixel.
        # Shape changes from (height, width, 3) to (pixel_count, 3).
        flat_pixels = image_array.reshape(-1, 3)

        # Convert NumPy values to ordinary Python ints in explicit tuples.
        # This avoids Pillow ImagingCore and getpixel() typing issues.
        pixel_list: list[tuple[int, int, int]] = [
            (int(pixel[0]), int(pixel[1]), int(pixel[2]))
            for pixel in flat_pixels
        ]

    # Try to remove the nearly white background of a product image.
    # Don't ALWAYS discard white, because the clothing itself may be white!
    clothing_pixels: list[tuple[int, int, int]] = []
    for red, green, blue in pixel_list:
        if red > 245 and green > 245 and blue > 245:
            continue
        clothing_pixels.append((red, green, blue))

    # If at least 10% of pixels remain, use the non-white pixels.
    # Otherwise use all pixels, to avoid throwing out a white garment.
    pixels_to_check: list[tuple[int, int, int]]
    if len(clothing_pixels) >= len(pixel_list) * 0.10:
        pixels_to_check = clothing_pixels
    else:
        pixels_to_check = pixel_list

    # Assign a color name to each pixel, then count the names.
    color_counts: Counter[str] = Counter()
    for red, green, blue in pixels_to_check:
        color_name = identify_color(red, green, blue)
        color_counts[color_name] += 1

    # Return the most frequent color name.
    # The fallback only matters for unusual, empty input data.
    if not color_counts:
        return "unknown"
    return color_counts.most_common(1)[0][0]


# ------------------------------------------
# STEP 4: TEST FROM THE TERMINAL
# ------------------------------------------

def main() -> None:
    """Read an image path from the terminal and print its main color."""

    print("\nDESIGNER AI - COLOR DETECTION")
    print("-----------------------------")

    if len(sys.argv) < 2:
        print('Usage: python Backend/color_detection.py "image.jpg"')
        return

    image_path = sys.argv[1]
    color = detect_color(image_path)

    print(f"\nImage: {Path(image_path).name}")
    print(f"Detected color: {color}")
    print("\nColor detection complete!")


# Only run main() when this script is executed directly.
# Other files can import detect_color() without running main().
if __name__ == "__main__":
    main()
