"""
Designer AI - Outfit Recommendations

This file combines three parts of our project:
1. Clothing classification from predict.py
2. Automatic color detection from color_detection.py
3. Simple outfit suggestions based on the category and color

Run from the main AI_Project-main folder:
    python Backend/recommendations.py "Data/images/12345.jpg"

You can also override the detected color if it is incorrect:
    python Backend/recommendations.py "Data/images/12345.jpg" blue
"""

# ------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------

# Path lets us display the image filename.
from pathlib import Path

# sys lets us read command-line arguments.
import sys


# ------------------------------------------
# STEP 2: DEFINE COLOR MATCHING RULES
# ------------------------------------------

# These are basic styling suggestions, NOT predictions learned by the AI model.
# The dictionary key is the color of the uploaded clothing item.
# The corresponding list contains colors that generally coordinate with it.
COLOR_MATCHES = {
    "black": ["white", "grey", "beige"],
    "white": ["black", "blue", "beige"],
    "blue": ["white", "beige", "grey"],
    "navy": ["white", "beige", "grey"],
    "red": ["black", "white", "blue"],
    "green": ["beige", "white", "black"],
    "pink": ["white", "grey", "beige"],
    "purple": ["white", "black", "grey"],
    "yellow": ["blue", "white", "black"],
    "brown": ["cream", "white", "blue"],
    "beige": ["white", "navy", "brown"],
    "grey": ["black", "white", "blue"],
    "orange": ["navy", "white", "beige"],
    "cream": ["brown", "navy", "black"],
}

# Fall back to neutral suggestions when the detected color is unfamiliar.
DEFAULT_COLORS = ["black", "white", "beige"]


# ------------------------------------------
# STEP 3: FIND COORDINATING COLORS
# ------------------------------------------

def get_matching_colors(color: str) -> list[str]:
    """Return a list of colors that generally match the given color."""

    # Remove extra spaces and make capitalization consistent.
    color = color.strip().lower()

    # Accept common alternate names.
    if color == "navy blue":
        color = "navy"
    elif color == "gray":
        color = "grey"

    # If we don't recognize the color, use neutral colors instead.
    return COLOR_MATCHES.get(color, DEFAULT_COLORS)


# ------------------------------------------
# STEP 4: RECOMMEND CLOTHING PIECES
# ------------------------------------------

def recommend_outfit(category: str, color: str) -> dict:
    """
    Suggest clothing pieces to go with the given item.

    category: tops, bottoms, dresses, or shoes
    color: the provided or automatically detected color

    Returns a dictionary that the website can eventually display.
    """

    # Standardize the inputs so 'Tops' and 'tops' behave the same way.
    category = category.strip().lower()
    color = color.strip().lower()

    # Get matching color choices from our dictionary.
    colors = get_matching_colors(color)

    # Generate different recommendations depending on the clothing type.
    if category == "tops":
        recommendations = [
            f"{colors[0]} pants or jeans",
            f"{colors[1]} sneakers",
        ]
        explanation = (
            "Pairing your top with coordinating bottoms and shoes "
            "creates a balanced casual outfit."
        )

    elif category == "bottoms":
        recommendations = [
            f"{colors[0]} shirt or blouse",
            f"{colors[1]} sneakers",
        ]
        explanation = (
            "A coordinating top and neutral shoes help complete the outfit."
        )

    elif category == "dresses":
        recommendations = [
            f"{colors[0]} sandals or heels",
            f"{colors[1]} handbag",
        ]
        explanation = (
            "Simple footwear and accessories complement a dress "
            "without overpowering it."
        )

    elif category == "shoes":
        recommendations = [
            f"{colors[0]} pants or jeans",
            f"{colors[1]} top",
        ]
        explanation = (
            "Coordinating your clothing with your shoes "
            "helps create a complete outfit."
        )

    else:
        # Only the four categories used by our trained model are supported.
        raise ValueError(f"Unknown clothing category: {category}")

    # Keep all results together so another file (or the website) can use them.
    return {
        "category": category,
        "color": color,
        "recommendations": recommendations,
        "explanation": explanation,
    }


# ------------------------------------------
# STEP 5: CONNECT PREDICTION + COLOR DETECTION
# ------------------------------------------

def recommend_from_image(image_path: str, color: str | None = None) -> dict:
    """
    Classify a clothing image and generate outfit suggestions.

    If color is not supplied, detect it automatically.
    If color IS supplied, use it as a manual override.
    """

    # These imports work whether this file is run directly from the terminal
    # or imported as Backend.recommendations in another Python file.
    # Importing here also avoids loading TensorFlow for simple rule-only tests.
    if __package__:
        from .predict import predict_clothing
        from .color_detection import detect_color
    else:
        from predict import predict_clothing
        from color_detection import detect_color

    # Ask our trained model which clothing category is pictured.
    category, confidence = predict_clothing(image_path)

    # If the user did not type a color, analyze the image's pixels.
    if color is None:
        color = detect_color(image_path)

    # Use the category and color to suggest matching pieces.
    outfit = recommend_outfit(category, color)

    # Save the model's confidence score in the result for display.
    outfit["confidence"] = confidence

    return outfit


# ------------------------------------------
# STEP 6: RUN THE PROGRAM FROM THE TERMINAL
# ------------------------------------------

def main() -> None:
    """Read an image path, make recommendations, and print the results."""

    print("\nDESIGNER AI - OUTFIT RECOMMENDATIONS")
    print("------------------------------------")

    # Only the image path is required; a manual color is optional.
    if len(sys.argv) < 2:
        print(
            'Usage: python Backend/recommendations.py '
            '"image.jpg" [optional_color]'
        )
        return

    # First terminal argument: image location.
    image_path = sys.argv[1]

    # Second argument: optional color override.
    # None tells our program to detect the color from the image.
    color = sys.argv[2] if len(sys.argv) >= 3 else None

    # Run classification, color detection, and recommendation rules.
    result = recommend_from_image(image_path, color)

    # Show the category and confidence predicted by our AI model.
    print(f"\nImage: {Path(image_path).name}")
    print(f"Predicted category: {result['category']}")
    print(f"Prediction confidence: {result['confidence']:.2f}%")

    # Explain whether we detected or manually supplied the color.
    color_label = "Detected color" if color is None else "Provided color"
    print(f"{color_label}: {result['color']}")

    # Print each recommended outfit item on its own line.
    print("\nRecommended outfit pieces:")
    for item in result["recommendations"]:
        print(f"- {item}")

    # Display the explanation for the outfit suggestions.
    print("\nWhy this outfit works:")
    print(result["explanation"])
    print("\nRecommendation complete!")


# Only execute the terminal program when this file is run directly.
# This allows teammates to import its functions without starting main().
if __name__ == "__main__":
    main()

