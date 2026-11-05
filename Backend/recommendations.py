
"""
Designer AI - Outfit Recommendations

This file recommends clothing pieces that match
an uploaded item.

Our AI recognizes four categories:
- Tops
- Bottoms
- Dresses
- Shoes

How it works:
1. predict.py identifies the clothing category.
2. The user can provide the clothing item's color.
3. This file suggests matching outfit pieces.
4. It explains why the outfit works.

We use simple clothing and color-matching rules.
We do not need to train another AI model.

Example:
python Backend/recommendations.py "Data/images/12345.jpg" blue
"""

# ------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------

# Path helps us work with image filenames.
from pathlib import Path

# sys lets us read arguments from the terminal.
import sys

# Import our existing clothing prediction function.
# Both files are inside the Backend folder.
# Import the clothing prediction function from our Backend folder.
# Import our clothing prediction function.
# Use a different import depending on how this file is run.

if __name__ == "__main__":
    # When running recommendations.py directly.
    from predict import predict_clothing
else:
    # When importing recommendations from another Python file.
    from Backend.predict import predict_clothing
# ------------------------------------------
# STEP 2: DEFINE COLOR MATCHING RULES
# ------------------------------------------

# This dictionary stores colors that generally
# work well together in everyday outfits.
#
# The first color is the clothing item's color.
# The list contains suggested matching colors.

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
}

# Use neutral colors if the user enters
# a color that is not in our dictionary.
DEFAULT_COLORS = ["black", "white", "beige"]


# ------------------------------------------
# STEP 3: FIND MATCHING COLORS
# ------------------------------------------

def get_matching_colors(color):
    """
    Find colors that match an uploaded item.

    Example:
    Input: "blue"
    Output: ["white", "beige", "grey"]
    """

    # Convert the color to lowercase.
    # This lets "Blue" and "blue" work the same.
    color = color.strip().lower()

    # Handle a few common alternative names.
    if color == "navy blue":
        color = "navy"

    if color == "gray":
        color = "grey"

    # Return matching colors.
    # Use default colors if the color is unknown.
    return COLOR_MATCHES.get(
        color,
        DEFAULT_COLORS
    )


# ------------------------------------------
# STEP 4: RECOMMEND OUTFIT PIECES
# ------------------------------------------

def recommend_outfit(category, color):
    """
    Recommend clothing based on category and color.

    Returns a dictionary containing:
    - The uploaded clothing category
    - The uploaded item's color
    - Suggested outfit pieces
    - An explanation
    """

    # Standardize category and color names.
    category = category.strip().lower()
    color = color.strip().lower()

    # Find colors that coordinate with the item.
    colors = get_matching_colors(color)

    # Store our recommendations here.
    recommendations = []

    # --------------------------------------
    # IF THE UPLOADED ITEM IS A TOP
    # --------------------------------------

    if category == "tops":

        # Recommend bottoms and shoes.
        recommendations = [
            f"{colors[0]} pants or jeans",
            f"{colors[1]} sneakers"
        ]

        explanation = (
            "Pairing your top with coordinating bottoms "
            "and shoes creates a balanced casual outfit."
        )

    # --------------------------------------
    # IF THE UPLOADED ITEM IS A BOTTOM
    # --------------------------------------

    elif category == "bottoms":

        # Recommend a shirt and shoes.
        recommendations = [
            f"{colors[0]} shirt or blouse",
            f"{colors[1]} sneakers"
        ]

        explanation = (
            "A coordinating top and neutral shoes "
            "help complete the outfit."
        )

    # --------------------------------------
    # IF THE UPLOADED ITEM IS A DRESS
    # --------------------------------------

    elif category == "dresses":

        # Dresses can be paired with footwear
        # and accessories.
        recommendations = [
            f"{colors[0]} sandals or heels",
            f"{colors[1]} handbag"
        ]

        explanation = (
            "Simple footwear and accessories "
            "complement a dress without overpowering it."
        )

    # --------------------------------------
    # IF THE UPLOADED ITEM IS A SHOE
    # --------------------------------------

    elif category == "shoes":

        # Recommend a top and bottoms.
        recommendations = [
            f"{colors[0]} pants or jeans",
            f"{colors[1]} top"
        ]

        explanation = (
            "Coordinating your clothing with your shoes "
            "helps create a complete outfit."
        )

    # --------------------------------------
    # HANDLE UNKNOWN CATEGORIES
    # --------------------------------------

    else:
        raise ValueError(
            f"Unknown clothing category: {category}"
        )

    # Return all recommendation information.
    return {
        "category": category,
        "color": color,
        "recommendations": recommendations,
        "explanation": explanation
    }


# ------------------------------------------
# STEP 5: CONNECT TO OUR TRAINED AI MODEL
# ------------------------------------------

def recommend_from_image(image_path, color):
    """
    Predict the category of an uploaded image
    and generate outfit recommendations.

    This connects predict.py with our
    recommendation rules.
    """

    # Use our existing trained model to
    # identify the clothing category.
    category, confidence = predict_clothing(
        image_path
    )

    # Generate outfit recommendations using
    # the predicted category and given color.
    outfit = recommend_outfit(
        category,
        color
    )

    # Include the prediction confidence.
    outfit["confidence"] = confidence

    return outfit


# ------------------------------------------
# STEP 6: RUN THE PROGRAM
# ------------------------------------------

def main():
    """Test outfit recommendations in the terminal."""

    print("\nDESIGNER AI - OUTFIT RECOMMENDATIONS")
    print("------------------------------------")

    # We need both an image and a color.
    if len(sys.argv) < 3:
        print(
            'Usage: python Backend/recommendations.py '
            '"image.jpg" "color"'
        )
        return

    # Get the image path entered by the user.
    image_path = sys.argv[1]

    # Get the clothing item's color.
    color = sys.argv[2]

    # Predict the category and recommend an outfit.
    result = recommend_from_image(
        image_path,
        color
    )

    # --------------------------------------
    # DISPLAY THE RESULTS
    # --------------------------------------

    print(f"\nImage: {Path(image_path).name}")

    print(
        f"Predicted category: {result['category']}"
    )

    print(
        f"Prediction confidence: "
        f"{result['confidence']:.2f}%"
    )

    print(f"Item color: {result['color']}")

    print("\nRecommended outfit pieces:")

    # Display each recommended item.
    for item in result["recommendations"]:
        print(f"- {item}")

    # Explain why the outfit works.
    print("\nWhy this outfit works:")
    print(result["explanation"])

    print("\nRecommendation complete!")


# Run the program when executed directly.
if __name__ == "__main__":
    main()
