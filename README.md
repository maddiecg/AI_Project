# AI_Project

## Project Description:
We propose developing an AI-powered clothing classification and outfit recommendation system. Given a user-uploaded clothing image, a computer vision model will identify the primary clothing category, focusing on tops, bottoms, jackets, and dresses. We will fine-tune a pretrained vision model, such as a Vision Transformer (ViT), using a labeled fashion dataset.
After classification, the system will extract additional clothing characteristics such as color and style. A fashion-specific vision model such as FashionCLIP may be used to help identify style-related attributes. The recommendation system will then combine these attributes with predefined fashion principles, including color compatibility, structural style, and clothing-category compatibility.
A locally hosted large language model (LLM), such as Qwen2.5-7B-Instruct, will be incorporated into the recommendation process. Rather than allowing the LLM to independently determine what clothing combinations are appropriate, it will receive the results of the computer vision model and the project's fashion rules and generate several outfit suggestions with short explanations. This creates a hybrid system combining machine learning, computer vision, rule-based reasoning, and natural language generation.

## How to Run Designer AI

**Requirements:** Python 3.12 and Git.

**1. Clone the repository:**
```bash
git clone https://github.com/maddiecg/AI_Project.git
cd AI_Project
```

**2. Install dependencies:**
```bash
python -m pip install -r requirements.txt
```

**3. Start the application:**
```bash
python -m uvicorn Backend.API.routes:app --reload
```

**4. Open the website:**

Visit http://127.0.0.1:8000/ in your browser.

Upload a clothing image to receive an AI-generated clothing category prediction, automatic color detection, and outfit recommendations.

**5. Stop the application:**

Press `Ctrl + C` in the terminal.

**Note:** The trained model is included in the project, so no additional training or dataset download is required.
