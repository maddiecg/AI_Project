# AI_Project

## Project Description:
We propose developing an AI-powered clothing classification and outfit recommendation system. Given a user-uploaded clothing image, a computer vision model will identify the primary clothing category, focusing on tops, bottoms, jackets, and dresses. We will fine-tune a pretrained vision model, such as a Vision Transformer (ViT), using a labeled fashion dataset.
After classification, the system will extract additional clothing characteristics such as color and style. A fashion-specific vision model such as FashionCLIP may be used to help identify style-related attributes. The recommendation system will then combine these attributes with predefined fashion principles, including color compatibility, structural style, and clothing-category compatibility.
A locally hosted large language model (LLM), such as Qwen2.5-7B-Instruct, will be incorporated into the recommendation process. Rather than allowing the LLM to independently determine what clothing combinations are appropriate, it will receive the results of the computer vision model and the project's fashion rules and generate several outfit suggestions with short explanations. This creates a hybrid system combining machine learning, computer vision, rule-based reasoning, and natural language generation.

## How To Make Model Work:
-Download and extract both Styles.csv.zip and Images.zip from the folder

## What We Have Done:
- Set up a github repo: create a backend folder, data folder, frontend folder, notebooks folder, test folders, and prompts.txt
- research AI Model
- Upload the Fashion Product Images (Small) dataset for images
- Tops, bottoms, jackets, and dresses
- Decided on a Y2K Retro Pixel design
- uploaded Styles.zip: the images are numbered and the styles connects that numbered image to what it is. Maps the products from what style it is to the image itself. Categorization.
- Images.zip: images of items from dataset.


## What We Plan To Do:
- look into QWEN and reconvene :) 
- Make a file that will clean code
- Make a file that will fine tune models
- Make a file that predicts the item's category
- Make a file (or build on one) that identifies an item's color and style
- Make a file that suggests combinations
- Make a LLM recommender that sends the combinations to QWEN and generates explanations
- Make a pipeline that connects everything together
- Make a app. file that will do front and backend work. 
