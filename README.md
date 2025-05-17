# Recipe Generator - Streamlit App

A Streamlit-based recipe generator application that helps you find recipes based on recipe names, ingredients, or categories. The app uses machine learning and natural language processing to provide intelligent recipe matching.

## Features

- Search recipes by name
- Find recipes based on available ingredients
- Browse recipes by category
- View recipe details including ingredients, instructions, and images
- Smart ingredient matching using NLP techniques

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the recipe model:
   ```bash
   python train_model.py
   ```
   This will create a `recipe_model.pkl` file containing the trained model.

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Data Requirements

The application expects:
1. A CSV file named "Food Ingredients and Recipe Dataset with Image Name Mapping.csv" containing recipe data
2. A folder named "Food Images" containing the recipe images (with .jpg extension)

## Usage

1. **Search by Name**: Enter a recipe name to find matching recipes
2. **Search by Ingredients**: Enter your available ingredients (one per line) to find recipes you can make
3. **Browse Categories**: Select a category to view all recipes in that category

Each recipe will show:
- Recipe name
- List of ingredients
- Cooking instructions
- Recipe image (if available)

## Notes

- The model is trained on first run and cached for subsequent uses
- Recipe matching uses TF-IDF vectorization and cosine similarity for ingredient matching
- NLTK is used for text processing and ingredient matching 