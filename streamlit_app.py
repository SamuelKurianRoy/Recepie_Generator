import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Recipe Generator",
    page_icon="🍳",
    layout="wide"
)

import pickle
import os
import logging
import base64
from PIL import Image
import io
import numpy as np

# Initialize NLTK first
import nltk

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except Exception as e:
        st.error("⚠️ Error initializing language processing. Basic search will still work.")
        return False

# Only import NLTK components after downloading data
if download_nltk_data():
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from sklearn.metrics.pairwise import cosine_similarity

def preprocess_ingredient(ingredient, lemmatizer=None):
    """Clean and normalize ingredient text"""
    try:
        if lemmatizer:
            ingredient = str(ingredient).lower().strip()
            tokens = word_tokenize(ingredient)
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
            return ' '.join(tokens)
        return ingredient.lower().strip()
    except Exception:
        return ingredient.lower().strip()

def search_by_name(query, model_data):
    """Search recipes by name"""
    query = query.lower()
    matches = []
    
    for idx, recipe in enumerate(model_data['recipes_data']):
        name = recipe['name'].lower()
        if query in name:
            matches.append((idx, recipe))
    
    return matches

def search_by_ingredients(ingredients, model_data, top_k=5):
    """Search recipes by ingredients"""
    try:
        lemmatizer = WordNetLemmatizer() if 'nltk.tokenize.punkt' in nltk.data.path else None
        processed_ingredients = [preprocess_ingredient(ing, lemmatizer) for ing in ingredients]
        
        # Create ingredient vector
        ingredients_text = ' '.join(processed_ingredients)
        ingredient_vector = model_data['vectorizer'].transform([ingredients_text])
        
        # Calculate similarity scores
        similarities = cosine_similarity(ingredient_vector, model_data['recipe_vectors']).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        matches = []
        
        for idx in top_indices:
            recipe = model_data['recipes_data'][idx]
            score = similarities[idx]
            if score > 0:  # Only include matches with positive similarity
                matches.append((idx, recipe, score))
        
        return matches
    except Exception:
        return []

def display_recipe(recipe, score=None):
    """Helper function to display a recipe with consistent formatting"""
    with st.expander(f"📖 {recipe['name']}{f' (Match: {score:.0%})' if score else ''}"):
        # Ingredients section
        st.write("**Ingredients:**")
        for ing in recipe['ingredients']:
            st.write(f"- {ing}")
        
        # Instructions section
        st.write("\n**Instructions:**")
        for i, step in enumerate(recipe['instructions'], 1):
            if step.strip():
                st.write(f"{i}. {step.strip()}")
        
        # Image section
        if recipe.get('image_data'):
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(recipe['image_data'])
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, caption=recipe['name'], use_container_width=True)
            except Exception:
                pass

def main():
    st.title("🍳 Recipe Generator")
    st.write("Find recipes by name or ingredients!")
    
    # Load the model
    model_path = "recipe_model.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    except Exception:
        st.error("⚠️ Unable to load recipe data. Please try again later.")
        st.stop()
    
    # Show model statistics in sidebar
    st.sidebar.title("📊 Recipe Stats")
    total_recipes = len(model_data['recipes_data'])
    total_images = sum(1 for recipe in model_data['recipes_data'] if recipe.get('image_data'))
    st.sidebar.write(f"Total Recipes: {total_recipes}")
    st.sidebar.write(f"Recipes with Images: {total_images}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search by Name", "🥗 Search by Ingredients", "📚 Browse Categories"])
    
    # Tab 1: Search by Name
    with tab1:
        st.header("Search by Recipe Name")
        recipe_name = st.text_input("Enter recipe name:")
        if recipe_name:
            matches = search_by_name(recipe_name, model_data)
            if matches:
                for idx, recipe in matches:
                    display_recipe(recipe)
            else:
                st.warning("No matching recipes found.")
    
    # Tab 2: Search by Ingredients
    with tab2:
        st.header("Search by Ingredients")
        ingredients_input = st.text_area("Enter ingredients (one per line):")
        if ingredients_input:
            ingredients = [ing.strip() for ing in ingredients_input.split('\n') if ing.strip()]
            matches = search_by_ingredients(ingredients, model_data)
            
            if matches:
                for idx, recipe, score in matches:
                    display_recipe(recipe, score)
            else:
                st.warning("No matching recipes found.")
    
    # Tab 3: Browse Categories
    with tab3:
        st.header("Browse by Category")
        categories = set(recipe['category'] for recipe in model_data['recipes_data'])
        selected_category = st.selectbox("Select a category:", sorted(categories))
        
        if selected_category:
            category_recipes = [recipe for recipe in model_data['recipes_data'] 
                              if recipe['category'] == selected_category]
            
            if category_recipes:
                st.write(f"Found {len(category_recipes)} recipes in {selected_category}")
                for recipe in category_recipes:
                    display_recipe(recipe)
            else:
                st.warning(f"No recipes found in category: {selected_category}")

if __name__ == "__main__":
    main() 