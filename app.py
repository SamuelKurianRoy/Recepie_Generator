from flask import Flask, render_template, request, jsonify, send_from_directory
import random
import json
import pandas as pd
import os
from difflib import get_close_matches, SequenceMatcher
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import time

# Initialize Flask app
app = Flask(__name__)

# Configure the path for food images
FOOD_IMAGES_DIR = r"C:\Users\hai\Desktop\Recepie Generator\Food Images"

# Global variables for recipe data and indexes
RECIPES = {}
ALL_RECIPES = []
INGREDIENT_INDEX = defaultdict(list)  # Inverted index: ingredient -> list of recipe indices
INGREDIENT_VARIATIONS = defaultdict(set)  # Map of ingredient variations
VECTORIZER = TfidfVectorizer(stop_words='english')
RECIPE_VECTORS = None

# Download required NLTK data with explicit error handling
try:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=False)
    nltk.download('averaged_perceptron_tagger', quiet=False)
    nltk.download('wordnet', quiet=False)
    nltk.download('omw-1.4', quiet=False)
    print("NLTK data download completed")
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

# Initialize NLP components with fallback
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer(stop_words='english')

# Global flag for WordNet availability
WORDNET_AVAILABLE = False
try:
    # Test WordNet functionality
    test_synsets = wordnet.synsets('test')
    if test_synsets:
        WORDNET_AVAILABLE = True
    print(f"WordNet functionality is {'available' if WORDNET_AVAILABLE else 'not available'}")
except Exception as e:
    print(f"WordNet initialization failed: {str(e)}")

class IngredientProcessor:
    def __init__(self):
        self.common_units = {'cup', 'tablespoon', 'teaspoon', 'pound', 'ounce', 'gram', 'kg', 'ml', 'g', 'oz', 'lb', 'tbsp', 'tsp'}
        self.measurement_pattern = re.compile(r'(\d+(?:/\d+)?(?:\s*\d+(?:/\d+))?(?:\s*-\s*\d+(?:/\d+)?)?)\s*([a-zA-Z]+)')
        self.ingredient_vectors = {}

    def preprocess_ingredient(self, ingredient):
        """Clean and normalize ingredient text"""
        # Convert to lowercase and remove extra whitespace
        ingredient = ingredient.lower().strip()
        
        # Remove quantities and units
        ingredient = self.remove_measurements(ingredient)
        
        # Simple word splitting instead of complex tokenization
        words = ingredient.split()
        
        # Only attempt lemmatization if WordNet is available
        if WORDNET_AVAILABLE:
            try:
                words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
            except Exception:
                # Fallback to original words if lemmatization fails
                words = [word for word in words if word.isalnum()]
        else:
            words = [word for word in words if word.isalnum()]
        
        return ' '.join(words)

    def remove_measurements(self, text):
        """Remove measurements and units from ingredient text"""
        # Remove numeric values and common units
        text = self.measurement_pattern.sub('', text)
        words = text.split()
        return ' '.join(word for word in words if word not in self.common_units)

    def calculate_text_similarity(self, str1, str2):
        """Calculate similarity between strings using multiple text-based methods"""
        # Convert to lowercase for comparison
        str1_lower = str1.lower()
        str2_lower = str2.lower()
        
        # 1. Exact match check
        if str1_lower == str2_lower:
            return 1.0
            
        # 2. Substring check
        if str1_lower in str2_lower or str2_lower in str1_lower:
            return 0.9
        
        # 3. Token overlap similarity
        tokens1 = set(str1_lower.split())
        tokens2 = set(str2_lower.split())
        if tokens1 and tokens2:
            overlap_sim = len(tokens1.intersection(tokens2)) / max(len(tokens1), len(tokens2))
        else:
            overlap_sim = 0
            
        # 4. Sequence similarity using SequenceMatcher
        sequence_sim = SequenceMatcher(None, str1_lower, str2_lower).ratio()
        
        # 5. Approximate string matching using get_close_matches
        close_match_sim = 0
        if get_close_matches(str1_lower, [str2_lower], n=1, cutoff=0.0):
            close_match_sim = len(get_close_matches(str1_lower, [str2_lower], n=1, cutoff=0.0)[0]) / max(len(str1_lower), len(str2_lower))
        
        # Combine similarities with weights
        return (0.4 * overlap_sim + 0.3 * sequence_sim + 0.3 * close_match_sim)

    def calculate_ingredient_similarity(self, ing1, ing2):
        """Calculate overall similarity between two ingredients"""
        # Preprocess ingredients
        proc_ing1 = self.preprocess_ingredient(ing1)
        proc_ing2 = self.preprocess_ingredient(ing2)
        
        # Calculate text-based similarity
        text_sim = self.calculate_text_similarity(proc_ing1, proc_ing2)
        
        # If WordNet is available, try to enhance the similarity score
        if WORDNET_AVAILABLE:
            try:
                synsets1 = wordnet.synsets(proc_ing1, pos=wordnet.NOUN)
                synsets2 = wordnet.synsets(proc_ing2, pos=wordnet.NOUN)
                
                if synsets1 and synsets2:
                    # Get max similarity between any pair of synsets
                    max_sim = 0
                    for syn1 in synsets1:
                        for syn2 in synsets2:
                            try:
                                sim = syn1.path_similarity(syn2)
                                if sim and sim > max_sim:
                                    max_sim = sim
                            except:
                                continue
                    
                    if max_sim > 0:
                        # Blend WordNet similarity with text similarity
                        return (0.7 * text_sim + 0.3 * max_sim)
            except Exception:
                pass
        
        # Return text-based similarity if WordNet enhancement failed or isn't available
        return text_sim

def clean_text(value):
    """Convert any value to string and handle None/NaN cases"""
    if pd.isna(value) or value is None:
        return ''
    return str(value).strip()

def split_and_clean(text, separator):
    """Split text by separator and clean the results"""
    if pd.isna(text) or text is None or text == '':
        return []
    text = clean_text(text)
    return [item.strip().lower() for item in text.split(separator) if item.strip()]

ingredient_processor = IngredientProcessor()

def calculate_recipe_match(recipe_ingredients, user_ingredients):
    """Calculate how well a recipe matches with user's ingredients"""
    matches = []
    match_scores = []
    
    print(f"\nMatching ingredients for recipe with {len(recipe_ingredients)} ingredients")
    print(f"User ingredients: {user_ingredients}")
    
    for user_ing in user_ingredients:
        best_match_score = 0
        best_match_ingredient = None
        
        print(f"\nTrying to match user ingredient: {user_ing}")
        
        for recipe_ing in recipe_ingredients:
            try:
                similarity = ingredient_processor.calculate_ingredient_similarity(user_ing, recipe_ing)
                print(f"  Comparing with '{recipe_ing}': similarity = {similarity:.2f}")
                if similarity > best_match_score and similarity > 0.4:  # Lowered threshold for better matching
                    best_match_score = similarity
                    best_match_ingredient = recipe_ing
            except Exception as e:
                print(f"Error comparing ingredients '{user_ing}' and '{recipe_ing}': {str(e)}")
                continue
        
        if best_match_ingredient:
            matches.append((user_ing, best_match_ingredient, best_match_score * 100))
            match_scores.append(best_match_score)
            print(f"  Best match found: '{best_match_ingredient}' with score {best_match_score:.2f}")
    
    # Calculate overall match percentage
    if not recipe_ingredients:
        return 0, []
    
    # Adjust the scoring algorithm
    if matches:
        # Calculate scores
        match_count_score = len(matches) / len(user_ingredients) * 100  # Changed to user ingredients count
        match_quality_score = (sum(match_scores) / len(matches)) * 100
        
        # Weighted average with more emphasis on the number of matches
        final_score = (0.8 * match_count_score + 0.2 * match_quality_score)
        
        print(f"\nMatch summary:")
        print(f"Match count score: {match_count_score:.1f}")
        print(f"Match quality score: {match_quality_score:.1f}")
        print(f"Final score: {final_score:.1f}")
    else:
        final_score = 0
        print("\nNo matches found")
    
    # Return matched ingredients with their similarity scores
    matched_ingredients = [f"{user_ing} ({score:.0f}% match with {recipe_ing})"
                         for user_ing, recipe_ing, score in matches]
    
    return final_score, matched_ingredients

def get_image_path(image_name):
    """Get the full image path with .jpg extension"""
    if not image_name:
        return None
    image_name = f"{clean_text(image_name)}.jpg"
    potential_image = os.path.join(FOOD_IMAGES_DIR, image_name)
    return image_name if os.path.exists(potential_image) else None

def search_recipes_by_name(query, recipes, max_results=5):
    """Search for recipes by name using NLTK and text similarity"""
    query = query.lower()
    matches = []
    
    # Preprocess query with error handling
    try:
        query_tokens = set(word_tokenize(query))
        query_lemmas = {lemmatizer.lemmatize(token) for token in query_tokens}
    except Exception as e:
        print(f"Error processing query '{query}': {str(e)}")
        query_lemmas = set(query.split())  # Fallback to simple splitting
    
    for recipe in recipes:
        recipe_name = recipe['name'].lower()
        
        try:
            recipe_tokens = set(word_tokenize(recipe_name))
            recipe_lemmas = {lemmatizer.lemmatize(token) for token in recipe_tokens}
        except Exception as e:
            print(f"Error processing recipe name '{recipe_name}': {str(e)}")
            recipe_lemmas = set(recipe_name.split())  # Fallback to simple splitting
        
        # Calculate similarity score using multiple methods
        # 1. Exact and partial string matching
        exact_match = query == recipe_name
        partial_match = query in recipe_name
        
        # 2. Token overlap similarity with error handling
        try:
            token_overlap = len(query_lemmas.intersection(recipe_lemmas)) / len(query_lemmas) if query_lemmas else 0
        except Exception as e:
            print(f"Error calculating token overlap: {str(e)}")
            token_overlap = 0
        
        # 3. Sequence similarity using difflib
        sequence_sim = 0
        try:
            if get_close_matches(query, [recipe_name], n=1, cutoff=0.6):
                sequence_sim = 1
        except Exception as e:
            print(f"Error calculating sequence similarity: {str(e)}")
        
        # Calculate final score
        if exact_match:
            score = 100
        elif partial_match:
            score = 90
        else:
            score = (token_overlap * 60) + (sequence_sim * 40)
        
        if score > 20:  # Minimum threshold
            matches.append((score, recipe))
    
    # Sort by score and return top matches
    matches.sort(reverse=True, key=lambda x: x[0])
    return [recipe for score, recipe in matches[:max_results]]

def load_recipes_from_csv():
    csv_path = "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return {}, []
    
    print(f"Loading CSV file from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV file loaded successfully. Found {len(df)} recipes")
    
    # Print column names to debug
    print("CSV columns:", df.columns.tolist())
    
    recipes_dict = {}
    all_recipes = []
    
    # Process each row in the DataFrame
    for index, row in df.iterrows():
        try:
            # Get the category, defaulting to 'Main Dish' if not specified
            category = clean_text(row.get('Category', row.get('category', 'Main Dish'))).lower()
            if not category:
                category = 'main dish'
            
            # Get the title/name of the recipe
            name = clean_text(row.get('Title', row.get('title', row.get('Name', row.get('name', '')))))
            if not name:
                continue  # Skip recipes without names
                
            # Get ingredients
            ingredients_text = row.get('Ingredients', row.get('ingredients', ''))
            if pd.isna(ingredients_text):
                continue  # Skip recipes without ingredients
                
            ingredients = split_and_clean(ingredients_text, ',')
            if not ingredients:
                continue  # Skip recipes without ingredients
                
            # Get instructions
            instructions_text = row.get('Instructions', row.get('instructions', row.get('Steps', row.get('steps', ''))))
            instructions = split_and_clean(str(instructions_text), '.')
            
            # Get image name and path
            image_name = row.get('Image_Name', '')
            image_path = get_image_path(image_name)
            
            recipe = {
                "name": name,
                "ingredients": ingredients,
                "instructions": instructions,
                "cooking_time": clean_text(row.get('CookingTime', row.get('Time', 'Not specified'))),
                "difficulty": clean_text(row.get('Difficulty', 'Medium')),
                "category": category,
                "image": image_path
            }
            
            if category not in recipes_dict:
                recipes_dict[category] = []
            recipes_dict[category].append(recipe)
            all_recipes.append(recipe)
            
            if index == 0:  # Print first recipe for debugging
                print("\nFirst recipe loaded:")
                print(f"Name: {recipe['name']}")
                print(f"Category: {recipe['category']}")
                print(f"Ingredients count: {len(recipe['ingredients'])}")
                print(f"Image: {recipe['image']}")
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            continue
    
    print(f"\nLoaded {len(all_recipes)} total recipes")
    print(f"Categories found: {list(recipes_dict.keys())}")
    return recipes_dict, all_recipes

def build_ingredient_index():
    """Build an inverted index of ingredients and their variations"""
    global INGREDIENT_INDEX, INGREDIENT_VARIATIONS, RECIPE_VECTORS
    
    print("Building ingredient index...")
    start_time = time.time()
    
    # Clear existing indexes
    INGREDIENT_INDEX.clear()
    INGREDIENT_VARIATIONS.clear()
    
    # Process each recipe
    all_ingredients = []
    for recipe_idx, recipe in enumerate(ALL_RECIPES):
        recipe_ingredients = set()
        ingredients_text = " ".join(recipe['ingredients'])
        all_ingredients.append(ingredients_text)
        
        for ing in recipe['ingredients']:
            # Clean and normalize ingredient
            clean_ing = ingredient_processor.preprocess_ingredient(ing)
            words = clean_ing.split()
            
            # Add to ingredient variations
            INGREDIENT_VARIATIONS[clean_ing].add(ing)
            for word in words:
                INGREDIENT_VARIATIONS[word].add(ing)
            
            # Add to inverted index
            INGREDIENT_INDEX[clean_ing].append(recipe_idx)
            for word in words:
                INGREDIENT_INDEX[word].append(recipe_idx)
            
            recipe_ingredients.add(clean_ing)
    
    # Create TF-IDF vectors for recipes
    try:
        RECIPE_VECTORS = VECTORIZER.fit_transform(all_ingredients)
    except Exception as e:
        print(f"Error creating recipe vectors: {str(e)}")
        RECIPE_VECTORS = None
    
    print(f"Index built in {time.time() - start_time:.2f} seconds")
    print(f"Indexed {len(INGREDIENT_INDEX)} unique ingredients")
    print(f"Created {len(INGREDIENT_VARIATIONS)} ingredient variations")

def find_matching_recipes(user_ingredients, max_results=5):
    """Find recipes matching user ingredients using the inverted index"""
    start_time = time.time()
    
    # Clean and normalize user ingredients
    clean_user_ingredients = [ingredient_processor.preprocess_ingredient(ing) for ing in user_ingredients]
    
    # Find candidate recipes using the inverted index
    candidate_recipes = defaultdict(float)
    
    for user_ing in clean_user_ingredients:
        # Split ingredient into words for partial matching
        words = user_ing.split()
        
        # Find exact and partial matches
        matched_recipes = set()
        for word in words:
            # Get recipes containing this ingredient word
            recipe_indices = INGREDIENT_INDEX.get(word, [])
            for idx in recipe_indices:
                if idx not in matched_recipes:
                    matched_recipes.add(idx)
                    recipe = ALL_RECIPES[idx]
                    
                    # Calculate match score
                    match_score = calculate_quick_match_score(user_ingredients, recipe['ingredients'])
                    if match_score > 0.2:  # Minimum threshold
                        candidate_recipes[idx] = max(candidate_recipes[idx], match_score)
    
    # Sort candidates by score
    sorted_candidates = sorted(candidate_recipes.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare results
    results = []
    for idx, score in sorted_candidates[:max_results]:
        recipe = ALL_RECIPES[idx]
        recipe_copy = recipe.copy()
        recipe_copy['match_percentage'] = round(score * 100, 1)
        recipe_copy['matched_ingredients'] = find_matching_ingredients(user_ingredients, recipe['ingredients'])
        results.append(recipe_copy)
    
    print(f"Search completed in {time.time() - start_time:.3f} seconds")
    print(f"Found {len(results)} matches out of {len(candidate_recipes)} candidates")
    
    return results

def calculate_quick_match_score(user_ingredients, recipe_ingredients):
    """Calculate a quick similarity score between ingredient lists"""
    user_set = {ingredient_processor.preprocess_ingredient(ing) for ing in user_ingredients}
    recipe_set = {ingredient_processor.preprocess_ingredient(ing) for ing in recipe_ingredients}
    
    # Calculate Jaccard similarity
    intersection = len(user_set.intersection(recipe_set))
    union = len(user_set.union(recipe_set))
    
    # Calculate scores
    match_ratio = intersection / len(user_set) if user_set else 0
    quality_score = intersection / union if union else 0
    
    # Weighted combination
    return 0.8 * match_ratio + 0.2 * quality_score

def find_matching_ingredients(user_ingredients, recipe_ingredients):
    """Find which user ingredients match with recipe ingredients"""
    matches = []
    
    for user_ing in user_ingredients:
        clean_user_ing = ingredient_processor.preprocess_ingredient(user_ing)
        best_match = None
        best_score = 0
        
        for recipe_ing in recipe_ingredients:
            clean_recipe_ing = ingredient_processor.preprocess_ingredient(recipe_ing)
            score = ingredient_processor.calculate_text_similarity(clean_user_ing, clean_recipe_ing)
            
            if score > best_score and score > 0.4:
                best_score = score
                best_match = recipe_ing
        
        if best_match:
            matches.append(f"{user_ing} ({int(best_score * 100)}% match with {best_match})")
    
    return matches

@app.route('/')
def home():
    global RECIPES, ALL_RECIPES
    RECIPES, ALL_RECIPES = load_recipes_from_csv()
    # Build the ingredient index when loading recipes
    build_ingredient_index()
    categories = list(RECIPES.keys())
    print(f"Categories available on home page: {categories}")  # Debug print
    return render_template('index.html', categories=categories)

@app.route('/food_image/<path:filename>')
def food_image(filename):
    return send_from_directory(FOOD_IMAGES_DIR, filename)

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    if request.form.get('dish_name'):
        # Search by dish name
        query = request.form.get('dish_name').strip()
        if not query:
            return jsonify({"error": "Please enter a dish name"}), 400
            
        matching_recipes = search_recipes_by_name(query, ALL_RECIPES)
        if matching_recipes:
            return jsonify({
                "type": "name_match",
                "recipes": matching_recipes
            })
        else:
            return jsonify({"error": "No matching recipes found"}), 404
            
    elif request.form.get('ingredients'):
        # Get user ingredients and find matching recipes
        user_ingredients_text = request.form.get('ingredients', '').strip()
        print(f"\nReceived ingredients search request: '{user_ingredients_text}'")
        
        user_ingredients = split_and_clean(user_ingredients_text, ',')
        if not user_ingredients:
            return jsonify({"error": "Please provide some ingredients"}), 400
        
        # Use optimized search
        matching_recipes = find_matching_recipes(user_ingredients)
        
        if matching_recipes:
            return jsonify({
                "type": "ingredient_match",
                "recipes": matching_recipes
            })
        else:
            return jsonify({"error": "No matching recipes found"}), 404
    
    else:
        # Original category-based random recipe
        category = request.form.get('category', '').lower()
        if category in RECIPES and RECIPES[category]:
            recipe = random.choice(RECIPES[category])
            return jsonify({"type": "category_match", "recipe": recipe})
        return jsonify({"error": "Category not found"}), 404

@app.route('/all_recipes')
def all_recipes():
    global RECIPES
    if not RECIPES:
        RECIPES, _ = load_recipes_from_csv()
    return render_template('all_recipes.html', recipes=RECIPES)

if __name__ == '__main__':
    print("Starting Recipe Generator...")
    # Load recipes at startup
    RECIPES, ALL_RECIPES = load_recipes_from_csv()
    app.run(debug=True)
