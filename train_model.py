import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import os
import base64
from PIL import Image
import io

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

class RecipeModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.lemmatizer = WordNetLemmatizer()
        self.recipes_data = None
        self.recipe_vectors = None
        self.ingredient_index = {}
        self.image_data = {}
        
    def preprocess_ingredient(self, ingredient):
        """Clean and normalize ingredient text"""
        ingredient = str(ingredient).lower().strip()
        tokens = word_tokenize(ingredient)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        return ' '.join(tokens)
    
    def load_and_encode_image(self, image_path):
        """Load and encode image to base64"""
        try:
            # Open and resize image to reduce model size
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Resize to reasonable dimensions while maintaining aspect ratio
                max_size = (800, 800)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                # Save to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr = img_byte_arr.getvalue()
                return base64.b64encode(img_byte_arr).decode('utf-8')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
    def train(self, csv_path, images_folder):
        """Train the model using the recipe dataset and load images"""
        print("Loading and processing recipe data...")
        # Load and process the dataset
        df = pd.read_csv(csv_path)
        
        # Process recipes
        processed_recipes = []
        total_images_processed = 0
        
        for idx, row in df.iterrows():
            try:
                # Extract recipe data with proper parentheses
                name = str(row.get('Title', row.get('title', row.get('Name', row.get('name', '')))))
                ingredients = str(row.get('Ingredients', row.get('ingredients', '')))
                instructions = str(row.get('Instructions', row.get('instructions', '')))
                category = str(row.get('Category', row.get('category', 'Main Dish'))).lower()
                image_name = str(row.get('Image_Name', ''))
                
                # Clean ingredients
                ingredients_list = [ing.strip() for ing in ingredients.split(',') if ing.strip()]
                processed_ingredients = [self.preprocess_ingredient(ing) for ing in ingredients_list]
                
                # Handle image
                image_data = None
                if image_name and images_folder:
                    image_path = os.path.join(images_folder, f"{image_name}.jpg")
                    if os.path.exists(image_path):
                        image_data = self.load_and_encode_image(image_path)
                        if image_data:
                            total_images_processed += 1
                
                recipe = {
                    'name': name,
                    'ingredients': ingredients_list,
                    'processed_ingredients': processed_ingredients,
                    'instructions': instructions.split('.'),
                    'category': category,
                    'image_name': image_name,
                    'image_data': image_data
                }
                processed_recipes.append(recipe)
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} recipes, {total_images_processed} images loaded...")
                
            except Exception as e:
                print(f"Error processing recipe {idx}: {e}")
                continue
        
        print(f"\nCreating TF-IDF vectors for {len(processed_recipes)} recipes...")
        # Create TF-IDF vectors for recipe ingredients
        ingredients_text = [' '.join(recipe['processed_ingredients']) for recipe in processed_recipes]
        self.recipe_vectors = self.vectorizer.fit_transform(ingredients_text)
        
        print("Building ingredient index...")
        # Build ingredient index
        for idx, recipe in enumerate(processed_recipes):
            for ingredient in recipe['processed_ingredients']:
                if ingredient not in self.ingredient_index:
                    self.ingredient_index[ingredient] = []
                self.ingredient_index[ingredient].append(idx)
        
        self.recipes_data = processed_recipes
        print(f"\nProcessing completed:")
        print(f"- Total recipes: {len(processed_recipes)}")
        print(f"- Total images: {total_images_processed}")
        print(f"- Unique ingredients: {len(self.ingredient_index)}")
        
    def save(self, model_path):
        """Save the trained model"""
        print(f"\nSaving model to {model_path}...")
        model_data = {
            'vectorizer': self.vectorizer,
            'recipes_data': self.recipes_data,
            'recipe_vectors': self.recipe_vectors,
            'ingredient_index': self.ingredient_index
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print("Model saved successfully!")

def main():
    # Initialize and train the model
    model = RecipeModel()
    csv_path = "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    images_folder = "Food Images"
    
    # Check if required files exist
    if not os.path.exists(csv_path):
        print(f"Error: Could not find the CSV file at {csv_path}")
        return
        
    if not os.path.exists(images_folder):
        print(f"Warning: Could not find the images folder at {images_folder}")
        images_folder = None
    
    print("Starting recipe model training...")
    model.train(csv_path, images_folder)
    
    # Save the trained model
    model_path = "recipe_model.pkl"
    model.save(model_path)

if __name__ == "__main__":
    main() 