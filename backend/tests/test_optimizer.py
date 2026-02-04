import unittest
import json
import csv
import os
import logging
from backend.optimizer import plan_meals, optimize_meal_plan, score_meal_plan, evaluate_macros_and_micros


TEST_DIR = os.path.dirname(__file__)
SYNTHETIC_CSV_PATH = os.path.join(TEST_DIR, "USDA.csv")

"""
Unit tests for meal optimizer using USDA Nutrient Database
SOURCE: https://www.kaggle.com/datasets/demomaster/usda-national-nutrient-database?resource=download
ID,Description,Calories,Protein,TotalFat,Carbohydrate,Sodium,SaturatedFat,Cholesterol,Sugar,Calcium,Iron,Potassium,VitaminC,VitaminE,VitaminD

"""
class TestMealOptimizerWithSyntheticData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load and transform CSV into nested format matching FDA guidelines."""
        cls.recipes = []
        required_fields = ['Calories', 'Protein', 'TotalFat', 'Carbohydrate']
        
        with open(SYNTHETIC_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if all required fields are present
                if not all(field in row for field in required_fields):
                    continue
                
                # Try to convert required fields to float
                try:
                    calories = float(row['Calories'])
                    protein = float(row['Protein'])
                    total_fat = float(row['TotalFat'])
                    total_carb = float(row['Carbohydrate'])
                except (ValueError, TypeError):
                    # Skip rows with invalid numeric values
                    continue
                
                # Only process rows with valid data
                if not (calories and protein and total_fat and total_carb):
                    continue
                
                # Safe conversion for optional fields with error handling
                def safe_float(value, default=0):
                    try:
                        return float(value) if value else default
                    except (ValueError, TypeError):
                        return default
                
                recipe = {
                    'name': row.get('Description', 'Unknown'),
                    'macronutrients': {
                        'calories': calories,
                        'protein': protein,
                        'total_fat': total_fat,
                        'saturated_fat': safe_float(row.get('SaturatedFat')),
                        'cholesterol': safe_float(row.get('Cholesterol')),
                        'total_carbohydrate': total_carb,
                    },
                    'vitamins': {
                        'vitamin_c': safe_float(row.get('VitaminC')),
                        'vitamin_e': safe_float(row.get('VitaminE')),
                        'vitamin_d': safe_float(row.get('VitaminD')),
                    },
                    'minerals': {
                        'calcium': safe_float(row.get('Calcium')),
                        'iron': safe_float(row.get('Iron')),
                        'potassium': safe_float(row.get('Potassium')),
                        'sodium': safe_float(row.get('Sodium')),
                    }
                }
                cls.recipes.append(recipe)

        # Minimal user preferences for testing
        cls.user_prefs = {
            "meal_count_per_day": 3,
            "budget_per_week": 100,
            "weights": {
                "cost": 0.25,
                "nutrition": 0.3,
                "time": 0.2,
                "diversity": 0.1,
                "leftovers": 0.15,
            },
            "macro_goal_ratio": {"protein": 0.3, "fat": 0.3, "carbs": 0.4},
            "diversity_rule": True,
            "prioritize_existing_groceries": True,
            "cuisine_preferences": ["vegetarian", "greek", "italian"],
        }

    def test_optimizer_returns_correct_meal_count(self):
        """Optimizer should return exactly the number of meals requested."""
        chosen = optimize_meal_plan(self.user_prefs, self.recipes, use_similarity=False)
        self.assertEqual(len(chosen), self.user_prefs["meal_count_per_day"])
        self.assertTrue(all("name" in m for m in chosen))

    def test_fda_score_range(self):
        """Check that FDA score is between 0 and 1."""
        fda_score = evaluate_macros_and_micros(self.recipes)
        self.assertGreaterEqual(fda_score, 0)
        self.assertLessEqual(fda_score, 1)

    def test_total_score_calculation(self):
        """Check total score calculation does not error and returns positive number."""
        score = score_meal_plan(self.user_prefs, self.recipes)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)

    def test_plan_meals_wrapper(self):
        """Test the plan_meals wrapper that loads CSV."""
        chosen = plan_meals(self.user_prefs, recipe_csv_path=SYNTHETIC_CSV_PATH, use_similarity=True)
        
        logging.basicConfig(level=logging.INFO)

        logging.info(f"Chosen meals: {[m['name'] for m in chosen['chosen_meals']]}")
        logging.info(f"Total score: {chosen['total_score']}")
        logging.info(f"FDA score: {chosen['fda_score']}")

        self.assertEqual(len(chosen), self.user_prefs["meal_count_per_day"])
        self.assertTrue(all("name" in m for m in chosen["chosen_meals"]))


if __name__ == "__main__":
    unittest.main()
