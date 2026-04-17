import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nutrition.nutrition import NutritionLookup

lookup = NutritionLookup('nutrition/usda.index', 'nutrition/usda_records.json')

tests = ['fried rice', 'pizza', 'apple pie', 'grilled salmon', 'caesar salad']
for food in tests:
    result = lookup.lookup(food)
    print(f"{food}:")
    print(f"  matched: {result['usda_description']}")
    print(f"  nutrition: {result['nutrition_per_100g']}")
    print()