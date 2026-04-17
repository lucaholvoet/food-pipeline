import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from portion.portion import PortionEstimator
from nutrition.nutrition import NutritionLookup

estimator = PortionEstimator()
lookup = NutritionLookup('nutrition/usda.index', 'nutrition/usda_records.json')

image_h, image_w = 640, 640
cy, cx = image_h // 2, image_w // 2
y, x = np.ogrid[:image_h, :image_w]

plate_mask = np.zeros((image_h, image_w), dtype=np.uint8)
plate_mask[(x - cx)**2 + (y - cy)**2 <= 200**2] = 1

food_mask = np.zeros((image_h, image_w), dtype=np.uint8)
food_mask[250:400, 220:420] = 1

foods = ["fried_rice", "pizza", "caesar_salad", "grilled_salmon"]

print("=" * 55)
for food in foods:
    portion = estimator.estimate(food, food_mask, plate_mask)
    nutrition_100g = lookup.get_nutrition(food.replace("_", " "))

    grams = portion["estimated_grams"]
    factor = grams / 100.0

    nutrition_total = {
        k: round(v * factor, 1)
        for k, v in nutrition_100g.items()
    }

    print(f"Food: {food}")
    print(f"  Estimated grams : {grams}g")
    print(f"  Nutrition/100g  : {nutrition_100g}")
    print(f"  Nutrition total : {nutrition_total}")
    print("-" * 55)