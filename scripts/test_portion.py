import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from portion.portion import PortionEstimator

estimator = PortionEstimator()

image_h, image_w = 640, 640

plate_mask = np.zeros((image_h, image_w), dtype=np.uint8)
plate_radius_px = 200
cy, cx = image_h // 2, image_w // 2
y, x = np.ogrid[:image_h, :image_w]
plate_mask[(x - cx)**2 + (y - cy)**2 <= plate_radius_px**2] = 1

food_mask = np.zeros((image_h, image_w), dtype=np.uint8)
food_mask[250:400, 220:420] = 1

tests = ["fried_rice", "pizza", "caesar_salad", "hamburger", "unknown_food"]
for food in tests:
    result = estimator.estimate(food, food_mask, plate_mask)
    print(f"{food}:")
    print(f"  grams: {result['estimated_grams']}g")
    print(f"  method: {result['portion_method']}")
    print(f"  debug: {result['debug']}")
    print()