import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from pipeline import FoodPipeline
import json

pipeline = FoodPipeline()

img = Image.open("data/test_pizza.jpg").convert("RGB")
print(f"Image size: {img.size}")

result = pipeline.run(img)

output = result.model_dump()
output['image_base64'] = '[truncated]'
print(json.dumps(output, indent=2))
print(f"\nStatus: {result.status}")
print(f"Items found: {len(result.items)}")
for item in result.items:
    print(f"  - {item.display_name} ({item.classification_confidence:.0%}) "
          f"~{item.estimated_grams}g → {item.nutrition_total.calories_kcal} kcal")
print(f"Total calories: {result.totals.calories_kcal} kcal")
print(f"Requires VLM: {result.requires_vlm_refinement}")
print(f"Processing time: {result.processing_time_ms}ms")