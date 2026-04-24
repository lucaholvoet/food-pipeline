import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from pipeline import FoodPipeline
import json

pipeline = FoodPipeline()

# Test with a solid color image first (smoke test)
img = Image.new("RGB", (640, 640), color=(180, 120, 80))
result = pipeline.run(img)

print(json.dumps(result.model_dump(), indent=2))
print(f"\nStatus: {result.status}")
print(f"Items found: {len(result.items)}")
print(f"Requires VLM: {result.requires_vlm_refinement}")
print(f"Processing time: {result.processing_time_ms}ms")