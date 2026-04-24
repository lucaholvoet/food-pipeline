import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from PIL import Image
from detector import FoodDetector

MODEL_PATH = "models/yolov8n_food_best.pt"

@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="YOLO weights not found"
)
def test_detector_loads():
    det = FoodDetector(MODEL_PATH)
    assert det.model is not None

@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="YOLO weights not found"
)
def test_detector_output_format():
    det = FoodDetector(MODEL_PATH)
    img = Image.new("RGB", (640, 640), color=(200, 150, 100))
    result = det.detect(img)

    assert "food_items" in result
    assert "plate_mask" in result
    assert "plate_detected" in result
    assert "image_size" in result
    assert result["image_size"] == (640, 640)
    assert isinstance(result["food_items"], list)

@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="YOLO weights not found"
)
def test_detector_crop_format():
    det = FoodDetector(MODEL_PATH)
    img = Image.new("RGB", (640, 640), color=(200, 150, 100))
    result = det.detect(img)

    for item in result["food_items"]:
        assert "bbox" in item
        assert "detection_confidence" in item
        assert "mask" in item
        assert "crop" in item
        assert len(item["bbox"]) == 4
        assert isinstance(item["crop"], Image.Image)
        assert isinstance(item["mask"], np.ndarray)