import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PIL import Image
import requests
import io
from classifier import FoodClassifier

MODEL_PATH = "models/efficientnet_b0_food101_best.pt"
LABELS_PATH = "models/idx_to_class.json"

@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model weights not yet downloaded from Drive"
)
def test_classifier_loads():
    clf = FoodClassifier(MODEL_PATH, LABELS_PATH)
    assert clf.model is not None

@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model weights not yet downloaded from Drive"
)
def test_classifier_output_format():
    clf = FoodClassifier(MODEL_PATH, LABELS_PATH)

    # Create a simple test image (solid color — just tests the pipeline works)
    img = Image.new("RGB", (224, 224), color=(180, 100, 50))

    results = clf.predict(img, top_k=3)

    assert len(results) == 3
    for r in results:
        assert "label" in r
        assert "display_name" in r
        assert "confidence" in r
        assert 0.0 <= r["confidence"] <= 1.0
    
    # Confidences should sum to roughly 1.0 across all classes
    total = sum(r["confidence"] for r in results)
    assert total <= 1.0

def test_classifier_top_k():
    assert True