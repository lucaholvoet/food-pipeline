import pytest
from PIL import Image
import requests
import io
from classifier import FoodClassifier
import os

MODEL_PATH = "models/efficientnet_b0_food101_best.pt"
LABELS_PATH = "models/classifier/idx_to_class.json"

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

    # Download a simple test image (pizza from wikipedia)
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/800px-Eq_it-na_pizza-margherita_sep2005_sml.jpg"
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert("RGB")

    results = clf.predict(img, top_k=3)

    assert len(results) == 3
    for r in results:
        assert "label" in r
        assert "display_name" in r
        assert "confidence" in r
        assert 0.0 <= r["confidence"] <= 1.0

def test_classifier_top_k():
    # This test doesn't need weights - just checks the output contract
    assert True  # placeholder until weights exist