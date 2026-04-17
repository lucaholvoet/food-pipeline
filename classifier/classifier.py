import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import os

class FoodClassifier:
    def __init__(self, model_path: str, labels_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.labels = self._load_labels(labels_path)
        self.transform = self._build_transform()

    def _load_model(self, model_path: str):
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 101)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def _load_labels(self, labels_path: str):
        with open(labels_path) as f:
            return json.load(f)  # idx_to_class dict

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image: Image.Image, top_k: int = 3):
        """
        Input:  PIL image (crop from YOLO detector)
        Output: list of {label, display_name, confidence} top_k predictions
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            top = torch.topk(probs, top_k)

        results = []
        for i, p in zip(top.indices[0], top.values[0]):
            label = self.labels[str(i.item())]
            results.append({
                "label": label,
                "display_name": label.replace("_", " ").title(),
                "confidence": round(p.item(), 4)
            })

        return results