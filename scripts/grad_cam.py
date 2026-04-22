"""
Grad-CAM Visualization for Food Classifier

Shows which pixels the CNN focuses on when classifying food.
Covers: Lab 2 — CNN Interpretation (Grad-CAM, Saliency Maps)

Requirements:
  - Model weights: models/efficientnet_b0_food101_best.pt
  - Class labels: models/classifier/idx_to_class.json
  - Test images in data/eval/

Run:
    venv\\Scripts\\activate
    python -u scripts/grad_cam.py
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
sys.path.insert(0, PROJECT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(PROJECT, "models",
    "efficientnet_b0_food101_best.pt")
LABELS_PATH = os.path.join(PROJECT, "models",
    "classifier", "idx_to_class.json")
EVAL_DIR = os.path.join(PROJECT, "data", "eval")
OUTPUT_DIR = os.path.join(PROJECT, "output")

# ── Load Model ──

def load_model(model_path, labels_path, device="cpu"):
    """Load the trained EfficientNet-B0 classifier."""
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 512),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 101),
    )
    state = torch.load(model_path,
                        map_location=device)
    model.load_state_dict(state)
    model.eval()

    with open(labels_path) as f:
        idx_to_class = json.load(f)

    return model, idx_to_class


# ── Grad-CAM Implementation ──

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    How it works (from Lab 2 interpretation):
    1. Forward pass: image through CNN → prediction
    2. Get feature maps from last conv layer
    3. Backward pass: compute gradients of
       predicted class vs feature maps
    4. Pool gradients globally → get weight per
       feature map channel
    5. Weighted sum of feature maps → heatmap
    6. Resize heatmap to input size → overlay
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the target layer to capture
        activations and gradients."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer = self._find_layer(self.target_layer)
        self._hooks.append(
            layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            layer.register_full_backward_hook(backward_hook)
        )

    def _find_layer(self, name):
        """Find a layer by name in the model."""
        for n, m in self.model.named_modules():
            if n == name:
                return m
        # Fallback: last conv layer of EfficientNet
        return list(self.model.features.children())[-1]

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: preprocessed image tensor (1,3,224,224)
            target_class: which class to explain (default: top pred)
        
        Returns:
            heatmap: numpy array (224, 224) values 0-1
            pred_class: predicted class index
            pred_conf: confidence score
        """
        # Forward pass
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        pred_conf = probs[0, target_class].item()

        # Backward pass (compute gradients)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Generate heatmap
        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3),
                                       keepdim=True)
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1,
                                                keepdim=True)
        cam = F.relu(cam)  # ReLU: only positive influence

        # Normalize to 0-1
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / cam.max()

        # Resize to input size
        cam_img = Image.fromarray(
            (cam * 255).astype(np.uint8)
        )
        cam_resized = cam_img.resize(
            (224, 224), Image.BILINEAR
        )
        heatmap = np.array(cam_resized) / 255.0

        return heatmap, target_class, pred_conf

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

# ── Visualization ──

def visualize_gradcam(image_path, model, idx_to_class,
                      save_path=None):
    """Generate and save Grad-CAM visualization
    for one image."""

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    # Original image for overlay (resized)
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized) / 255.0

    # Target layer: last feature layer of EfficientNet
    grad_cam = GradCAM(model, "features.7")
    heatmap, pred_class, pred_conf = grad_cam.generate(
        input_tensor
    )
    grad_cam.remove_hooks()

    # Get class name
    class_name = idx_to_class.get(
        str(pred_class), "unknown"
    )

    # Create color heatmap
    cmap = plt.cm.jet
    colored_heatmap = cmap(heatmap)[:, :, :3]

    # Overlay heatmap on image
    alpha = 0.5
    overlay = (1 - alpha) * img_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    # ── Create figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    display = class_name.replace("_", " ").title()
    axes[2].set_title(
        f"Overlay\nPrediction: {display} "
        f"({pred_conf:.1%})",
        fontsize=12,
    )
    axes[2].axis("off")

    plt.suptitle(
        "Grad-CAM: What the CNN sees when classifying food",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200,
                    bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()

    return class_name, pred_conf

# ── Main ──

def main():
    print("=" * 55)
    print("Grad-CAM: CNN Interpretation")
    print("(Lab 2: Which pixels matter for prediction)")
    print("=" * 55)

    # Check if model weights exist
    if not os.path.exists(MODEL_PATH):
        print(f"\nModel weights not found:")
        print(f"  {MODEL_PATH}")
        print(f"\nTo run Grad-CAM:")
        print(f"  1. Download weights from Google Drive")
        print(f"  2. Place in models/")
        print(f"\nGenerating demo with random weights...")
        demo_mode = True
        # Create a random model for demo
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 101),
        )
        model.eval()
        idx_to_class = {str(i): f"food_{i}"
                        for i in range(101)}
    else:
        demo_mode = False
        print("\nLoading trained model...")
        model, idx_to_class = load_model(MODEL_PATH,
                                          LABELS_PATH)
        print("  Model loaded!")

    # Process eval images
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(EVAL_DIR):
        images = sorted(
            f for f in os.listdir(EVAL_DIR)
            if f.endswith((".jpg", ".png"))
        )
    else:
        images = []

    if not images:
        print("\nNo eval images found.")
        print(f"  Expected: {EVAL_DIR}/*.jpg")
        return

    print(f"\nProcessing {len(images)} images...")

    for img_name in images:
        img_path = os.path.join(EVAL_DIR, img_name)
        save_path = os.path.join(
            OUTPUT_DIR,
            f"gradcam_{img_name.replace('.jpg','.png')}"
        )
        print(f"\n  {img_name}:", end="", flush=True)
        try:
            cls, conf = visualize_gradcam(
                img_path, model, idx_to_class, save_path
            )
            display = cls.replace("_", " ").title()
            tag = " (DEMO - random weights)" if demo_mode else ""
            print(f" predicted as {display} "
                  f"({conf:.1%}){tag}")
        except Exception as e:
            print(f" ERROR: {e}")

    # ── Summary explanation ──
    print(f"\n{'=' * 55}")
    print("HOW TO INTERPRET GRAD-CAM")
    print(f"{'=' * 55}")
    print("""
RED/YELLOW areas = pixels the CNN focused on
BLUE areas = pixels the CNN ignored

If the CNN predicts 'pizza' correctly:
  → Red area should be on the pizza surface
  → This means it learned the right visual features

If the CNN predicts wrong (e.g., 'hot_dog' for burger):
  → Red area might be on the bun (confusing part)
  → This reveals WHY the model made the mistake

The VLM (Gemma 4) helps fix these mistakes by
looking at the whole image, not just CNN features.
""")


if __name__ == "__main__":
    main()
