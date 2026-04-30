import torch
import numpy as np
from PIL import Image


class DepthEstimator:
    def __init__(self):
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = midas_transforms.small_transform

    def get_depth_map(self, image: Image.Image) -> np.ndarray:
        img = np.array(image.convert("RGB"))
        input_tensor = self.transform(img)
        with torch.no_grad():
            depth = self.model(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_np = depth.numpy()
        # normalize to 0-1
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        return depth_np