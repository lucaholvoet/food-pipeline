import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

CLASS_NAMES = {0: "plate", 1: "food"}

class FoodDetector:
    def __init__(self, model_path: str, device: str = "cpu",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image: Image.Image):
        """
        Input:  PIL image
        Output: dict with food_items and plate_mask
        """
        img_w, img_h = image.size
        img_array = np.array(image)

        results = self.model(
            img_array,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]

        food_items = []
        plate_mask = None

        if results.masks is None:
            return {"food_items": food_items, "plate_mask": plate_mask,
                    "plate_detected": False, "image_size": (img_w, img_h)}

        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()

            # Resize mask to original image size
            mask_resized = cv2.resize(
                masks[i], (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)

            if cls_id == 0:  # plate
                plate_mask = mask_resized
            elif cls_id == 1:  # food
                # Crop image using bbox with padding
                pad = 10
                x1 = max(0, bbox[0] - pad)
                y1 = max(0, bbox[1] - pad)
                x2 = min(img_w, bbox[2] + pad)
                y2 = min(img_h, bbox[3] + pad)
                crop = image.crop((x1, y1, x2, y2))

                food_items.append({
                    "bbox": bbox,
                    "detection_confidence": round(conf, 4),
                    "mask": mask_resized,
                    "crop": crop
                })

        return {
            "food_items": food_items,
            "plate_mask": plate_mask,
            "plate_detected": plate_mask is not None,
            "image_size": (img_w, img_h)
        }