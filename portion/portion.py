import numpy as np
import json
import os
from PIL import Image

PLATE_DIAMETER_CM = 26.0

DENSITY_TABLE = {
    "apple_pie": {"height_cm": 4.0, "density_g_cm3": 0.9},
    "baby_back_ribs": {"height_cm": 3.0, "density_g_cm3": 0.85},
    "baklava": {"height_cm": 3.0, "density_g_cm3": 0.9},
    "beef_carpaccio": {"height_cm": 0.5, "density_g_cm3": 1.0},
    "beef_tartare": {"height_cm": 4.0, "density_g_cm3": 1.0},
    "beet_salad": {"height_cm": 4.0, "density_g_cm3": 0.5},
    "beignets": {"height_cm": 4.0, "density_g_cm3": 0.4},
    "bibimbap": {"height_cm": 5.0, "density_g_cm3": 0.7},
    "bread_pudding": {"height_cm": 4.0, "density_g_cm3": 0.85},
    "breakfast_burrito": {"height_cm": 5.0, "density_g_cm3": 0.75},
    "bruschetta": {"height_cm": 2.5, "density_g_cm3": 0.6},
    "caesar_salad": {"height_cm": 5.0, "density_g_cm3": 0.3},
    "cannoli": {"height_cm": 4.0, "density_g_cm3": 0.7},
    "caprese_salad": {"height_cm": 3.0, "density_g_cm3": 0.7},
    "carrot_cake": {"height_cm": 6.0, "density_g_cm3": 0.8},
    "ceviche": {"height_cm": 3.0, "density_g_cm3": 0.9},
    "cheesecake": {"height_cm": 5.0, "density_g_cm3": 0.9},
    "cheese_plate": {"height_cm": 2.0, "density_g_cm3": 1.1},
    "chicken_curry": {"height_cm": 4.0, "density_g_cm3": 0.85},
    "chicken_quesadilla": {"height_cm": 2.0, "density_g_cm3": 0.8},
    "chicken_wings": {"height_cm": 4.0, "density_g_cm3": 0.85},
    "chocolate_cake": {"height_cm": 6.0, "density_g_cm3": 0.85},
    "chocolate_mousse": {"height_cm": 5.0, "density_g_cm3": 0.5},
    "churros": {"height_cm": 3.0, "density_g_cm3": 0.5},
    "clam_chowder": {"height_cm": 4.0, "density_g_cm3": 0.95},
    "club_sandwich": {"height_cm": 8.0, "density_g_cm3": 0.6},
    "crab_cakes": {"height_cm": 3.0, "density_g_cm3": 0.85},
    "creme_brulee": {"height_cm": 3.0, "density_g_cm3": 0.95},
    "croque_madame": {"height_cm": 4.0, "density_g_cm3": 0.75},
    "cup_cakes": {"height_cm": 6.0, "density_g_cm3": 0.5},
    "deviled_eggs": {"height_cm": 3.0, "density_g_cm3": 0.95},
    "donuts": {"height_cm": 4.0, "density_g_cm3": 0.4},
    "dumplings": {"height_cm": 3.0, "density_g_cm3": 0.8},
    "edamame": {"height_cm": 3.0, "density_g_cm3": 0.7},
    "eggs_benedict": {"height_cm": 6.0, "density_g_cm3": 0.75},
    "escargots": {"height_cm": 3.0, "density_g_cm3": 0.9},
    "falafel": {"height_cm": 4.0, "density_g_cm3": 0.7},
    "filet_mignon": {"height_cm": 4.0, "density_g_cm3": 1.0},
    "fish_and_chips": {"height_cm": 4.0, "density_g_cm3": 0.7},
    "foie_gras": {"height_cm": 2.0, "density_g_cm3": 1.0},
    "french_fries": {"height_cm": 5.0, "density_g_cm3": 0.4},
    "french_onion_soup": {"height_cm": 5.0, "density_g_cm3": 0.95},
    "french_toast": {"height_cm": 3.0, "density_g_cm3": 0.7},
    "fried_calamari": {"height_cm": 3.0, "density_g_cm3": 0.6},
    "fried_rice": {"height_cm": 3.0, "density_g_cm3": 0.75},
    "frozen_yogurt": {"height_cm": 5.0, "density_g_cm3": 0.6},
    "garlic_bread": {"height_cm": 3.0, "density_g_cm3": 0.5},
    "gnocchi": {"height_cm": 3.0, "density_g_cm3": 0.85},
    "greek_salad": {"height_cm": 5.0, "density_g_cm3": 0.4},
    "grilled_cheese_sandwich": {"height_cm": 5.0, "density_g_cm3": 0.65},
    "grilled_salmon": {"height_cm": 3.0, "density_g_cm3": 0.95},
    "guacamole": {"height_cm": 3.0, "density_g_cm3": 0.85},
    "gyoza": {"height_cm": 2.5, "density_g_cm3": 0.8},
    "hamburger": {"height_cm": 8.0, "density_g_cm3": 0.7},
    "hot_and_sour_soup": {"height_cm": 4.0, "density_g_cm3": 0.95},
    "hot_dog": {"height_cm": 5.0, "density_g_cm3": 0.75},
    "huevos_rancheros": {"height_cm": 3.0, "density_g_cm3": 0.8},
    "hummus": {"height_cm": 2.0, "density_g_cm3": 0.95},
    "ice_cream": {"height_cm": 5.0, "density_g_cm3": 0.55},
    "lobster_bisque": {"height_cm": 4.0, "density_g_cm3": 0.95},
    "lobster_roll_sandwich": {"height_cm": 6.0, "density_g_cm3": 0.65},
    "macaroni_and_cheese": {"height_cm": 3.0, "density_g_cm3": 0.85},
    "macarons": {"height_cm": 3.0, "density_g_cm3": 0.6},
    "miso_soup": {"height_cm": 4.0, "density_g_cm3": 0.98},
    "mussels": {"height_cm": 4.0, "density_g_cm3": 0.7},
    "nachos": {"height_cm": 5.0, "density_g_cm3": 0.4},
    "omelette": {"height_cm": 2.0, "density_g_cm3": 0.85},
    "onion_rings": {"height_cm": 4.0, "density_g_cm3": 0.45},
    "oysters": {"height_cm": 2.0, "density_g_cm3": 0.9},
    "pad_thai": {"height_cm": 3.0, "density_g_cm3": 0.75},
    "paella": {"height_cm": 3.0, "density_g_cm3": 0.8},
    "pancakes": {"height_cm": 3.0, "density_g_cm3": 0.65},
    "panna_cotta": {"height_cm": 5.0, "density_g_cm3": 0.9},
    "peking_duck": {"height_cm": 3.0, "density_g_cm3": 0.9},
    "pho": {"height_cm": 5.0, "density_g_cm3": 0.95},
    "pizza": {"height_cm": 2.5, "density_g_cm3": 0.7},
    "pork_chop": {"height_cm": 2.5, "density_g_cm3": 1.0},
    "poutine": {"height_cm": 5.0, "density_g_cm3": 0.7},
    "prime_rib": {"height_cm": 4.0, "density_g_cm3": 1.0},
    "pulled_pork_sandwich": {"height_cm": 7.0, "density_g_cm3": 0.7},
    "ramen": {"height_cm": 5.0, "density_g_cm3": 0.95},
    "ravioli": {"height_cm": 2.0, "density_g_cm3": 0.85},
    "red_velvet_cake": {"height_cm": 7.0, "density_g_cm3": 0.8},
    "risotto": {"height_cm": 3.0, "density_g_cm3": 0.85},
    "samosa": {"height_cm": 4.0, "density_g_cm3": 0.7},
    "sashimi": {"height_cm": 1.5, "density_g_cm3": 1.0},
    "scallops": {"height_cm": 2.0, "density_g_cm3": 0.95},
    "seaweed_salad": {"height_cm": 3.0, "density_g_cm3": 0.6},
    "shrimp_and_grits": {"height_cm": 4.0, "density_g_cm3": 0.8},
    "spaghetti_bolognese": {"height_cm": 4.0, "density_g_cm3": 0.75},
    "spaghetti_carbonara": {"height_cm": 4.0, "density_g_cm3": 0.75},
    "spring_rolls": {"height_cm": 3.0, "density_g_cm3": 0.6},
    "steak": {"height_cm": 3.0, "density_g_cm3": 1.0},
    "strawberry_shortcake": {"height_cm": 6.0, "density_g_cm3": 0.5},
    "sushi": {"height_cm": 2.5, "density_g_cm3": 0.85},
    "tacos": {"height_cm": 6.0, "density_g_cm3": 0.6},
    "takoyaki": {"height_cm": 3.0, "density_g_cm3": 0.8},
    "tiramisu": {"height_cm": 5.0, "density_g_cm3": 0.7},
    "tuna_tartare": {"height_cm": 3.0, "density_g_cm3": 1.0},
    "waffles": {"height_cm": 3.0, "density_g_cm3": 0.5},
}

DEFAULT_DENSITY = {"height_cm": 3.0, "density_g_cm3": 0.75}


class PortionEstimator:
    def __init__(self, density_table: dict = None,
                 plate_diameter_cm: float = PLATE_DIAMETER_CM):
        self.density_table = density_table or DENSITY_TABLE
        self.plate_diameter_cm = plate_diameter_cm

    def estimate(self, food_label: str, mask: np.ndarray,
                 plate_mask: np.ndarray = None, depth_map: np.ndarray = None):

        food_px = float(mask.sum())

        if plate_mask is not None and plate_mask.sum() > 0:
            plate_px = float(plate_mask.sum())
            plate_radius_px = np.sqrt(plate_px / np.pi)
            plate_radius_cm = self.plate_diameter_cm / 2.0
            cm_per_px = plate_radius_cm / plate_radius_px
            portion_method = "plate_reference"
        else:
            cm_per_px = self._fallback_scale(mask)
            portion_method = "fallback_scale"

        food_area_cm2 = food_px * (cm_per_px ** 2)

        props = self.density_table.get(food_label, DEFAULT_DENSITY)
        density = props["density_g_cm3"]

        # Use MiDaS depth if available, otherwise fall back to density table height
        if depth_map is not None and depth_map.shape == mask.shape:
            food_depths = depth_map[mask > 0]
            if len(food_depths) > 0:
                # relative height: difference between food surface and background
                height_relative = float(food_depths.mean() - food_depths.min())
                # scale relative depth to cm (calibrated: 1.0 relative ~ 8cm max height)
                height_cm = max(0.5, min(height_relative * 8.0, 20.0))
                portion_method = portion_method + "_midas"
            else:
                height_cm = props["height_cm"]
        else:
            height_cm = props["height_cm"]

        volume_cm3 = food_area_cm2 * height_cm
        grams = volume_cm3 * density
        grams = max(10.0, min(grams, 1500.0))

        return {
            "estimated_grams": round(grams, 1),
            "portion_method": portion_method,
            "debug": {
                "food_area_cm2": round(food_area_cm2, 2),
                "height_cm": round(height_cm, 2),
                "density_g_cm3": density,
                "cm_per_px": round(cm_per_px, 5)
            }
        }

    def _fallback_scale(self, mask: np.ndarray):
        """
        When no plate is detected, assume food occupies
        roughly 40% of a standard 26cm plate footprint.
        Returns an estimated cm_per_px.
        """
        h, w = mask.shape
        image_area_px = h * w
        assumed_plate_area_cm2 = np.pi * (self.plate_diameter_cm / 2) ** 2
        cm2_per_px2 = assumed_plate_area_cm2 / image_area_px
        return np.sqrt(cm2_per_px2)