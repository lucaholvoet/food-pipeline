import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class NutritionLookup:
    def __init__(self, index_path: str, records_path: str, device: str = "cpu"):
        self.index = faiss.read_index(index_path)
        with open(records_path) as f:
            self.records = json.load(f)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def lookup(self, food_name: str, top_k: int = 1):
        """
        Input:  food name string (e.g. 'fried_rice')
        Output: nutrition dict per 100g
        """
        query = food_name.replace("_", " ")
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(embedding)

        distances, indices = self.index.search(embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            record = self.records[idx]
            results.append({
                "usda_description": record["description"],
                "similarity": round(float(dist), 4),
                "nutrition_per_100g": record["nutrition"]
            })

        return results[0] if top_k == 1 else results

    def get_nutrition(self, food_name: str):
        """
        Simplified interface for the pipeline.
        Input:  food name string
        Output: nutrition_per_100g dict directly
        """
        result = self.lookup(food_name)
        return result.get("nutrition_per_100g", {})