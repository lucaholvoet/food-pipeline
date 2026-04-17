import pandas as pd
import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

USDA_DIR = "data/usda"
OUTPUT_DIR = "nutrition"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUTRIENT_IDS = {
    1008: "calories_kcal",
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbs_g",
    1079: "fiber_g",
}

print("Loading USDA data...")

foundation_food = pd.read_csv(f"{USDA_DIR}/foundation/food.csv")
foundation_nutrient = pd.read_csv(f"{USDA_DIR}/foundation/food_nutrient.csv")

sr_food = pd.read_csv(f"{USDA_DIR}/sr_legacy/food.csv")
sr_nutrient = pd.read_csv(f"{USDA_DIR}/sr_legacy/food_nutrient.csv")

foods = pd.concat([foundation_food, sr_food], ignore_index=True)
nutrients = pd.concat([foundation_nutrient, sr_nutrient], ignore_index=True)

print(f"Total food entries: {len(foods)}")

print("Building nutrition lookup per food...")

nutrition_map = {}

relevant = nutrients[nutrients["nutrient_id"].isin(NUTRIENT_IDS.keys())]

for fdc_id, group in tqdm(relevant.groupby("fdc_id")):
    entry = {}
    for _, row in group.iterrows():
        field = NUTRIENT_IDS.get(row["nutrient_id"])
        if field:
            entry[field] = round(float(row["amount"]), 2)
    if len(entry) >= 3:
        nutrition_map[int(fdc_id)] = entry

print(f"Foods with nutrition data: {len(nutrition_map)}")

print("Filtering foods to those with nutrition data...")

foods_filtered = foods[foods["fdc_id"].isin(nutrition_map.keys())].copy()
foods_filtered = foods_filtered.dropna(subset=["description"])
foods_filtered = foods_filtered.reset_index(drop=True)

print(f"Foods after filtering: {len(foods_filtered)}")

print("Embedding food names with sentence-transformers...")

model = SentenceTransformer("all-MiniLM-L6-v2")

descriptions = foods_filtered["description"].tolist()
embeddings = model.encode(
    descriptions,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype(np.float32)
faiss.normalize_L2(embeddings)

print("Building FAISS index...")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"Index built with {index.ntotal} vectors of dimension {dimension}")

faiss.write_index(index, f"{OUTPUT_DIR}/usda.index")
print("Saved usda.index")

records = []
for i, row in foods_filtered.iterrows():
    fdc_id = int(row["fdc_id"])
    records.append({
        "idx": i,
        "fdc_id": fdc_id,
        "description": row["description"],
        "nutrition": nutrition_map.get(fdc_id, {})
    })

with open(f"{OUTPUT_DIR}/usda_records.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"Saved usda_records.json with {len(records)} entries")
print("\nDone. Index is ready.")