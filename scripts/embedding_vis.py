"""
Food-101 Embedding Visualization — t-SNE & UMAP

Visualizes how food names cluster in embedding space.
Uses sentence-transformers (same model as nutrition lookup).

Covers: Lab 3 — Embeddings, t-SNE, UMAP

Run:
    venv\\Scripts\\activate
    python -u scripts/embedding_vis.py
"""

import sys
import os
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
sys.path.insert(0, PROJECT)

from vlm.prompts import FOOD101_CLASSES

print("=" * 55)
print("Food-101 Embedding Visualization")
print("(Lab 3: Embeddings, t-SNE, UMAP)")
print("=" * 55)

# ── Step 1: Generate embeddings ──
print("\n[1/4] Loading sentence-transformer model...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"[1/4] Encoding {len(FOOD101_CLASSES)} food names...")
food_display = [f.replace("_", " ") for f in FOOD101_CLASSES]
embeddings = model.encode(food_display, show_progress_bar=True)
print(f"  Embedding shape: {embeddings.shape}")
print(f"  (101 foods × 384 dimensions)")

# ── Step 2: Compute cosine similarity matrix ──
print("\n[2/4] Computing cosine similarity matrix...")
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(embeddings)

# Show most similar pairs
print("\n  Top 10 most similar food pairs:")
pairs = []
for i in range(len(FOOD101_CLASSES)):
    for j in range(i + 1, len(FOOD101_CLASSES)):
        pairs.append((
            FOOD101_CLASSES[i], FOOD101_CLASSES[j],
            sim_matrix[i][j]
        ))
pairs.sort(key=lambda x: x[2], reverse=True)
for a, b, sim in pairs[:10]:
    print(f"    {a:<25} <-> {b:<25} sim={sim:.3f}")

# Show least similar pairs
print("\n  Top 5 least similar food pairs:")
for a, b, sim in pairs[-5:]:
    print(f"    {a:<25} <-> {b:<25} sim={sim:.3f}")

# ── Step 3: t-SNE visualization ──
print("\n[3/4] Running t-SNE dimensionality reduction...")
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

tsne = TSNE(
    n_components=2,
    perplexity=15,
    random_state=42,
    max_iter=1000,
)
tsne_coords = tsne.fit_transform(embeddings)

# ── Step 4: Plot t-SNE ──
print("[4/4] Generating plots...")

# Define food categories for coloring
CATEGORIES = {
    "Desserts": ["apple_pie", "baklava", "bread_pudding", "cannoli",
        "carrot_cake", "cheesecake", "chocolate_cake",
        "chocolate_mousse", "creme_brulee", "cup_cakes",
        "donuts", "ice_cream", "macarons", "panna_cotta",
        "red_velvet_cake", "strawberry_shortcake", "tiramisu",
        "frozen_yogurt", "beignets"],
    "Asian": ["bibimbap", "dumplings", "edamame", "gyoza",
        "ramen", "sashimi", "sushi", "takoyaki", "pho",
        "pad_thai", "peking_duck", "chicken_curry",
        "hot_and_sour_soup", "miso_soup", "spring_rolls",
        "fried_rice", "samosa"],
    "Western Main": ["hamburger", "hot_dog", "pizza", "steak",
        "filet_mignon", "prime_rib", "pork_chop",
        "grilled_salmon", "fish_and_chips", "poutine",
        "spaghetti_bolognese", "spaghetti_carbonara",
        "grilled_cheese_sandwich", "club_sandwich",
        "breakfast_burrito", "chicken_quesadilla",
        "pulled_pork_sandwich", "lobster_roll_sandwich"],
    "Salads/Veggie": ["caesar_salad", "greek_salad", "beet_salad",
        "caprese_salad", "ceviche", "seaweed_salad",
        "guacamole", "hummus", "falafel", "bruschetta"],
    "Seafood": ["fried_calamari", "clam_chowder", "crab_cakes",
        "lobster_bisque", "mussels", "oysters", "scallops",
        "shrimp_and_grits", "escargots", "tuna_tartare",
        "foie_gras", "beef_carpaccio", "beef_tartare"],
    "Breakfast": ["pancakes", "waffles", "french_toast",
        "eggs_benedict", "omelette", "huevos_rancheros",
        "croque_madame", "deviled_eggs"],
}

CAT_COLORS = {
    "Desserts": "#e74c3c",
    "Asian": "#f39c12",
    "Western Main": "#3498db",
    "Salads/Veggie": "#2ecc71",
    "Seafood": "#9b59b6",
    "Breakfast": "#e67e22",
}

# Assign category to each food
def get_category(food):
    for cat, foods in CATEGORIES.items():
        if food in foods:
            return cat
    return "Other"

categories = [get_category(f) for f in FOOD101_CLASSES]

# ── Plot 1: t-SNE colored by category ──
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

for cat in CAT_COLORS:
    mask = [c == cat for c in categories]
    pts = tsne_coords[mask]
    names = [FOOD101_CLASSES[i]
             for i, m in enumerate(mask) if m]
    ax.scatter(pts[:, 0], pts[:, 1],
               c=CAT_COLORS[cat], label=cat,
               s=80, alpha=0.8, edgecolors="white",
               linewidth=0.5)
    for i, name in enumerate(names):
        ax.annotate(
            name.replace("_", "\n"),
            (pts[i, 0], pts[i, 1]),
            fontsize=6, ha="center", va="bottom",
            alpha=0.7,
        )

# Other foods (not in defined categories)
mask_other = [c == "Other" for c in categories]
pts_other = tsne_coords[mask_other]
ax.scatter(pts_other[:, 0], pts_other[:, 1],
           c="#bdc3c7", label="Other",
           s=40, alpha=0.5, edgecolors="white")

ax.set_title(
    "Food-101 Embedding Space (t-SNE)\n"
    "Sentence-transformer: all-MiniLM-L6-v2",
    fontsize=14, fontweight="bold",
)
ax.legend(loc="upper right", fontsize=10)
ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
plt.tight_layout()

out_path = os.path.join(PROJECT, "output", "tsne_food101.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"  Saved: {out_path}")
plt.close()

# ── Plot 2: Similarity heatmap (top 30 foods) ──
# Pick 30 representative foods
REPRESENTATIVE = [
    "pizza", "hamburger", "hot_dog", "steak", "sushi",
    "ramen", "pho", "pad_thai", "fried_rice", "bibimbap",
    "spaghetti_bolognese", "spaghetti_carbonara", "pasta",
    "caesar_salad", "greek_salad", "pancakes", "waffles",
    "french_toast", "ice_cream", "frozen_yogurt",
    "chocolate_cake", "cheesecake", "tiramisu", "donuts",
    "grilled_salmon", "lobster_bisque", "clam_chowder",
    "chicken_curry", "falafel", "huevos_rancheros",
]

# Find indices (some might not exist)
rep_idx = []
rep_names = []
for f in REPRESENTATIVE:
    if f in FOOD101_CLASSES:
        rep_idx.append(FOOD101_CLASSES.index(f))
        rep_names.append(f.replace("_", " "))

rep_sim = sim_matrix[np.ix_(rep_idx, rep_idx)]

fig, ax = plt.subplots(1, 1, figsize=(14, 12))
im = ax.imshow(rep_sim, cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(rep_names)))
ax.set_yticks(range(len(rep_names)))
ax.set_xticklabels(rep_names, rotation=75,
                    ha="right", fontsize=8)
ax.set_yticklabels(rep_names, fontsize=8)
ax.set_title(
    "Food Name Similarity Matrix (cosine)\n"
    "Green=high similarity, Red=low similarity",
    fontsize=13, fontweight="bold",
)
plt.colorbar(im, ax=ax, shrink=0.7,
             label="Cosine Similarity")
plt.tight_layout()

out_path2 = os.path.join(PROJECT, "output",
                          "similarity_heatmap.png")
plt.savefig(out_path2, dpi=200, bbox_inches="tight")
print(f"  Saved: {out_path2}")
plt.close()

# ── Plot 3: Confusion-prone food pairs ──
print("\n" + "=" * 55)
print("ANALYSIS: Confusion-Prone Food Pairs")
print("(Why CV pipeline gets confused)")
print("=" * 55)

# Foods that are very similar but different classes
CONFUSION_PAIRS = [
    ("ramen", "pho"), ("ramen", "hot_and_sour_soup"),
    ("spaghetti_bolognese", "spaghetti_carbonara"),
    ("spaghetti_bolognese", "risotto"),
    ("fried_rice", "bibimbap"), ("fried_rice", "paella"),
    ("caesar_salad", "greek_salad"),
    ("caesar_salad", "beet_salad"),
    ("pancakes", "french_toast"), ("pancakes", "waffles"),
    ("ice_cream", "frozen_yogurt"),
    ("chocolate_cake", "red_velvet_cake"),
    ("hamburger", "hot_dog"),
    ("steak", "filet_mignon"), ("steak", "prime_rib"),
    ("sushi", "sashimi"),
]

print(f"\n{'Food A':<25} {'Food B':<25} {'Similarity'}")
print("-" * 60)
for a, b in CONFUSION_PAIRS:
    if a in FOOD101_CLASSES and b in FOOD101_CLASSES:
        ia = FOOD101_CLASSES.index(a)
        ib = FOOD101_CLASSES.index(b)
        sim = sim_matrix[ia][ib]
        bar = "█" * int(sim * 20)
        print(f"{a:<25} {b:<25} {sim:.3f} {bar}")

print("\n" + "=" * 55)
print("These high-similarity pairs explain why the")
print("CV classifier struggles with certain foods.")
print("The VLM helps resolve these ambiguities!")
print("=" * 55)

# ── Plot 4: Confusion pair similarity bars ──
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
pair_labels = []
pair_sims = []
for a, b in CONFUSION_PAIRS:
    if a in FOOD101_CLASSES and b in FOOD101_CLASSES:
        ia = FOOD101_CLASSES.index(a)
        ib = FOOD101_CLASSES.index(b)
        pair_labels.append(
            f"{a.replace('_',' ')}\nvs\n"
            f"{b.replace('_',' ')}"
        )
        pair_sims.append(sim_matrix[ia][ib])

colors = ["#e74c3c" if s > 0.7
          else "#f39c12" if s > 0.5
          else "#2ecc71"
          for s in pair_sims]
ax.barh(range(len(pair_labels)), pair_sims,
        color=colors, edgecolor="white")
ax.set_yticks(range(len(pair_labels)))
ax.set_yticklabels(pair_labels, fontsize=8)
ax.set_xlabel("Cosine Similarity", fontsize=11)
ax.set_title(
    "Embedding Similarity of Confusion-Prone Pairs\n"
    "Red = very similar (CV likely confuses these)",
    fontsize=13, fontweight="bold",
)
ax.axvline(x=0.7, color="red", linestyle="--",
           alpha=0.5, label="High similarity")
ax.legend()
plt.tight_layout()

out_path3 = os.path.join(
    PROJECT, "output", "confusion_pairs.png"
)
plt.savefig(out_path3, dpi=200, bbox_inches="tight")
print(f"\n  Saved: {out_path3}")
plt.close()

print(f"\nDone! All plots saved to output/")
