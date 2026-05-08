import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "reviews.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "embeddings")

os.makedirs(OUTPUT_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)

model = SentenceTransformer("all-MiniLM-L6-v2")

# clean text
df["text"] = df["text"].fillna("").astype(str)

print("Generating embeddings...")

embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

np.save(os.path.join(OUTPUT_PATH, "review_embeddings.npy"), embeddings)
df.to_csv(os.path.join(OUTPUT_PATH, "reviews_with_ids.csv"), index=False)

print("Done ✔")