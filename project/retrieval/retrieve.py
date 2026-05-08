import os
import numpy as np
import pandas as pd
import faiss

import sys
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from sentence_transformers import SentenceTransformer
from models.behavior_model import build_behavior_profile

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "review_embeddings.npy")
META_PATH = os.path.join(BASE_DIR, "data", "embeddings", "reviews_with_ids.csv")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "faiss.index")

# -------------------------------------------------
# Load model + data
# -------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv(META_PATH)
embeddings = np.load(EMB_PATH)

index = faiss.read_index(INDEX_PATH)

# -------------------------------------------------
# CORE: behavior-aware scoring
# -------------------------------------------------
def compute_behavior_score(item, profile):
    score = item["score"]

    traits = profile["behavioral_traits"]
    sentiment = profile["sentiment_profile"]
    rating = profile["rating_behavior"]

    # positive reviewer boost
    if traits.get("positive_reviewer"):
        score *= 1.05

    # critical reviewer reduces positivity bias
    if traits.get("critical_reviewer"):
        score *= 0.95

    # detail oriented users prefer longer reviews
    if traits.get("detail_oriented"):
        score *= 1.03

    # strict users prefer higher quality signals
    score *= (1 + rating["strictness"] * 0.02)

    # sentiment alignment
    score *= (1 + sentiment["avg_sentiment"] * 0.01)

    return score


# -------------------------------------------------
# MAIN RETRIEVE FUNCTION
# -------------------------------------------------
def retrieve(query, user_id=None, k=5):

    query_emb = model.encode([query])
    query_emb = np.array(query_emb).astype("float32")

    distances, indices = index.search(query_emb, k * 5)

    results = []

    for idx, dist in zip(indices[0], distances[0]):

        row = df.iloc[idx].to_dict()

        results.append({
            "user_id": row.get("user_id"),
            "business_id": row.get("business_id"),
            "text": row.get("text"),
            "stars": row.get("stars"),
            "score": float(1 / (1 + dist))
        })

    # -------------------------------------------------
    # STEP 3 CORE: behavior reranking
    # -------------------------------------------------
    if user_id:

        profile = build_behavior_profile(user_id)

        for r in results:
            r["score"] = compute_behavior_score(r, profile)

    # final sort
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:k]


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":

    test_query = "good food but slow service"
    test_user = df["user_id"].sample(1).values[0]

    print("\nTesting user:", test_user)

    output = retrieve(test_query, user_id=test_user)

    for r in output:
        print("\n---")
        print(r)