import os
import sys
import numpy as np
import pandas as pd
import faiss

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from sentence_transformers import SentenceTransformer
from models.behavior_model import build_behavior_profile
from rag.response_generator import generate_response


# -------------------------------------------------
# PATHS
# -------------------------------------------------
META_PATH = os.path.join(BASE_DIR, "data", "embeddings", "reviews_with_ids.csv")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "faiss.index")
BUSINESS_PATH = os.path.join(BASE_DIR, "data", "processed", "businesses.csv")


# -------------------------------------------------
# LOAD
# -------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv(META_PATH)
index = faiss.read_index(INDEX_PATH)
businesses = pd.read_csv(BUSINESS_PATH)


# -------------------------------------------------
# SAFE SCORING
# -------------------------------------------------
def compute_behavior_score(item, profile):

    score = item.get("score", 0)

    traits = profile.get("behavioral_traits", {})
    sentiment = profile.get("sentiment_profile", {})
    rating = profile.get("rating_behavior", {})

    if traits.get("positive_reviewer"):
        score *= 1.05

    if traits.get("critical_reviewer"):
        score *= 0.95

    if traits.get("detail_oriented"):
        score *= 1.03

    score *= (1 + rating.get("strictness", 0) * 0.02)
    score *= (1 + sentiment.get("avg_sentiment", 0) * 0.01)

    return score


# -------------------------------------------------
# MAIN RETRIEVE
# -------------------------------------------------
def retrieve(query, user_id=None, k=5, use_llm=True):

    # -------------------------
    # EMBEDDING SEARCH
    # -------------------------
    query_emb = model.encode([query])
    query_emb = np.array(query_emb).astype("float32")

    distances, indices = index.search(query_emb, k * 5)

    results = []

    # -------------------------
    # SAFE LOOP (FIXES CRASHES)
    # -------------------------
    if indices is None:
        return []

    for idx, dist in zip(indices[0], distances[0]):

        if idx is None or idx < 0 or idx >= len(df):
            continue

        row = df.iloc[idx].to_dict()

        results.append({
            "user_id": row.get("user_id"),
            "business_id": row.get("business_id"),
            "text": row.get("text"),
            "stars": row.get("stars"),
            "score": float(1 / (1 + dist))
        })

    # -------------------------
    # PROFILE RERANKING
    # -------------------------
    if user_id and len(results) > 0:

        try:
            profile = build_behavior_profile(user_id)

            for r in results:
                r["score"] = compute_behavior_score(r, profile)

        except Exception as e:
            print("Profile error:", e)

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top_results = results[:k]

    # -------------------------
    # LLM LAYER (SAFE WRAP)
    # -------------------------
    if use_llm and user_id:

        try:
            llm_response = generate_response(query, top_results, user_id)

        except Exception as e:
            llm_response = {
                "success": False,
                "error": str(e)
            }

        return {
            "results": top_results,
            "llm": llm_response
        }

    # -------------------------
    # FALLBACK MODE
    # -------------------------
    return {
        "results": top_results,
        "llm": None
    }


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":

    test_query = "good food but slow service"
    test_user = df["user_id"].sample(1).values[0]

    output = retrieve(test_query, user_id=test_user, use_llm=False)

    print(output)