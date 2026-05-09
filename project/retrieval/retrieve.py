import os
import sys
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
from models.user_profile import build_user_profile
from rag.response_generator import generate_response

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

META_PATH = os.path.join(BASE_DIR, "data", "embeddings", "reviews_with_ids.csv")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "faiss.index")
BUSINESS_PATH = os.path.join(BASE_DIR, "data", "processed", "businesses.csv")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv(META_PATH)
index = faiss.read_index(INDEX_PATH)
businesses = pd.read_csv(BUSINESS_PATH)

businesses["business_id"] = businesses["business_id"].astype(str).str.strip()

business_map = dict(zip(businesses["business_id"], businesses["name"]))


# -------------------------------------------------
# BASE SIMILARITY SCORE
# -------------------------------------------------
def base_score(dist):
    return float(1 / (1 + dist))


# -------------------------------------------------
# PERSONALIZED RERANKING
# -------------------------------------------------
def compute_behavior_score(item, profile):

    score = item["score"]

    traits = profile.get("behavioral_traits", {})
    rating = profile.get("rating_behavior", {})
    sentiment = profile.get("sentiment_profile", {})
    style = profile.get("linguistic_style", {})

    # -------------------------------------------------
    # Trait-based adjustments
    # -------------------------------------------------
    if traits.get("positive_reviewer"):
        score *= 1.05

    if traits.get("critical_reviewer"):
        score *= 0.90

    if traits.get("detail_oriented"):
        score *= 1.08

    if traits.get("casual_style"):
        score *= 1.02

    # -------------------------------------------------
    # Strictness adjustment (important fix)
    # -------------------------------------------------
    strictness = rating.get("strictness", 0.5)
    score *= (1 + strictness * 0.15)

    # -------------------------------------------------
    # Sentiment alignment
    # -------------------------------------------------
    avg_sentiment = sentiment.get("avg_sentiment", 0)
    score *= (1 + avg_sentiment * 0.10)

    # -------------------------------------------------
    # Style alignment (verbosity preference)
    # -------------------------------------------------
    if style.get("verbosity") == "detailed":
        score *= 1.03

    return score


# -------------------------------------------------
# MAIN RETRIEVE FUNCTION
# -------------------------------------------------
def retrieve(query, user_id=None, k=5, use_llm=True):

    # -------------------------------------------------
    # EMBEDDING SEARCH
    # -------------------------------------------------
    query_emb = model.encode([query]).astype("float32")

    distances, indices = index.search(query_emb, k * 10)

    results = []

    for idx, dist in zip(indices[0], distances[0]):

        if idx < 0 or idx >= len(df):
            continue

        row = df.iloc[idx]

        bid = str(row.get("business_id", "")).strip()

        results.append({
            "business_id": bid,
            "business_name": business_map.get(bid, "Unknown"),
            "text": row.get("text", ""),
            "stars": row.get("stars", 0),
            "score": base_score(dist)
        })

    # -------------------------------------------------
    # PROFILE RERANKING (IMPORTANT FIX)
    # -------------------------------------------------
    if user_id:

        try:
            profile = build_user_profile(user_id)

            for r in results:
                r["score"] = compute_behavior_score(r, profile)

        except Exception as e:
            print("Profile error:", e)

    # -------------------------------------------------
    # FINAL SORTING
    # -------------------------------------------------
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    top_results = results[:k]

    # -------------------------------------------------
    # LLM RESPONSE
    # -------------------------------------------------
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

    print(retrieve(test_query, user_id=test_user))