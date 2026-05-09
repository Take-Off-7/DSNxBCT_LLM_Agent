import os
import sys
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile

# -------------------------------------------------
# MODEL
# -------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# PATHS
# -------------------------------------------------
META_PATH = os.path.join(BASE_DIR, "data", "embeddings", "reviews_with_ids.csv")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "faiss.index")
BUSINESS_PATH = os.path.join(BASE_DIR, "data", "processed", "businesses.csv")

df = pd.read_csv(META_PATH)
index = faiss.read_index(INDEX_PATH)
businesses = pd.read_csv(BUSINESS_PATH)

businesses["business_id"] = businesses["business_id"].astype(str).str.strip()
business_map = dict(zip(businesses["business_id"], businesses["name"]))


# -------------------------------------------------
# BASE SCORE
# -------------------------------------------------
def base_score(dist):
    return float(1 / (1 + dist))


# -------------------------------------------------
# USER VECTOR
# -------------------------------------------------
def build_user_vector(profile):

    traits = profile.get("behavioral_traits", {})
    rating = profile.get("rating_behavior", {})
    sentiment = profile.get("sentiment_profile", {})

    vec = np.array([
        rating.get("avg_rating", 3),
        rating.get("strictness", 0.5),
        sentiment.get("avg_sentiment", 0),
        float(traits.get("positive_reviewer", False)),
        float(traits.get("critical_reviewer", False)),
        float(traits.get("detail_oriented", False))
    ])

    return vec / (np.linalg.norm(vec) + 1e-8)


# -------------------------------------------------
# USER SCORING (FIXED STABILITY)
# -------------------------------------------------
def compute_user_score(item, profile):

    score = item["score"]

    traits = profile.get("behavioral_traits", {})
    sentiment = profile.get("sentiment_profile", {})
    rating = profile.get("rating_behavior", {})

    strictness = rating.get("strictness", 0.5)
    avg_rating = rating.get("avg_rating", 3)

    # rating alignment penalty (IMPORTANT FIX)
    rating_gap = abs(item.get("stars", 3) - avg_rating)
    score -= rating_gap * 0.12

    # behavioral modifiers
    score *= (1 + strictness * 0.20)
    score *= (1 + sentiment.get("avg_sentiment", 0) * 0.10)

    if traits.get("positive_reviewer"):
        score *= 1.03

    if traits.get("critical_reviewer"):
        score *= 0.92

    if traits.get("detail_oriented"):
        score *= 1.05

    return score


# -------------------------------------------------
# MAIN RETRIEVE ENGINE (IMPROVED)
# -------------------------------------------------
def retrieve(query, user_id=None, k=5, use_llm=True):

    query_emb = model.encode([query]).astype("float32")

    distances, indices = index.search(query_emb, k * 20)

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
            "stars": float(row.get("stars", 0)),
            "score": base_score(dist)
        })

    # -------------------------------------------------
    # PERSONALIZATION LAYER (FIXED)
    # -------------------------------------------------
    profile = None
    user_vec = None

    if user_id:
        try:
            profile = build_user_profile(user_id)
            user_vec = build_user_vector(profile)

            for r in results:

                # core personalization
                r["score"] = compute_user_score(r, profile)

                # semantic alignment boost (stabilized)
                item_vec = model.encode([r["text"]])[0]
                item_vec = item_vec / (np.linalg.norm(item_vec) + 1e-8)

                similarity = np.dot(user_vec, item_vec)

                r["score"] += similarity * 0.12

        except Exception as e:
            print("Profile error:", e)

    # -------------------------------------------------
    # FINAL SORT
    # -------------------------------------------------
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # -------------------------------------------------
    # IMPROVED DIVERSITY (FIXED BUGGY LOGIC)
    # -------------------------------------------------
    final = []
    seen_business = set()
    seen_prefix = set()

    for r in results:

        bid = r["business_id"]
        prefix = r["business_name"].split()[0].lower()

        # avoid duplicates
        if bid in seen_business:
            continue

        # avoid over-clustering same category
        if prefix in seen_prefix and len(final) >= k:
            continue

        seen_business.add(bid)
        seen_prefix.add(prefix)

        final.append(r)

        if len(final) >= k:
            break

    if len(final) < k:
        final = results[:k]

    return {
        "results": final,
        "raw_ranked": results
    }