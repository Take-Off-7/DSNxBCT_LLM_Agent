import os
import sys
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile
from utils.data_loader import load_businesses

# -------------------------------------------------
# MODEL
# -------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# DATA PATHS
# -------------------------------------------------
META_PATH = os.path.join(BASE_DIR, "data", "embeddings", "reviews_with_ids.csv")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "faiss.index")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv(META_PATH)

businesses = load_businesses()
businesses.columns = businesses.columns.str.strip()

# -------------------------------------------------
# SAFE BUSINESS MAPS
# -------------------------------------------------
if "business_id" in businesses.columns:
    businesses["business_id"] = businesses["business_id"].astype(str).str.strip()
else:
    raise ValueError("businesses.csv missing 'business_id' column")

business_name_map = dict(zip(
    businesses["business_id"],
    businesses.get("name", "").fillna("Unknown")
))

business_cat_map = dict(zip(
    businesses["business_id"],
    businesses.get("categories", "").fillna("").astype(str).str.strip().str.lower()
))

# -------------------------------------------------
# LOAD FAISS INDEX
# -------------------------------------------------
index = faiss.read_index(INDEX_PATH)

# -------------------------------------------------
# SAFE UTILITIES
# -------------------------------------------------
def safe_float(x):
    try:
        if x is None:
            return 0.0
        if isinstance(x, float) and np.isnan(x):
            return 0.0
        return float(x)
    except:
        return 0.0


def l2norm(v):
    v = np.array(v)
    return v / (np.linalg.norm(v) + 1e-8)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# -------------------------------------------------
# EMBEDDING
# -------------------------------------------------
def embed_item(text: str):
    return l2norm(model.encode([str(text)])[0])


# -------------------------------------------------
# BASE SCORE
# -------------------------------------------------
def base_score(dist):
    return 1 / (1 + float(dist))


# -------------------------------------------------
# USER VECTOR (SAFE)
# -------------------------------------------------
def build_user_vector(profile):
    traits = profile.get("behavioral_traits", {})
    rating = profile.get("rating_behavior", {})
    sentiment = profile.get("sentiment_profile", {})

    vec = np.array([
        safe_float(rating.get("avg_rating", 3)),
        safe_float(rating.get("strictness", 0.5)),
        safe_float(rating.get("variance", 0.5)),
        safe_float(sentiment.get("avg_sentiment", 0)),
        float(traits.get("positive_reviewer", 0)),
        float(traits.get("critical_reviewer", 0)),
        float(traits.get("detail_oriented", 0))
    ], dtype=np.float32)

    return l2norm(vec)


# -------------------------------------------------
# USER SCORING
# -------------------------------------------------
def compute_user_score(item, profile):

    score = safe_float(item.get("score", 0))

    traits = profile.get("behavioral_traits", {})
    sentiment = profile.get("sentiment_profile", {})
    rating = profile.get("rating_behavior", {})

    rating_gap = abs(
        safe_float(item.get("stars", 3)) -
        safe_float(rating.get("avg_rating", 3))
    )

    score *= (1 - rating_gap * 0.20)
    score *= (1 + safe_float(rating.get("strictness", 0.5)) * 0.25)
    score *= (1 + safe_float(sentiment.get("avg_sentiment", 0)) * 0.15)

    if traits.get("positive_reviewer"):
        score *= 1.08
    if traits.get("critical_reviewer"):
        score *= 0.85
    if traits.get("detail_oriented"):
        score *= 1.10

    return float(score)


# -------------------------------------------------
# CATEGORY BOOST (SAFE)
# -------------------------------------------------
def category_boost(category, profile):

    if not category:
        return 1.0

    category = str(category).lower()

    traits = profile.get("behavioral_traits", {})
    rating = profile.get("rating_behavior", {})

    boost = 1.0

    if rating.get("strictness", 0) > 0.65:
        if "restaurant" in category or "hotel" in category:
            boost += 0.06

    if traits.get("detail_oriented"):
        if "luxury" in category or "spa" in category:
            boost += 0.10

    if traits.get("positive_reviewer"):
        if "bar" in category or "restaurant" in category:
            boost += 0.06

    return boost


# -------------------------------------------------
# FIXED: USER-COMPATIBILITY SCORE (NO EMBEDDING MIX)
# -------------------------------------------------
def user_compatibility_score(item, profile):
    """
    FIX: replaces invalid cosine(user_vec, item_vec)
    with scalar behavioral compatibility only
    """

    if not profile:
        return 0.0

    rating = profile.get("rating_behavior", {})
    sentiment = profile.get("sentiment_profile", {})

    rating_bias = abs(
        safe_float(item.get("stars", 3)) -
        safe_float(rating.get("avg_rating", 3))
    )

    sentiment_bias = abs(safe_float(sentiment.get("avg_sentiment", 0)))

    return float(1.0 - (rating_bias * 0.2) + (sentiment_bias * 0.1))


# -------------------------------------------------
# DIVERSITY (MMR)
# -------------------------------------------------
def diversify(results, k):

    if not results:
        return []

    selected = []
    candidates = results.copy()

    while len(selected) < k and candidates:

        best = None
        best_score = -1

        for c in candidates:

            relevance = c["score"]

            if not selected:
                diversity_penalty = 0
            else:
                diversity_penalty = max(
                    cosine(c["vec"], s["vec"])
                    for s in selected
                )

            mmr = 0.75 * relevance - 0.25 * diversity_penalty

            if mmr > best_score:
                best_score = mmr
                best = c

        selected.append(best)
        candidates.remove(best)

    return selected


# -------------------------------------------------
# MAIN RETRIEVE FUNCTION
# -------------------------------------------------
def retrieve(query, user_id=None, k=5, use_llm=True):

    query_emb = embed_item(query)

    profile = None
    user_vec = None

    if user_id:
        try:
            profile = build_user_profile(user_id)
            user_vec = build_user_vector(profile)
        except Exception:
            profile = None

    distances, indices = index.search(query_emb.reshape(1, -1), k * 30)

    results = []

    for idx, dist in zip(indices[0], distances[0]):

        if idx < 0 or idx >= len(df):
            continue

        row = df.iloc[idx]

        bid = str(row.get("business_id", "")).strip()
        category = business_cat_map.get(bid, "")

        review_score = base_score(float(dist))
        text = str(row.get("text", ""))
        item_vec = embed_item(text)

        semantic_alignment = cosine(query_emb, item_vec)

        score = review_score + semantic_alignment * 0.30

        results.append({
            "business_id": bid,
            "business_name": business_name_map.get(bid, "Unknown"),
            "category": category,
            "text": text,
            "stars": safe_float(row.get("stars", 0)),
            "score": float(score),
            "vec": item_vec
        })

    # -------------------------------------------------
    # PERSONALIZATION (FIXED)
    # -------------------------------------------------
    if profile:

        for r in results:

            r["score"] = (
                compute_user_score(r, profile) * 0.70 +
                r["score"] * 0.30
            )

            # FIX: removed invalid cosine(user_vec, item_vec)
            r["score"] += user_compatibility_score(r, profile) * 0.15

            r["score"] *= category_boost(r.get("category", ""), profile)

    results.sort(key=lambda x: x["score"], reverse=True)

    final = diversify(results, k)

    return {
        "query": str(query),
        "results": [
            {
                "business_id": str(r["business_id"]),
                "business_name": str(r["business_name"]),
                "category": str(r["category"]),
                "text": str(r["text"]),
                "stars": float(r["stars"]),
                "score": float(r["score"])
            }
            for r in final
        ]
    }