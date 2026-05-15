import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from retrieval.retrieve import retrieve
from models.user_profile import build_user_profile

# -------------------------------------------------
# MODEL
# -------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------------------------
# BUILD CATEGORY SPACE (NO HARDCODING)
# -------------------------------------------------
def build_category_space(businesses_df):
    """
    Creates semantic embeddings for all categories in dataset.
    """
    category_map = {}

    for _, row in businesses_df.iterrows():
        cats = str(row.get("categories", "")).lower().split(",")

        for c in cats:
            c = c.strip()
            if c and c not in category_map:
                category_map[c] = embedder.encode(c)

    return category_map


# -------------------------------------------------
# QUERY UNDERSTANDING (SEMANTIC ONLY)
# -------------------------------------------------
def infer_query_vector(query):
    return embedder.encode(query)


# -------------------------------------------------
# CATEGORY MATCH SCORE
# -------------------------------------------------
def category_alignment_score(query_vec, category_vec):
    if category_vec is None:
        return 0.0

    return float(np.dot(query_vec, category_vec) /
                 (np.linalg.norm(query_vec) * np.linalg.norm(category_vec) + 1e-9))


# -------------------------------------------------
# MAIN RECOMMENDER AGENT
# -------------------------------------------------
def recommend(query, user_id=None, businesses_df=None, k=5):
    """
    Agentic recommender:
    - no hardcoded intent rules
    - semantic category inference
    - retrieval + reranking + personalization
    """

    # ---------------------------
    # STEP 1: USER PROFILE
    # ---------------------------
    try:
        profile = build_user_profile(user_id) if user_id else None
    except:
        profile = None

    # ---------------------------
    # STEP 2: SEMANTIC QUERY VECTOR
    # ---------------------------
    query_vec = infer_query_vector(query)

    # ---------------------------
    # STEP 3: CATEGORY SPACE (DYNAMIC)
    # ---------------------------
    category_space = build_category_space(businesses_df) if businesses_df is not None else {}

    # infer closest categories (soft matching)
    category_scores = {}
    for cat, vec in category_space.items():
        category_scores[cat] = category_alignment_score(query_vec, vec)

    top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_categories = [c[0] for c in top_categories]

    # ---------------------------
    # STEP 4: BASE RETRIEVAL
    # ---------------------------
    raw = retrieve(query=query, user_id=user_id, k=30)
    candidates = raw.get("results", [])

    if not candidates:
        return {
            "query": query,
            "recommendations": []
        }

    # ---------------------------
    # STEP 5: RERANKING (AGENTIC SCORING)
    # ---------------------------
    final = []

    for item in candidates:

        item_text = f"{item['business_name']} {item['category']} {item['text']}"
        item_vec = embedder.encode(item_text)

        # semantic similarity
        sim = float(np.dot(query_vec, item_vec) /
                    (np.linalg.norm(query_vec) * np.linalg.norm(item_vec) + 1e-9))

        # category bonus (soft, not hardcoded)
        cat_bonus = 0.0
        for tc in top_categories:
            if tc in item["category"].lower():
                cat_bonus += 0.10

        # base score
        score = item["score"]

        # final blended score
        score = (
            0.45 * sim +
            0.35 * score +
            0.20 * cat_bonus
        )

        item["final_score"] = score
        final.append(item)

    # ---------------------------
    # STEP 6: SORT
    # ---------------------------
    final.sort(key=lambda x: x["final_score"], reverse=True)

    # ---------------------------
    # STEP 7: RETURN TOP-K
    # ---------------------------
    return {
        "query": query,
        "user_id": user_id,
        "inferred_categories": top_categories,
        "recommendations": final[:k]
    }