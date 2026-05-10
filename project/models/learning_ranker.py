import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

WEIGHTS_PATH = "project/data/learning_weights.json"


# -------------------------------------------------
# LOAD / INIT WEIGHTS
# -------------------------------------------------
def load_weights():
    if os.path.exists(WEIGHTS_PATH):
        try:
            with open(WEIGHTS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "user_match": 1.25,
        "rating_boost": 0.6,
        "novelty_boost": 0.9,
        "diversity_penalty": 0.35
    }


def save_weights(weights):
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)


# -------------------------------------------------
# COSINE
# -------------------------------------------------
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# -------------------------------------------------
# CORE SCORING FUNCTION
# -------------------------------------------------
def score_item(user_vec, item_vec, item, seen, weights, selected_vecs=None):

    user_sim = cosine(user_vec, item_vec)

    rating = float(item.get("stars", 3)) / 5.0

    bid = str(item.get("business_id", ""))

    novelty = 0.0 if bid in seen else 1.0

    # diversity penalty (MMR-style)
    diversity_penalty = 0.0
    if selected_vecs:
        diversity_penalty = max(cosine(item_vec, v) for v in selected_vecs)

    score = (
        weights["user_match"] * user_sim +
        weights["rating_boost"] * rating +
        weights["novelty_boost"] * novelty -
        weights["diversity_penalty"] * diversity_penalty
    )

    return float(score)


# -------------------------------------------------
# MAIN AGENTIC RANKER
# -------------------------------------------------
def rank_businesses(user_id, query, candidates, user_profile=None, user_memory=None):

    if not candidates:
        return []

    weights = load_weights()

    # safe seen history
    seen = set()
    if isinstance(user_memory, dict):
        seen = set(user_memory.get("seen_businesses", []))

    # -------------------------------------------------
    # USER VECTOR (FIXED + STABLE)
    # avoids user collapse bug
    # -------------------------------------------------
    if user_profile:
        base_text = json.dumps(user_profile, default=str)
    else:
        base_text = query

    # inject identity to avoid cross-user leakage
    base_text = f"{base_text} <USER:{user_id}>"

    user_vec = embedder.encode(base_text)
    user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-9)

    ranked = []
    selected_vecs = []

    # -------------------------------------------------
    # SCORING LOOP
    # -------------------------------------------------
    for item in candidates:

        text = f"{item.get('business_name','')} {item.get('text','')}"
        item_vec = embedder.encode(text)
        item_vec = item_vec / (np.linalg.norm(item_vec) + 1e-9)

        score = score_item(
            user_vec=user_vec,
            item_vec=item_vec,
            item=item,
            seen=seen,
            weights=weights,
            selected_vecs=selected_vecs
        )

        ranked.append((score, item, item_vec))

    # sort
    ranked.sort(key=lambda x: x[0], reverse=True)

    # -------------------------------------------------
    # DIVERSIFIED TOP-K (MMR-lite)
    # -------------------------------------------------
    final = []
    selected_vecs = []

    for score, item, vec in ranked:

        final.append(item)
        selected_vecs.append(vec)

        if len(final) >= 5:
            break

    return final


# -------------------------------------------------
# BACKWARD COMPATIBILITY
# -------------------------------------------------
def rank_items(items, user_vec, seen=None):

    if seen is None:
        seen = set()

    weights = load_weights()

    scored = []

    for item in items:

        text = f"{item.get('business_name','')} {item.get('text','')}"
        item_vec = embedder.encode(text)
        item_vec = item_vec / (np.linalg.norm(item_vec) + 1e-9)

        score = score_item(
            user_vec=user_vec,
            item_vec=item_vec,
            item=item,
            seen=seen,
            weights=weights
        )

        item["score"] = score
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in scored[:5]]