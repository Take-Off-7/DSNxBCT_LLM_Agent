import numpy as np
from sentence_transformers import SentenceTransformer
from models.user_profile import build_user_profile

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# INTENT DETECTION (AGENT STEP 1)
# -------------------------------------------------
def detect_intent(query: str):

    q = query.lower()

    if any(w in q for w in ["weekend", "go", "visit", "hangout", "place"]):
        return "experience"

    if any(w in q for w in ["eat", "food", "restaurant", "brunch"]):
        return "food"

    if any(w in q for w in ["fun", "activity", "do", "something"]):
        return "activity"

    if any(w in q for w in ["relax", "chill", "date"]):
        return "leisure"

    return "general"


# -------------------------------------------------
# USER INTELLIGENCE VECTOR
# -------------------------------------------------
def build_user_vector(profile):

    if not profile:
        return np.zeros(384)

    rating = profile.get("rating_behavior", {})
    sentiment = profile.get("sentiment_profile", {})
    traits = profile.get("behavioral_traits", {})

    text = f"""
    avg_rating:{rating.get('avg_rating',3)}
    strictness:{rating.get('strictness',0.5)}
    sentiment:{sentiment.get('avg_sentiment',0)}
    positive:{traits.get('positive_reviewer',False)}
    critical:{traits.get('critical_reviewer',False)}
    detail:{traits.get('detail_oriented',False)}
    """

    vec = embedder.encode(text)
    return vec / (np.linalg.norm(vec) + 1e-8)


# -------------------------------------------------
# ITEM VECTOR
# -------------------------------------------------
def build_item_vector(item):
    text = f"{item.get('business_name','')} {item.get('category','')} {item.get('text','')}"
    vec = embedder.encode(text)
    return vec / (np.linalg.norm(vec) + 1e-8)


# -------------------------------------------------
# EXPLANATION ENGINE (CRITICAL FOR TASK B)
# -------------------------------------------------
def explain(user_profile, item, intent):

    reasons = []

    rating = user_profile.get("rating_behavior", {})
    traits = user_profile.get("behavioral_traits", {})

    if rating.get("avg_rating", 3) >= 4:
        reasons.append("you tend to prefer high-quality experiences")

    if traits.get("positive_reviewer"):
        reasons.append("you generally leave positive feedback")

    if intent == "experience":
        reasons.append("you asked for a place to go this weekend")

    if intent == "food":
        reasons.append("you show interest in food-related experiences")

    if "restaurant" in item.get("category", ""):
        reasons.append("this is a highly rated food venue")

    if "nightlife" in item.get("category", ""):
        reasons.append("matches social/outing behavior patterns")

    return reasons[:3]


# -------------------------------------------------
# MAIN AGENTIC RECOMMENDER
# -------------------------------------------------
def get_recommendations(user_id, query, candidates, top_k=5):

    profile = build_user_profile(user_id)
    intent = detect_intent(query)

    user_vec = build_user_vector(profile)
    query_vec = embedder.encode(query)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    scored = []

    for item in candidates:

        item_vec = build_item_vector(item)

        # similarity signals
        user_sim = np.dot(user_vec, item_vec)
        query_sim = np.dot(query_vec, item_vec)

        stars = item.get("stars", 3) / 5

        # INTENT WEIGHTING (IMPORTANT FIX)
        if intent == "experience":
            score = 0.4 * user_sim + 0.4 * query_sim + 0.2 * stars

        elif intent == "food":
            score = 0.3 * user_sim + 0.5 * query_sim + 0.2 * stars

        elif intent == "activity":
            score = 0.5 * user_sim + 0.3 * query_sim + 0.2 * stars

        else:
            score = 0.4 * user_sim + 0.3 * query_sim + 0.3 * stars

        scored.append({
            "business_id": item.get("business_id"),
            "business_name": item.get("business_name"),
            "category": item.get("category"),
            "text": item.get("text"),
            "stars": item.get("stars"),
            "score": float(score),
            "reason": explain(profile, item, intent)
        })

    # sort
    scored.sort(key=lambda x: x["score"], reverse=True)

    # top 5 ONLY (STRICT)
    return scored[:top_k]