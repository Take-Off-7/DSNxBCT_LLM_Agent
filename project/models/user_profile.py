import os
import numpy as np
import pandas as pd
import sys
import json

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.behavior_model import (
    compute_rating_behavior,
    compute_category_bias,
    compute_linguistic_style,
    compute_sentiment_profile,
    infer_traits
)

# -------------------------------------------------
# DATA PATHS
# -------------------------------------------------
reviews_path = os.path.join(BASE_DIR, "data", "processed", "reviews.csv")
businesses_path = os.path.join(BASE_DIR, "data", "processed", "businesses.csv")

reviews = pd.read_csv(reviews_path)
businesses = pd.read_csv(businesses_path)

data = reviews.merge(businesses, on="business_id", how="left")


# -------------------------------------------------
# SAFE FLOAT
# -------------------------------------------------
def safe_float(x):
    try:
        if x is None:
            return 0.0
        return float(x)
    except:
        return 0.0


# -------------------------------------------------
# USER PROFILE BUILDER (OPTIMIZED)
# -------------------------------------------------
def build_user_profile(user_id):

    user_data = data[data["user_id"] == user_id]

    # -------------------------------------------------
    # COLD START PROFILE (IMPORTANT FOR TASK B)
    # -------------------------------------------------
    if user_data.empty:

        profile = {
            "user_id": user_id,

            "rating_behavior": {
                "avg_rating": 3.5,
                "variance": 0.5,
                "strictness": 0.5,
                "category_bias": {}
            },

            "linguistic_style": {
                "avg_review_length": 80,
                "emoji_rate": 0.0,
                "pidgin_usage": 0.0,
                "formality": 0.5
            },

            "sentiment_profile": {
                "avg_sentiment": 0.0,
                "positivity_bias": 0.5,
                "negativity_bias": 0.5
            }
        }

        profile["behavioral_traits"] = infer_traits(profile)

        profile["user_vector"] = np.array([
            3.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.8, 0.0, 0.0
        ], dtype=np.float32)

        return profile

    # -------------------------------------------------
    # REAL USER PROFILE
    # -------------------------------------------------
    rating_behavior = compute_rating_behavior(user_data)
    category_bias = compute_category_bias(user_data, businesses)
    linguistic_style = compute_linguistic_style(user_data)
    sentiment_profile = compute_sentiment_profile(user_data)

    rating_behavior["category_bias"] = category_bias

    profile = {
        "user_id": user_id,
        "rating_behavior": rating_behavior,
        "linguistic_style": linguistic_style,
        "sentiment_profile": sentiment_profile
    }

    profile["behavioral_traits"] = infer_traits(profile)

    # -------------------------------------------------
    # NORMALIZED USER VECTOR (CLEAN SIGNAL FOR RANKING)
    # -------------------------------------------------
    vec = np.array([
        safe_float(rating_behavior.get("avg_rating")),
        safe_float(rating_behavior.get("strictness")),
        safe_float(rating_behavior.get("variance")),
        safe_float(sentiment_profile.get("avg_sentiment")),
        safe_float(sentiment_profile.get("positivity_bias")),
        safe_float(sentiment_profile.get("negativity_bias")),
        safe_float(linguistic_style.get("avg_review_length")) / 100,
        safe_float(linguistic_style.get("emoji_rate")),
        safe_float(linguistic_style.get("pidgin_usage"))
    ], dtype=np.float32)

    norm = np.linalg.norm(vec) + 1e-8
    profile["user_vector"] = vec / norm

    return profile


# -------------------------------------------------
# TEST BLOCK
# -------------------------------------------------
if __name__ == "__main__":

    sample_user = reviews["user_id"].sample(1).values[0]

    print("\nTesting user:", sample_user)

    profile = build_user_profile(sample_user)

    print("\nBEHAVIORAL PROFILE:\n")
    print(json.dumps(
        {k: v for k, v in profile.items() if k != "user_vector"},
        indent=2
    ))

    print("\nUser Vector:", profile["user_vector"])