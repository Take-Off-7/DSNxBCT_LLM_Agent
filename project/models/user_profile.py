import os
import numpy as np
import pandas as pd

import sys
import os

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
# Resolve project root
# -------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

reviews_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "reviews.csv"
)

businesses_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "businesses.csv"
)

# -------------------------------------------------
# Load data
# -------------------------------------------------
if not os.path.exists(reviews_path):
    raise FileNotFoundError(f"Missing: {reviews_path}")

reviews = pd.read_csv(reviews_path)
businesses = pd.read_csv(businesses_path)

data = reviews.merge(
    businesses,
    on="business_id",
    how="left"
)

# -------------------------------------------------
# USER PROFILE BUILDER (STEP 2 UPGRADED)
# -------------------------------------------------
def build_user_profile(user_id):

    user_data = data[
        data["user_id"] == user_id
    ]

    # -------------------------------------------------
    # Fallback (still upgraded to behavioral structure)
    # -------------------------------------------------
    if len(user_data) == 0:

        return {
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
                "formality": 0.5,
                "verbosity": "concise"
            },

            "sentiment_profile": {
                "avg_sentiment": 0.0,
                "positivity_bias": 0.5,
                "negativity_bias": 0.5
            },

            "behavioral_traits": {
                "detail_oriented": False,
                "price_sensitive": False,
                "positive_reviewer": False,
                "critical_reviewer": False,
                "casual_style": False
            }
        }

    # -------------------------------------------------
    # CORE BEHAVIORAL MODULES
    # -------------------------------------------------
    rating_behavior = compute_rating_behavior(user_data)

    category_bias = compute_category_bias(
        user_data,
        businesses
    )

    linguistic_style = compute_linguistic_style(user_data)

    sentiment_profile = compute_sentiment_profile(user_data)

    # attach category bias into rating behavior
    rating_behavior["category_bias"] = category_bias

    # full profile
    profile = {
        "user_id": user_id,
        "rating_behavior": rating_behavior,
        "linguistic_style": linguistic_style,
        "sentiment_profile": sentiment_profile
    }

    # behavioral traits (derived layer)
    profile["behavioral_traits"] = infer_traits(profile)

    return profile

# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":

    sample_user = reviews[
        "user_id"
    ].sample(1).values[0]

    print("\nTesting user:", sample_user)

    profile = build_user_profile(sample_user)

    print("\nBEHAVIORAL PROFILE:\n")
    import json
    print(json.dumps(profile, indent=2))