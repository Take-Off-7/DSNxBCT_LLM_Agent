import os
import numpy as np
import pandas as pd
import sys
import json

from utils.data_loader import load_reviews, load_businesses
from models.behavior_model import (
    compute_rating_behavior,
    compute_category_bias,
    compute_linguistic_style,
    compute_sentiment_profile,
    infer_traits
)

# -------------------------------------------------
# LOAD DATA (CENTRALIZED)
# -------------------------------------------------
reviews = load_reviews()
businesses = load_businesses()

data = reviews.merge(businesses, on="business_id", how="left")


# -------------------------------------------------
# SAFE FLOAT
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


# -------------------------------------------------
# USER PROFILE BUILDER
# -------------------------------------------------
def build_user_profile(user_id):

    user_data = data[data["user_id"] == user_id]

    if user_data.empty:
        return {
            "user_id": user_id,
            "rating_behavior": {"avg_rating": 3.5, "strictness": 0.5},
            "linguistic_style": {},
            "sentiment_profile": {},
            "behavioral_traits": {},
            "user_vector": np.array([3.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0])
        }

    try:
        from models.behavior_model import build_behavior_profile

        profile = build_behavior_profile(user_id)

    except Exception as e:
        print("PROFILE ERROR:", e)

        return {
            "user_id": user_id,
            "rating_behavior": {"avg_rating": 3.5, "strictness": 0.5},
            "linguistic_style": {},
            "sentiment_profile": {},
            "behavioral_traits": {},
            "user_vector": np.array([3.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0])
        }

    return profile