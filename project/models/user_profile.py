import os
import numpy as np
import pandas as pd

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
# USER PROFILE BUILDER
# -------------------------------------------------
def build_user_profile(user_id):

    user_data = data[
        data["user_id"] == user_id
    ]

    # -------------------------------------------------
    # Fallback profile
    # -------------------------------------------------
    if len(user_data) == 0:

        return {
            "user_id": user_id,
            "average_rating": 3.5,
            "rating_variance": 0.5,
            "review_length_avg": 80,
            "favorite_categories": [],
            "style": "balanced reviewer",
            "harshness": "unknown"
        }

    # -------------------------------------------------
    # BASIC STATS
    # -------------------------------------------------
    avg_rating = user_data["stars"].mean()
    rating_var = user_data["stars"].var()

    # -------------------------------------------------
    # CATEGORY PREFERENCES
    # -------------------------------------------------
    categories = (
        user_data["categories"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(5)
        .index
        .tolist()
    )

    # -------------------------------------------------
    # REVIEW STYLE SIGNALS
    # -------------------------------------------------
    review_lengths = user_data["text"].dropna().astype(str).apply(len)

    avg_length = (
        review_lengths.mean()
        if len(review_lengths) > 0
        else 80
    )

    # -------------------------------------------------
    # SENTIMENT / STRICTNESS MODELING
    # -------------------------------------------------
    if avg_rating <= 2.5:
        style = "strict reviewer"
        harshness = "high"
    elif avg_rating >= 4:
        style = "lenient reviewer"
        harshness = "low"
    else:
        style = "balanced reviewer"
        harshness = "medium"

    # -------------------------------------------------
    # CONSISTENCY SCORE (important for realism)
    # -------------------------------------------------
    consistency = (
        1 / (1 + (rating_var if not np.isnan(rating_var) else 0.5))
    )

    # -------------------------------------------------
    # FINAL PROFILE
    # -------------------------------------------------
    return {
        "user_id": user_id,

        # core signals
        "average_rating": round(avg_rating, 2),
        "rating_variance": round(
            rating_var if not np.isnan(rating_var) else 0.5,
            2
        ),

        # behavioral signals
        "review_length_avg": int(avg_length),
        "consistency_score": round(consistency, 3),

        # preference signals
        "favorite_categories": categories,

        # persona signals
        "style": style,
        "harshness": harshness
    }

# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":

    sample_user = reviews[
        "user_id"
    ].sample(1).values[0]

    print("\nTesting user:", sample_user)

    profile = build_user_profile(sample_user)

    print("\nPROFILE:\n")
    print(profile)