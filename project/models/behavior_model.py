import re
import numpy as np
import pandas as pd
from textblob import TextBlob
import os

# -------------------------------------------------
# Resolve paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REVIEWS_PATH = os.path.join(BASE_DIR, "data", "processed", "reviews.csv")
BUSINESSES_PATH = os.path.join(BASE_DIR, "data", "processed", "businesses.csv")

reviews_df = pd.read_csv(REVIEWS_PATH)
businesses_df = pd.read_csv(BUSINESSES_PATH)


# -------------------------------------------------
# 1. RATING BEHAVIOR
# -------------------------------------------------
def compute_rating_behavior(user_reviews: pd.DataFrame):
    ratings = user_reviews["stars"]

    avg_rating = float(ratings.mean())
    std_rating = float(ratings.std()) if len(ratings) > 1 else 0.0
    variance = float(ratings.var()) if len(ratings) > 1 else 0.0

    strictness = float(max(0, 1 - (std_rating / 2.0)))

    return {
        "avg_rating": avg_rating,
        "variance": variance,
        "std": std_rating,
        "min_rating": int(ratings.min()),
        "max_rating": int(ratings.max()),
        "strictness": strictness
    }


# -------------------------------------------------
# 2. CATEGORY BIAS
# -------------------------------------------------
def compute_category_bias(user_reviews: pd.DataFrame, businesses: pd.DataFrame):

    merged = user_reviews.merge(
        businesses,
        on="business_id",
        how="left"
    )

    if "categories" not in merged.columns:
        return {}

    merged["categories"] = merged["categories"].fillna("unknown")

    global_avg = merged["stars"].mean()
    category_bias = {}

    for cat in merged["categories"].unique():
        cat_reviews = merged[merged["categories"] == cat]

        if len(cat_reviews) == 0:
            continue

        category_bias[cat] = float(cat_reviews["stars"].mean() - global_avg)

    return category_bias


# -------------------------------------------------
# 3. LINGUISTIC STYLE
# -------------------------------------------------
def compute_linguistic_style(user_reviews: pd.DataFrame):

    texts = user_reviews["text"].fillna("").astype(str)

    total_reviews = len(texts)
    total_words = sum(len(t.split()) for t in texts)

    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)

    emoji_count = sum(len(emoji_pattern.findall(t)) for t in texts)

    pidgin_words = [
        "sha", "abeg", "no be", "wahala",
        "small small", "na", "omo", "e no easy"
    ]

    pidgin_count = sum(
        any(p in t.lower() for p in pidgin_words)
        for t in texts
    )

    avg_length = total_words / max(total_reviews, 1)

    return {
        "avg_review_length": float(avg_length),
        "emoji_rate": float(emoji_count / max(total_words, 1)),
        "pidgin_usage": float(pidgin_count / max(total_reviews, 1)),
        "formality": float(1 - (pidgin_count / max(total_reviews, 1))),
        "verbosity": "detailed" if avg_length > 25 else "concise"
    }


# -------------------------------------------------
# 4. SENTIMENT
# -------------------------------------------------
def compute_sentiment_profile(user_reviews: pd.DataFrame):

    sentiments = [
        TextBlob(str(t)).sentiment.polarity
        for t in user_reviews["text"].fillna("")
    ]

    sentiments = np.array(sentiments)

    if len(sentiments) == 0:
        return {}

    return {
        "avg_sentiment": float(sentiments.mean()),
        "positivity_bias": float(np.mean(sentiments > 0)),
        "negativity_bias": float(np.mean(sentiments < 0)),
        "neutral_ratio": float(np.mean(sentiments == 0))
    }


# -------------------------------------------------
# 5. TRAITS
# -------------------------------------------------
def infer_traits(profile: dict):

    ling = profile.get("linguistic_style", {})
    rating = profile.get("rating_behavior", {})
    sentiment = profile.get("sentiment_profile", {})

    return {
        "detail_oriented": ling.get("avg_review_length", 0) > 20,
        "price_sensitive": rating.get("strictness", 0) > 0.6,
        "positive_reviewer": sentiment.get("avg_sentiment", 0) > 0.2,
        "critical_reviewer": sentiment.get("avg_sentiment", 0) < -0.2,
        "casual_style": ling.get("pidgin_usage", 0) > 0.3
    }


# -------------------------------------------------
# 6. FINAL STEP 3 FIX (IMPORTANT)
# -------------------------------------------------
def build_behavior_profile(user_id):

    user_reviews = reviews_df[reviews_df["user_id"] == user_id]

    if len(user_reviews) == 0:
        return {
            "rating_behavior": {"avg_rating": 3.5, "strictness": 0.5},
            "linguistic_style": {},
            "sentiment_profile": {},
            "behavioral_traits": {
                "detail_oriented": False,
                "price_sensitive": False,
                "positive_reviewer": False,
                "critical_reviewer": False,
                "casual_style": False
            }
        }

    rating_behavior = compute_rating_behavior(user_reviews)
    category_bias = compute_category_bias(user_reviews, businesses_df)
    linguistic_style = compute_linguistic_style(user_reviews)
    sentiment_profile = compute_sentiment_profile(user_reviews)

    profile = {
        "rating_behavior": rating_behavior,
        "linguistic_style": linguistic_style,
        "sentiment_profile": sentiment_profile
    }

    profile["rating_behavior"]["category_bias"] = category_bias
    profile["behavioral_traits"] = infer_traits(profile)

    return profile