import re
import numpy as np
import pandas as pd
from textblob import TextBlob
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REVIEWS_PATH = os.path.join(BASE_DIR, "data", "processed", "reviews.csv")
BUSINESSES_PATH = os.path.join(BASE_DIR, "data", "processed", "businesses.csv")

reviews_df = pd.read_csv(REVIEWS_PATH)
businesses_df = pd.read_csv(BUSINESSES_PATH)


# -------------------------------------------------
# SAFE SCALE
# -------------------------------------------------
def scale(x, min_v=0, max_v=1):
    try:
        x = float(x)
    except:
        return 0.0
    return max(min((x - min_v) / (max_v - min_v + 1e-8), 1.0), 0.0)


# -------------------------------------------------
# SAFE CATEGORY COLUMN RESOLVER (🔥 FIX)
# -------------------------------------------------
def get_category_column(df: pd.DataFrame):
    if df is None:
        return None
    if "categories" in df.columns:
        return "categories"
    if "category" in df.columns:
        return "category"
    return None


# -------------------------------------------------
# 1. RATING BEHAVIOR
# -------------------------------------------------
def compute_rating_behavior(user_reviews: pd.DataFrame):

    ratings = user_reviews["stars"]

    avg_rating = float(ratings.mean())
    std_rating = float(ratings.std()) if len(ratings) > 1 else 0.0
    variance = float(ratings.var()) if len(ratings) > 1 else 0.0

    strictness = scale(1 - (std_rating / 2.5))

    return {
        "avg_rating": avg_rating,
        "variance": variance,
        "std": std_rating,
        "min_rating": int(ratings.min()),
        "max_rating": int(ratings.max()),
        "strictness": strictness
    }


# -------------------------------------------------
# 2. CATEGORY BIAS (🔥 FIXED SAFE VERSION)
# -------------------------------------------------
def compute_category_bias(user_reviews: pd.DataFrame, businesses: pd.DataFrame):

    if user_reviews is None or businesses is None:
        return {}

    cat_col = get_category_column(businesses)

    # ❗ HARD GUARD (THIS FIXES YOUR ERROR)
    if cat_col is None:
        return {}

    merged = user_reviews.merge(
        businesses,
        on="business_id",
        how="left"
    )

    if cat_col not in merged.columns:
        return {}

    merged[cat_col] = merged[cat_col].fillna("unknown")

    global_avg = merged["stars"].mean()
    category_bias = {}

    for cat in merged[cat_col].unique():

        cat_reviews = merged[merged[cat_col] == cat]

        if len(cat_reviews) < 2:
            continue

        bias = float(cat_reviews["stars"].mean() - global_avg)
        category_bias[cat] = scale(bias, -2, 2)

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

    pidgin_words = ["sha", "abeg", "no be", "wahala", "small small", "na", "omo"]

    pidgin_count = sum(any(p in t.lower() for p in pidgin_words) for t in texts)

    avg_length = total_words / max(total_reviews, 1)

    return {
        "avg_review_length": float(avg_length),
        "emoji_rate": scale(emoji_count / max(total_words, 1)),
        "pidgin_usage": scale(pidgin_count / max(total_reviews, 1)),
        "formality": scale(1 - (pidgin_count / max(total_reviews, 1))),
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
        "positivity_bias": scale(np.mean(sentiments > 0)),
        "negativity_bias": scale(np.mean(sentiments < 0)),
        "neutral_ratio": scale(np.mean(sentiments == 0))
    }


# -------------------------------------------------
# 5. TRAITS
# -------------------------------------------------
def infer_traits(profile: dict):

    ling = profile.get("linguistic_style", {})
    rating = profile.get("rating_behavior", {})
    sentiment = profile.get("sentiment_profile", {})

    return {
        "detail_oriented": scale(ling.get("avg_review_length", 0) / 40),
        "price_sensitive": scale(rating.get("strictness", 0)),
        "positive_reviewer": scale(sentiment.get("avg_sentiment", 0) + 0.5),
        "critical_reviewer": scale(0.5 - sentiment.get("avg_sentiment", 0)),
        "casual_style": scale(ling.get("pidgin_usage", 0))
    }


# -------------------------------------------------
# 6. MAIN BUILDER
# -------------------------------------------------
def build_behavior_profile(user_input, businesses=None):

    if isinstance(user_input, str):

        user_reviews = reviews_df[reviews_df["user_id"] == user_input]
        businesses = businesses_df

    else:
        user_reviews = user_input
        if businesses is None:
            businesses = businesses_df

    if len(user_reviews) == 0:
        return {
            "rating_behavior": {"avg_rating": 3.5, "strictness": 0.5},
            "linguistic_style": {},
            "sentiment_profile": {},
            "behavioral_traits": {}
        }

    rating_behavior = compute_rating_behavior(user_reviews)
    category_bias = compute_category_bias(user_reviews, businesses)
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