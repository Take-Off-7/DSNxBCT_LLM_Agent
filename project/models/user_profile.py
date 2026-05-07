import pandas as pd

reviews = pd.read_csv("data/processed/reviews.csv")
businesses = pd.read_csv("data/processed/businesses.csv")

# merge for category access
data = reviews.merge(businesses, on="business_id")

def build_user_profile(user_id):

    user_data = data[data["user_id"] == user_id]

    if len(user_data) == 0:
        return {
            "user_id": user_id,
            "average_rating": 3.5,
            "favorite_categories": [],
            "style": "neutral"
        }

    avg_rating = user_data["stars"].mean()

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

    # simple behaviour classification
    if avg_rating <= 2.5:
        style = "strict reviewer"
    elif avg_rating >= 4:
        style = "lenient reviewer"
    else:
        style = "balanced reviewer"

    return {
        "user_id": user_id,
        "average_rating": round(avg_rating, 2),
        "favorite_categories": categories,
        "style": style
    }