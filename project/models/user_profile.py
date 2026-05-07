import os
import pandas as pd

# resolve project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

reviews_path = os.path.join(BASE_DIR, "data/processed/reviews.csv")
businesses_path = os.path.join(BASE_DIR, "data/processed/businesses.csv")

if not os.path.exists(reviews_path):
    raise FileNotFoundError(f"Missing: {reviews_path}")

reviews = pd.read_csv(reviews_path)
businesses = pd.read_csv(businesses_path)

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


if __name__ == "__main__":

    df = pd.read_csv(reviews_path)

    sample_user = df["user_id"].sample(1).values[0]

    print("Testing user:", sample_user)

    profile = build_user_profile(sample_user)

    print(profile)