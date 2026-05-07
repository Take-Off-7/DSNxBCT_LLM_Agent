import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile

reviews_path = os.path.join(BASE_DIR, "data/processed/reviews.csv")
businesses_path = os.path.join(BASE_DIR, "data/processed/businesses.csv")

reviews = pd.read_csv(reviews_path)
businesses = pd.read_csv(businesses_path)


def generate_review(user_id, business_id):

    profile = build_user_profile(user_id)

    biz = businesses[businesses["business_id"] == business_id]

    if len(biz) == 0:
        return {"error": "Business not found"}

    biz = biz.iloc[0]

    # rating logic (baseline)
    if profile["style"] == "strict reviewer":
        rating = 3
    elif profile["style"] == "lenient reviewer":
        rating = 5
    else:
        rating = 4

    review_text = f"""
I visited {biz['name']}.

As a {profile['style']}, I found the experience moderate.

I usually enjoy {', '.join(profile['favorite_categories'])}.

Overall, it aligns with my preferences but could improve.
""".strip()

    return {
        "user_id": user_id,
        "business_id": business_id,
        "rating": rating,
        "review": review_text
    }


# test run
if __name__ == "__main__":

    sample_user = reviews["user_id"].sample(1).values[0]
    sample_business = businesses["business_id"].sample(1).values[0]

    print(generate_review(sample_user, sample_business))