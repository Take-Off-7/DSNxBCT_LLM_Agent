import os
import sys
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------
# Load environment variables (.env)
# -------------------------------------------------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------
# Path setup
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile

reviews_path = os.path.join(BASE_DIR, "data/processed/reviews.csv")
businesses_path = os.path.join(BASE_DIR, "data/processed/businesses.csv")

reviews = pd.read_csv(reviews_path)
businesses = pd.read_csv(businesses_path)

# -------------------------------------------------
# LLM REVIEW GENERATOR
# -------------------------------------------------
def generate_review(user_id, business_id):

    profile = build_user_profile(user_id)

    biz = businesses[businesses["business_id"] == business_id]

    if len(biz) == 0:
        return {"error": "Business not found"}

    biz = biz.iloc[0]

    prompt = f"""
You are simulating a real Yelp user.

User profile:
- Average rating: {profile['average_rating']}
- Style: {profile['style']}
- Favorite categories: {profile['favorite_categories']}

Business:
- Name: {biz['name']}
- Categories: {biz['categories']}

Task:
1. Predict a star rating (1–5)
2. Write a realistic Yelp review

Return in JSON format:
{{
  "rating": number,
  "review": "text"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate realistic Yelp-style reviews."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )

        return {
            "user_id": user_id,
            "business_id": business_id,
            "result": response.choices[0].message.content
        }

    except Exception as e:
        # fallback (important for hackathon reliability)
        return {
            "user_id": user_id,
            "business_id": business_id,
            "error": str(e),
            "fallback_review": "System fallback triggered due to API issue."
        }

# -------------------------------------------------
# TEST RUN
# -------------------------------------------------
if __name__ == "__main__":

    sample_user = reviews["user_id"].sample(1).values[0]
    sample_business = businesses["business_id"].sample(1).values[0]

    print("\n🔥 LLM Review Generator Test\n")

    output = generate_review(sample_user, sample_business)

    print(output)