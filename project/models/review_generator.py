import os
import sys
import json
import pandas as pd
import requests
from dotenv import load_dotenv

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile

# -------------------------------------------------
# Data paths
# -------------------------------------------------
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

reviews = pd.read_csv(reviews_path)
businesses = pd.read_csv(businesses_path)

# -------------------------------------------------
# Ollama config (FIXED)
# -------------------------------------------------
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

# ✅ IMPORTANT FIX: default model is now llama3.2:1b
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:1b"
)

# -------------------------------------------------
# CLEAN JSON OUTPUT
# -------------------------------------------------
def clean_json_response(text):

    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "")

    if text.startswith("```"):
        text = text.replace("```", "")

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()

# -------------------------------------------------
# CALL OLLAMA
# -------------------------------------------------
def call_ollama(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,   # ✅ FIXED HERE
            "prompt": prompt,
            "stream": False
        },
        timeout=60   # reduced for hackathon stability
    )

    if response.status_code != 200:
        raise Exception(
            f"Ollama Error: {response.text}"
        )

    data = response.json()

    return data.get("response", "")

# -------------------------------------------------
# REVIEW GENERATOR
# -------------------------------------------------
def generate_review(user_id, business_id):

    profile = build_user_profile(user_id)

    biz = businesses[
        businesses["business_id"] == business_id
    ]

    if len(biz) == 0:
        return {
            "error": "Business not found",
            "business_id": business_id
        }

    biz = biz.iloc[0]

    prompt = f"""
You are simulating a realistic Nigerian-style Yelp reviewer.

User Profile:
- Avg Rating: {profile['average_rating']}
- Style: {profile['style']}
- Harshness: {profile.get('harshness', 'balanced')}
- Verbosity: {profile.get('verbosity', 'concise')}
- Favorite Categories: {profile['favorite_categories']}

Business:
- Name: {biz['name']}
- Categories: {biz['categories']}

TASK:
1. Predict rating (1-5)
2. Write realistic review

Return ONLY valid JSON:
{{
  "rating": 4,
  "review": "..."
}}
"""

    try:

        raw_output = call_ollama(prompt)
        cleaned_output = clean_json_response(raw_output)

        try:
            parsed = json.loads(cleaned_output)

            return {
                "success": True,
                "user_id": user_id,
                "business_id": business_id,
                "business_name": biz["name"],
                "rating": parsed.get("rating"),
                "review": parsed.get("review"),
                "model": OLLAMA_MODEL
            }

        except Exception:

            return {
                "success": False,
                "error": "JSON parsing failed",
                "raw_output": raw_output
            }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "fallback_review": "Service was okay overall."
        }

# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":

    sample_user = reviews["user_id"].sample(1).values[0]
    sample_business = businesses["business_id"].sample(1).values[0]

    print("\n🔥 TESTING REVIEW GENERATOR\n")

    output = generate_review(sample_user, sample_business)

    print(json.dumps(output, indent=2))