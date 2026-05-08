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
You are a strict review simulator.

CRITICAL RULES:
- The rating MUST be exactly {profile['rating_behavior']['avg_rating']} rounded to nearest integer OR derived from sentiment
- DO NOT contradict the rating in the review
- NEVER mention a different rating number inside the review text
- The review tone MUST match the rating

User Profile:
- Avg Rating: {profile['rating_behavior']['avg_rating']}
- Strictness: {profile['rating_behavior']['strictness']}

Business:
- Name: {biz['name']}
- Categories: {biz['categories']}

TASK:
1. Generate rating (1–5)
2. Generate review consistent with rating

Return ONLY JSON:
{{
  "rating": <int>,
  "review": "<text>"
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