import os
import sys
import json
import re
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile


# -------------------------------------------------
# DATA
# -------------------------------------------------
reviews_path = os.path.join(BASE_DIR, "data", "processed", "reviews.csv")
businesses_path = os.path.join(BASE_DIR, "data", "processed", "businesses.csv")

reviews = pd.read_csv(reviews_path)
businesses = pd.read_csv(businesses_path)


# -------------------------------------------------
# OLLAMA CONFIG
# -------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


# -------------------------------------------------
# JSON EXTRACTOR
# -------------------------------------------------
def extract_json(text):
    if not text:
        return None

    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group())
    except:
        return None


# -------------------------------------------------
# OLLAMA CALL
# -------------------------------------------------
def call_ollama(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.6
            }
        },
        timeout=45
    )

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json().get("response", "")


# -------------------------------------------------
# RATING LOGIC
# -------------------------------------------------
def compute_rating(profile):

    base = profile["rating_behavior"]["avg_rating"]
    strictness = profile["rating_behavior"].get("strictness", 0.5)

    rating = round(base)

    if strictness > 0.7:
        rating = max(1, rating - 1)

    if strictness < 0.3:
        rating = min(5, rating + 1)

    return int(rating)


# -------------------------------------------------
# GUARANTEED REVIEW FALLBACK (IMPORTANT FIX)
# -------------------------------------------------
def fallback_review(biz_name, rating):

    templates = {
        1: f"{biz_name} was a very disappointing experience. Would not recommend.",
        2: f"{biz_name} was below expectations and needs improvement.",
        3: f"{biz_name} was okay overall, nothing special but not bad.",
        4: f"{biz_name} was really good. I enjoyed the experience.",
        5: f"{biz_name} was excellent. Highly recommended!"
    }

    return templates.get(rating, templates[3])


# -------------------------------------------------
# REVIEW GENERATOR (FIXED)
# -------------------------------------------------
def generate_review(user_id, business_id):

    profile = build_user_profile(user_id)

    biz = businesses[businesses["business_id"] == business_id]

    if len(biz) == 0:
        return {
            "success": False,
            "error": "Business not found"
        }

    biz = biz.iloc[0]

    rating = compute_rating(profile)

    # -------------------------------------------------
    # STRONG PROMPT (NO EMPTY FIELD ALLOWED)
    # -------------------------------------------------
    prompt = f"""
You are a strict review generator.

Rules:
- rating MUST be {rating}
- review MUST be 2–3 sentences
- NEVER return empty review
- MUST match rating sentiment
- NO extra text

Business:
Name: {biz['name']}
Category: {biz['categories']}

Return ONLY valid JSON:

{{
  "rating": {rating},
  "review": "Write a natural human review here"
}}
"""

    try:

        raw = call_ollama(prompt)
        parsed = extract_json(raw)

        # -------------------------------------------------
        # RETRY ON FAILURE
        # -------------------------------------------------
        if not parsed or not parsed.get("review"):

            raw = call_ollama(prompt + " IMPORTANT: return full review text.")
            parsed = extract_json(raw)

        # -------------------------------------------------
        # FINAL SAFETY NET (CRITICAL FIX)
        # -------------------------------------------------
        review_text = parsed.get("review") if parsed else None

        if not review_text or review_text.strip() == "":
            review_text = fallback_review(biz["name"], rating)

        return {
            "success": True,
            "user_id": user_id,
            "business_id": business_id,
            "business_name": biz["name"],
            "rating": rating,
            "review": review_text,
            "model": OLLAMA_MODEL
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "fallback_review": fallback_review(biz["name"], rating)
        }


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":

    sample_user = reviews["user_id"].sample(1).values[0]
    sample_business = businesses["business_id"].sample(1).values[0]

    print(generate_review(sample_user, sample_business))