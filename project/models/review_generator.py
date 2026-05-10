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

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile


# -------------------------------------------------
# DATA
# -------------------------------------------------
reviews_path = os.path.join(BASE_DIR, "data", "processed", "reviews.csv")
businesses_path = os.path.join(BASE_DIR, "data", "processed", "businesses.csv")

reviews = pd.read_csv(reviews_path)
businesses = pd.read_csv(businesses_path)

businesses.columns = [c.strip().lower() for c in businesses.columns]


# -------------------------------------------------
# OLLAMA CONFIG
# -------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


# -------------------------------------------------
# SAFE BUSINESS HELPERS
# -------------------------------------------------
def get_business_name(biz):
    return str(biz.get("name", "Unknown Business"))


def get_business_category(biz):
    return str(biz.get("categories") or biz.get("category") or "general")


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

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        },
        timeout=40
    )

    if r.status_code != 200:
        raise Exception(r.text)

    return r.json().get("response", "")


# -------------------------------------------------
# RATING ENGINE (STABLE + CONSISTENT)
# -------------------------------------------------
def compute_rating(profile):

    rb = profile.get("rating_behavior", {})

    base = float(rb.get("avg_rating", 3.5))
    strictness = float(rb.get("strictness", 0.5))

    rating = round(base)

    # behavioral adjustment
    if strictness > 0.7:
        rating -= 1
    elif strictness < 0.3:
        rating += 1

    return max(1, min(5, rating))


# -------------------------------------------------
# FALLBACK REVIEW (HIGH QUALITY, CONSISTENT)
# -------------------------------------------------
def fallback_review(name, rating):

    tone = {
        1: "very negative",
        2: "negative",
        3: "neutral",
        4: "positive",
        5: "very positive"
    }

    return (
        f"{name} was a {tone.get(rating, 'neutral')} experience overall. "
        f"The experience matched expectations for its category. "
        f"I would {'not ' if rating <= 2 else ''}recommend it based on my visit."
    )


# -------------------------------------------------
# MAIN REVIEW GENERATOR
# -------------------------------------------------
def generate_review(user_id, business_id):

    # profile (safe fallback)
    try:
        profile = build_user_profile(user_id)
    except:
        profile = {}

    biz_df = businesses[businesses["business_id"].astype(str) == str(business_id)]

    if biz_df.empty:
        return {
            "success": False,
            "error": "Business not found"
        }

    biz = biz_df.iloc[0]

    name = get_business_name(biz)
    category = get_business_category(biz)

    rating = compute_rating(profile)

    # -------------------------------------------------
    # IMPROVED PROMPT (TASK A OPTIMIZED)
    # -------------------------------------------------
    prompt = f"""
You are a professional review writer.

STRICT RULES:
- Rating MUST be {rating}
- Write EXACTLY 2–3 sentences
- Must match rating sentiment
- No headers, no explanations
- Return ONLY JSON

Business:
Name: {name}
Category: {category}

Output format:
{{
  "rating": {rating},
  "review": "natural human review"
}}
"""

    try:

        raw = call_ollama(prompt)
        parsed = extract_json(raw)

        # retry once if broken
        if not parsed or not parsed.get("review"):
            raw = call_ollama(prompt + "\nReturn ONLY valid JSON.")
            parsed = extract_json(raw)

        review = parsed.get("review") if parsed else None

        # final safety net
        if not review or not review.strip():
            review = fallback_review(name, rating)

        return {
            "success": True,
            "user_id": user_id,
            "business_id": business_id,
            "business_name": name,
            "rating": rating,
            "review": review,
            "model": OLLAMA_MODEL
        }

    except Exception as e:

        return {
            "success": True,
            "user_id": user_id,
            "business_id": business_id,
            "business_name": name,
            "rating": rating,
            "review": fallback_review(name, rating),
            "model": "fallback",
            "error": str(e)
        }


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":

    sample_user = reviews["user_id"].sample(1).values[0]
    sample_business = businesses["business_id"].sample(1).values[0]

    print(generate_review(sample_user, sample_business))