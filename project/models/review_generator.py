import os
import sys
import json
import random
import pandas as pd
import requests
from dotenv import load_dotenv

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
# OLLAMA CONFIG (.env driven)
# -------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.8))


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_business_name(biz):
    return str(biz.get("name", "Unknown Business"))

def get_business_category(biz):
    return str(biz.get("categories") or biz.get("category") or "general")


# -------------------------------------------------
# CRITICAL FIX: SANITIZE USER PROFILE
# (PREVENTS MODEL COPYING RAW STRUCTURE)
# -------------------------------------------------
def simplify_profile(profile):
    rb = profile.get("rating_behavior", {})

    return {
        "avg_rating": float(rb.get("avg_rating", 3.5)),
        "strictness": float(rb.get("strictness", 0.5)),
        "variance": float(rb.get("variance", 0.7))
    }


# -------------------------------------------------
# JSON PARSER (ROBUST)
# -------------------------------------------------
def extract_json(text):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    try:
        return json.loads(text[start:end + 1])
    except:
        return None


# -------------------------------------------------
# OLLAMA CALL
# -------------------------------------------------
def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": 0.95
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=60)

    if r.status_code != 200:
        raise Exception(r.text)

    return r.json().get("response", "")


# -------------------------------------------------
# INTELLIGENT RATING MODEL
# -------------------------------------------------
def infer_rating(profile, category):
    rb = profile.get("rating_behavior", {})

    avg = float(rb.get("avg_rating", 3.6))
    strictness = float(rb.get("strictness", 0.5))
    variance = float(rb.get("variance", 0.7))

    bias_map = {
        "restaurant": 0.2,
        "food": 0.2,
        "hotel": 0.1,
        "service": -0.1,
        "tech": 0.0
    }

    bias = 0.0
    for k, v in bias_map.items():
        if k in category.lower():
            bias = v

    noise = random.uniform(-variance, variance)
    score = avg + bias + noise

    if strictness > 0.7:
        score -= random.uniform(0.2, 0.6)
    elif strictness < 0.3:
        score += random.uniform(0.1, 0.5)

    return max(1, min(5, round(score)))


# -------------------------------------------------
# CRITICAL FIX: STRONG PROMPT SEPARATION
# -------------------------------------------------
def build_prompt(profile, business, rating):

    safe_profile = simplify_profile(profile)

    return f"""
You are writing a REAL HUMAN REVIEW.

IMPORTANT:
- User profile is ONLY behavioral signal
- DO NOT output any profile fields
- DO NOT output JSON structures from input

USER BEHAVIOR SIGNALS:
{json.dumps(safe_profile)}

BUSINESS:
Name: {business["name"]}
Category: {business["categories"]}

TASK:
Write a believable 2–4 sentence review based on a real visit.

The tone MUST match rating: {rating}

ADDITIONAL STYLE REQUIREMENT (VERY IMPORTANT):
- You MUST include EXACTLY ONE sentence written in OBVIOUS Nigerian Pidgin English
- The Pidgin sentence must be clearly recognizable as Nigerian (e.g. "na so e be", "I no go lie", "the place dey ok but...")
- The Pidgin sentence should NOT be subtle or mixed English — it must be clearly Pidgin
- The remaining sentences must be standard English
- Do NOT repeat multiple Pidgin sentences — ONLY ONE

OUTPUT FORMAT (STRICT):
{{
  "review": "natural human review text only"
}}

RULES:
- NO metadata
- NO rating_behavior
- NO sentiment_profile
- NO explanations
- ONLY the review text
"""

# -------------------------------------------------
# FALLBACK (ONLY IF MODEL FAILS COMPLETELY)
# -------------------------------------------------
def fallback_review(name, rating):
    tones = {
        1: "terrible",
        2: "below average",
        3: "okay",
        4: "good",
        5: "excellent"
    }

    return (
        f"My experience at {name} was {tones.get(rating)}. "
        f"There were some noticeable aspects during the visit. "
        f"I would {'not ' if rating <= 2 else ''}visit again."
    )


# -------------------------------------------------
# OUTPUT VALIDATION (CRITICAL FIX)
# -------------------------------------------------
def is_valid_review(text):
    if not text:
        return False

    banned = [
        "rating_behavior",
        "sentiment_profile",
        "linguistic_style"
    ]

    for b in banned:
        if b in text:
            return False

    return len(text.strip()) > 15


# -------------------------------------------------
# MAIN GENERATOR
# -------------------------------------------------
def generate_review(user_id, business_id):

    used_fallback = False

    try:
        profile = build_user_profile(user_id)
    except:
        profile = {"rating_behavior": {}}

    biz_df = businesses[businesses["business_id"].astype(str) == str(business_id)]

    if biz_df.empty:
        return {"success": False, "error": "Business not found"}

    biz = biz_df.iloc[0]

    name = get_business_name(biz)
    category = get_business_category(biz)

    rating = infer_rating(profile, category)

    prompt = build_prompt(profile, biz, rating)

    review = None

    # retry loop (no premature fallback)
    for _ in range(3):
        raw = call_ollama(prompt)
        parsed = extract_json(raw)

        if parsed and is_valid_review(parsed.get("review")):
            review = parsed["review"]
            break

    # ONLY fallback if model fully fails
    if not review:
        used_fallback = True
        review = fallback_review(name, rating)

    return {
        "success": True,
        "user_id": user_id,
        "business_id": business_id,
        "business_name": name,
        # "category": category,
        "rating": rating,
        "review": review,
        "model": OLLAMA_MODEL,
        "temperature": OLLAMA_TEMPERATURE,
        "used_fallback": used_fallback
    }


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":

    sample_user = reviews["user_id"].sample(1).values[0]
    sample_business = businesses["business_id"].sample(1).values[0]

    result = generate_review(sample_user, sample_business)
    print(json.dumps(result, indent=2))