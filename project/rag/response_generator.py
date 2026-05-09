import requests
import os
from dotenv import load_dotenv

import sys
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


# -------------------------------------------------
# SAFE CONTEXT BUILDER
# -------------------------------------------------
def format_context(results):

    if not isinstance(results, list):
        return "No relevant results found."

    formatted = []

    for r in results:

        if not isinstance(r, dict):
            continue

        formatted.append(f"""
Business Name: {r.get('business_name', 'Unknown')}
Business ID: {r.get('business_id', 'N/A')}
Review: {r.get('text', 'N/A')}
Rating: {r.get('stars', 'N/A')}
Score: {r.get('score', 0)}
""")

    return "\n".join(formatted) if formatted else "No valid results found."


# -------------------------------------------------
# MAIN LLM GENERATOR (PERSONALIZED)
# -------------------------------------------------
def generate_response(query, results, user_id=None):

    context = format_context(results)

    # -------------------------------------------------
    # 🔥 LOAD USER PROFILE
    # -------------------------------------------------
    profile = {}
    if user_id:
        try:
            profile = build_user_profile(user_id)
        except Exception:
            profile = {}

    rating = profile.get("rating_behavior", {})
    traits = profile.get("behavioral_traits", {})
    style = profile.get("linguistic_style", {})
    sentiment = profile.get("sentiment_profile", {})

    # -------------------------------------------------
    # STYLE CONTROL
    # -------------------------------------------------
    verbosity = style.get("verbosity", "concise")
    formality = style.get("formality", 0.5)

    tone_instruction = "Keep response concise."
    if verbosity == "detailed":
        tone_instruction = "Provide a detailed explanation."

    if formality > 0.7:
        tone_instruction += " Use a formal tone."
    else:
        tone_instruction += " Use a casual, natural tone."

    # -------------------------------------------------
    # PERSONALIZATION INSTRUCTIONS
    # -------------------------------------------------
    personalization_rules = f"""
User Behavioral Traits:
- Price Sensitive: {traits.get('price_sensitive')}
- Positive Reviewer: {traits.get('positive_reviewer')}
- Critical Reviewer: {traits.get('critical_reviewer')}
- Detail Oriented: {traits.get('detail_oriented')}

User Preferences:
- Avg Rating Given: {rating.get('avg_rating')}
- Strictness: {rating.get('strictness')}
- Avg Sentiment: {sentiment.get('avg_sentiment')}

Guidelines:
- If price_sensitive → prioritize affordable, high-value options
- If positive_reviewer → emphasize good experiences
- If critical_reviewer → highlight both pros and cons
- If detail_oriented → include more explanation
"""

    # -------------------------------------------------
    # FINAL PROMPT
    # -------------------------------------------------
    prompt = f"""
You are a highly intelligent personalized recommendation assistant.

User Query:
{query}

{personalization_rules}

Top Retrieved Results:
{context}

STRICT RULES:
- ONLY recommend businesses from the provided results
- DO NOT introduce new businesses
- Base reasoning ONLY on the given reviews
- Do NOT hallucinate

STYLE:
{tone_instruction}

TASK:
Recommend the best options and explain WHY they fit this specific user.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            return {
                "success": False,
                "error": response.text,
                "fallback": "LLM unavailable, showing raw results"
            }

        data = response.json()

        return {
            "success": True,
            "response": data.get("response", ""),
            "model": OLLAMA_MODEL
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "fallback": "LLM request failed"
        }