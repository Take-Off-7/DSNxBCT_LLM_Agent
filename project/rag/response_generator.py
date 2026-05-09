import requests
import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

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
# EMBEDDING MODEL (USER VECTOR SPACE)
# -------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# SIMPLE USER MEMORY STORE (EVOLUTION LAYER)
# -------------------------------------------------
USER_MEMORY_PATH = os.path.join(BASE_DIR, "data", "user_memory.json")

def load_user_memory():
    if os.path.exists(USER_MEMORY_PATH):
        with open(USER_MEMORY_PATH, "r") as f:
            return json.load(f)
    return {}

def save_user_memory(memory):
    with open(USER_MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)

def update_user_memory(user_id, query, response_text):
    memory = load_user_memory()

    if user_id not in memory:
        memory[user_id] = {
            "recent_queries": [],
            "style_vector": None
        }

    memory[user_id]["recent_queries"].append(query)
    memory[user_id]["recent_queries"] = memory[user_id]["recent_queries"][-20:]

    # evolve user embedding over time
    text_blob = " ".join(memory[user_id]["recent_queries"])
    memory[user_id]["style_vector"] = embedder.encode(text_blob).tolist()

    save_user_memory(memory)


# -------------------------------------------------
# CONTEXT FORMATTER
# -------------------------------------------------
def format_context(results):
    if not isinstance(results, list):
        return "No relevant results found."

    formatted = []

    for r in results:
        if not isinstance(r, dict):
            continue

        formatted.append(f"""
Business: {r.get('business_name', 'Unknown')}
Review: {r.get('text', 'N/A')}
Rating: {r.get('stars', 'N/A')}
Score: {r.get('score', 0)}
""")

    return "\n".join(formatted)


# -------------------------------------------------
# USER VECTOR BUILDER (PERSONALIZATION CORE)
# -------------------------------------------------
def build_user_vector(profile):
    text = json.dumps(profile, default=str)
    return embedder.encode(text)


# -------------------------------------------------
# MAIN GENERATOR (NEXT-GEN PERSONALIZATION ENGINE)
# -------------------------------------------------
def generate_response(query, results, user_id=None):

    context = format_context(results)

    # -------------------------------------------------
    # PROFILE
    # -------------------------------------------------
    profile = {}
    user_memory = load_user_memory()

    if user_id:
        try:
            profile = build_user_profile(user_id)
        except:
            profile = {}

    traits = profile.get("behavioral_traits", {})
    rating = profile.get("rating_behavior", {})
    style = profile.get("linguistic_style", {})
    sentiment = profile.get("sentiment_profile", {})

    # -------------------------------------------------
    # USER MEMORY VECTOR
    # -------------------------------------------------
    user_vector = None
    if user_id and user_id in user_memory:
        user_vector = user_memory[user_id].get("style_vector")

    # -------------------------------------------------
    # SIMPLIFIED BUT STRONG DIVERSITY RULES
    # -------------------------------------------------
    diversity_instruction = """
DIVERSITY RULES:
- Avoid repeating similar businesses
- Ensure variety in recommendations (not all same category)
- Prefer different experiences (price, vibe, location)
"""

    # -------------------------------------------------
    # STRONGER PERSONALIZATION SIGNAL
    # -------------------------------------------------
    personalization = f"""
USER INTELLIGENCE PROFILE:

- Avg Rating Behavior: {rating.get('avg_rating')}
- Strictness Level: {rating.get('strictness')}
- Sentiment Bias: {sentiment.get('avg_sentiment')}

Behavior Traits:
- Price Sensitive: {traits.get('price_sensitive')}
- Critical Reviewer: {traits.get('critical_reviewer')}
- Detail Oriented: {traits.get('detail_oriented')}
- Positive Reviewer: {traits.get('positive_reviewer')}

Writing Style Preference:
- Formality: {style.get('formality')}
- Verbosity: {style.get('verbosity')}

NOTE:
User preferences evolve over time using memory signals.
"""

    # -------------------------------------------------
    # IMPORTANT FIX: FORCE MODEL TO USE ONLY DATA
    # -------------------------------------------------
    output_format = """
RESPONSE FORMAT (STRICT):

1. Ranked Recommendations (ONLY from context)
2. Why each matches user profile (short + precise)
3. Trade-offs (what user might dislike)
4. Final ranking justification

IMPORTANT:
- Do NOT invent new businesses
- Do NOT exaggerate quality
- Use ONLY provided context items
"""

    # -------------------------------------------------
    # STRONGER CONTEXT ENFORCEMENT
    # -------------------------------------------------
    prompt = f"""
You are a recommendation ranking assistant.

Your job is NOT just to describe — but to explain WHY these results are ranked this way.

{diversity_instruction}

{personalization}

USER QUERY:
{query}

CANDIDATE BUSINESSES (RANK THESE ONLY):
{context}

{output_format}

RULES:
- Use ONLY provided businesses
- Respect ranking order based on user fit
- Be honest about weaknesses
- No hallucinations
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
            timeout=25
        )

        if response.status_code != 200:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=30
            )

        data = response.json()
        result_text = data.get("response", "")

        # -------------------------------------------------
        # MEMORY UPDATE (IMPROVED SIGNAL QUALITY)
        # -------------------------------------------------
        if user_id:
            update_user_memory(user_id, query, result_text)

        return {
            "success": True,
            "response": result_text,
            "model": OLLAMA_MODEL,
            "user_vector_used": user_vector is not None,
            "context_size": len(results)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fallback": "LLM request failed"
        }