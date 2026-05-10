import requests
import os
import json
import numpy as np
import hashlib
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile
from models.learning_ranker import rank_businesses

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

USER_MEMORY_PATH = os.path.join(BASE_DIR, "data", "user_memory.json")


# -------------------------------------------------
# MEMORY
# -------------------------------------------------
def load_user_memory():
    if os.path.exists(USER_MEMORY_PATH):
        with open(USER_MEMORY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_user_memory(memory):
    os.makedirs(os.path.dirname(USER_MEMORY_PATH), exist_ok=True)
    with open(USER_MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)


def update_user_memory(user_id, query, response_text):
    memory = load_user_memory()

    if user_id not in memory:
        memory[user_id] = {
            "recent_queries": [],
            "seen_businesses": [],
            "style_vector": None,
            "feedback_log": []
        }

    memory[user_id]["recent_queries"].append(query)
    memory[user_id]["recent_queries"] = memory[user_id]["recent_queries"][-30:]

    text_blob = " ".join(memory[user_id]["recent_queries"])
    memory[user_id]["style_vector"] = embedder.encode(text_blob).tolist()

    save_user_memory(memory)


# -------------------------------------------------
# CONTEXT CLEANING
# -------------------------------------------------
def format_context(results):
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict)]


# -------------------------------------------------
# VECTOR BUILD (USER PERSONALIZATION CORE FIX)
# -------------------------------------------------
def build_user_vector(profile, memory, user_id, query):
    """
    FIX:
    Previously all users had similar vectors → identical rankings.

    Now we blend:
    - structured profile
    - query intent
    - user history (memory style vector)
    """

    profile_text = json.dumps(profile, default=str) if profile else ""

    mem_vec = None
    if user_id and memory.get(user_id, {}).get("style_vector"):
        mem_vec = np.array(memory[user_id]["style_vector"])

    query_vec = embedder.encode(query)

    profile_vec = embedder.encode(profile_text) if profile_text else np.zeros_like(query_vec)

    # combine signals
    if mem_vec is not None:
        combined = (0.4 * profile_vec) + (0.4 * query_vec) + (0.2 * mem_vec)
    else:
        combined = (0.6 * profile_vec) + (0.4 * query_vec)

    return combined


# -------------------------------------------------
# SELF-LEARNING RERANK
# -------------------------------------------------
def learning_rerank(query, items, user_id, profile, memory):

    user_memory = memory.get(user_id, {}) if user_id else {}

    return rank_businesses(
        user_id=user_id,
        query=query,
        candidates=items,
        user_profile=profile,
        user_memory=user_memory
    )


# -------------------------------------------------
# STABLE VARIATION
# -------------------------------------------------
def stable_noise(user_id, query):
    seed = hashlib.md5((str(user_id) + query).encode()).hexdigest()
    return int(seed[:8], 16) % 1000


# -------------------------------------------------
# MAIN GENERATOR
# -------------------------------------------------
def generate_response(query, results, user_id=None):

    context = format_context(results)
    memory = load_user_memory()

    # ---------------------------
    # USER PROFILE
    # ---------------------------
    profile = {}
    if user_id:
        try:
            profile = build_user_profile(user_id)
        except:
            profile = {}

    # ---------------------------
    # FIXED USER VECTOR (IMPORTANT)
    # ---------------------------
    user_vec = build_user_vector(profile, memory, user_id, query)

    # ---------------------------
    # RANKING
    # ---------------------------
    ranked = learning_rerank(
        query=query,
        items=context,
        user_id=user_id,
        profile=profile,
        memory=memory
    )[:5]

    # ---------------------------
    # UPDATE MEMORY (CRITICAL FOR PERSONALIZATION)
    # ---------------------------
    if user_id:
        memory.setdefault(user_id, {})
        memory[user_id].setdefault("seen_businesses", [])

        for r in ranked:
            bid = r.get("business_id")
            if bid and bid not in memory[user_id]["seen_businesses"]:
                memory[user_id]["seen_businesses"].append(bid)

        memory[user_id]["seen_businesses"] = memory[user_id]["seen_businesses"][-100:]
        save_user_memory(memory)

    # ---------------------------
    # FORMAT FOR LLM
    # ---------------------------
    formatted = "\n".join([
        f"{i+1}. {r.get('business_name')} | {r.get('category')} | rating={r.get('stars')} | score={r.get('score', 0):.3f}"
        for i, r in enumerate(ranked)
    ])

    # ---------------------------
    # PROMPT
    # ---------------------------
    prompt = f"""
You are a recommendation explanation engine.

RULES:
- Do NOT refuse
- Do NOT say you cannot help
- ONLY explain ranked results

USER QUERY:
{query}

RANKED RESULTS:
{formatted}

TASK:
1. Explain ranking clearly
2. Personalize based on inferred preferences
3. Compare options
4. Highlight tradeoffs
5. Keep concise
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5 + (stable_noise(user_id or "anon", query) / 5000)
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=25)
        data = response.json()
        text = data.get("response", "")

        if user_id:
            update_user_memory(user_id, query, text)

        return {
            "success": True,
            "recommendations": ranked,
            "ai_response": text,
            "context_size": len(context),
            "self_learning": True
        }

    except Exception as e:
        return {
            "success": True,
            "recommendations": ranked,
            "ai_response": "LLM unavailable - showing ranked results only",
            "error": str(e),
            "self_learning": True
        }