import requests
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


# -------------------------------------------------
# SAFE CONTEXT BUILDER (CRASH-PROOF)
# -------------------------------------------------
def format_context(results):

    if not isinstance(results, list):
        return "No relevant results found."

    formatted = []

    for r in results:

        # 🔥 FIX: skip invalid items (strings, None, etc.)
        if not isinstance(r, dict):
            continue

        formatted.append(f"""
Business ID: {r.get('business_id', 'N/A')}
Review: {r.get('text', 'N/A')}
Rating: {r.get('stars', 'N/A')}
Score: {r.get('score', 0)}
""")

    return "\n".join(formatted) if formatted else "No valid results found."


# -------------------------------------------------
# MAIN LLM GENERATOR (SAFE)
# -------------------------------------------------
def generate_response(query, results, user_id=None):

    context = format_context(results)

    prompt = f"""
You are a recommendation assistant.

User query:
{query}

User ID:
{user_id}

Top results:
{context}

Instructions:
- Recommend the best options
- Explain WHY they match the user profile
- Keep response short, clear, and natural
- Do NOT hallucinate businesses not in context
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

        # 🔥 FIX: always return structured output
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