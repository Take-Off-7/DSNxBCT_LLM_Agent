import requests

import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


OLLAMA_URL = "http://localhost:11434/api/generate"


def format_context(results):
    """
    Convert FAISS + behavior results into LLM-readable context
    """
    return "\n".join([
        f"""
Business ID: {r['business_id']}
Review: {r['text']}
Rating: {r['stars']}
Score: {r['score']}
"""
        for r in results
    ])


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
- Explain WHY they match the user
- Keep response short and clear
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    return response.json()["response"]