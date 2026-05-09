import os
import json
import requests
import pandas as pd
import sys
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.user_profile import build_user_profile
from models.review_generator import generate_review
from retrieval.retrieve import retrieve
from rag.response_generator import generate_response

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI(
    title="LLM Agent Hackathon API (Production Upgrade)",
    description="User Modeling + Review Generation + RAG System",
    version="2.0.0"
)

# -------------------------------------------------
# DATA CACHE
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

reviews_df = None
businesses_df = None


def load_data():
    global reviews_df, businesses_df

    if reviews_df is None:
        reviews_df = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))

    if businesses_df is None:
        businesses_df = pd.read_csv(os.path.join(DATA_DIR, "businesses.csv"))


@app.on_event("startup")
def startup():
    load_data()

# -------------------------------------------------
# REQUEST MODELS
# -------------------------------------------------
class ReviewRequest(BaseModel):
    user_id: str
    business_id: str


class ProfileRequest(BaseModel):
    user_id: str


class RecommendRequest(BaseModel):
    query: str
    user_id: str

# -------------------------------------------------
# SAFE RESPONSE WRAPPER
# -------------------------------------------------
def safe_response(success=True, data=None, error=None):
    return {
        "success": success,
        "data": data,
        "error": error
    }

# -------------------------------------------------
# HOME
# -------------------------------------------------
@app.get("/")
def home():
    return safe_response(data={
        "message": "LLM Agent API Running ✔",
        "features": [
            "Personalized User Modeling",
            "Review Simulation",
            "RAG Recommendation Engine"
        ]
    })

# -------------------------------------------------
# SAMPLES
# -------------------------------------------------
@app.get("/samples")
def samples():
    try:
        if reviews_df is None or businesses_df is None:
            load_data()

        import random

        random_user = random.choice(reviews_df["user_id"].tolist())
        random_business = businesses_df.sample(1).iloc[0]

        return safe_response(data={
            "random_user_id": random_user,
            "random_business": {
                "name": random_business["name"],
                "business_id": random_business["business_id"]
            }
        })

    except Exception as e:
        return safe_response(success=False, error=str(e))

# -------------------------------------------------
# PROFILE
# -------------------------------------------------
@app.post("/profile")
def profile(req: ProfileRequest):
    try:
        profile = build_user_profile(req.user_id)
        return safe_response(data=profile)
    except Exception as e:
        return safe_response(success=False, error=str(e))

# -------------------------------------------------
# REVIEW (SAFE + TIMEOUT PROTECTED)
# -------------------------------------------------
@app.post("/review")
def review(req: ReviewRequest):
    try:
        result = generate_review(req.user_id, req.business_id)

        return safe_response(data=result)

    except Exception as e:
        return safe_response(success=False, error=str(e))

# -------------------------------------------------
# RECOMMEND (FULLY STABILIZED PIPELINE)
# -------------------------------------------------
@app.post("/recommend")
def recommend(req: RecommendRequest):

    try:
        # STEP 1: retrieval (base candidates)
        retrieved = retrieve(
            query=req.query,
            user_id=req.user_id,
            use_llm=False
        )

        results = retrieved["results"]

        # STEP 2: load user profile for personalization
        profile = build_user_profile(req.user_id)

        avg_rating = profile.get("rating_behavior", {}).get("avg_rating", 3)
        strictness = profile.get("rating_behavior", {}).get("strictness", 0.5)
        sentiment = profile.get("sentiment_profile", {}).get("avg_sentiment", 0)

        # STEP 3: USER-AWARE RE-RANKING (🔥 FIX)
        for r in results:
            stars = r.get("stars", 3)

            # base score
            score = r.get("score", 0)

            # penalize mismatch with user taste
            rating_gap = abs(stars - avg_rating)

            penalty = (
                rating_gap * 0.15 +        # mismatch penalty
                strictness * 0.10          # strict users are harder to please
            )

            # boost good sentiment alignment
            if sentiment > 0:
                score += 0.05

            r["score"] = score - penalty

        # STEP 4: sort after rerank
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # STEP 5: LLM explanation (optional, safe fallback)
        try:
            response = generate_response(
                req.query,
                results,
                req.user_id
            )
        except Exception:
            response = {
                "success": False,
                "response": "Fallback mode: showing ranked results without AI explanation."
            }

        return safe_response(data={
            "query": req.query,
            "user_id": req.user_id,
            "recommendations": results,
            "ai_response": response
        })

    except Exception as e:
        return safe_response(
            success=False,
            error=str(e),
            data={"trace": traceback.format_exc()}
        )

# -------------------------------------------------
# LLM STATUS (HEALTH CHECK)
# -------------------------------------------------
@app.get("/llm-status")
def llm_status():

    try:
        r = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        if r.status_code != 200:
            return safe_response(data={
                "status": "unhealthy",
                "ollama_running": False,
                "reason": "Ollama API not responding"
            })

        models = r.json()

        # check if expected model exists
        model_list = []
        for m in models.get("models", []):
            model_list.append(m.get("name"))

        is_model_available = OLLAMA_MODEL in model_list

        return safe_response(data={
            "status": "healthy" if is_model_available else "degraded",
            "ollama_running": True,
            "model_loaded": is_model_available,
            "expected_model": OLLAMA_MODEL,
            "available_models": model_list
        })

    except requests.exceptions.Timeout:
        return safe_response(data={
            "status": "unhealthy",
            "ollama_running": False,
            "reason": "timeout connecting to Ollama"
        })

    except Exception as e:
        return safe_response(data={
            "status": "unhealthy",
            "ollama_running": False,
            "error": str(e)
        })