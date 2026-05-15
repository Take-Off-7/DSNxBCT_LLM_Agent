import os
import json
import requests
import pandas as pd
import sys
import traceback

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# -------------------------------------------------
# BASE PATH (FIXED FOR ALL ENVIRONMENTS)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    title="TeamAI LLM Agent Hackathon API",
    description="User Modeling + Review Generation + Agentic Recommendation System",
    version="2.0.0"
)

# -------------------------------------------------
# DATA PATHS (ROBUST)
# -------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

reviews_df = None
businesses_df = None


def load_data():
    """
    Robust loader with clear fallback logging
    """
    global reviews_df, businesses_df

    try:
        reviews_path = os.path.join(DATA_DIR, "reviews.csv")
        business_path = os.path.join(DATA_DIR, "businesses.csv")

        print("🔍 Looking for data in:", DATA_DIR)

        if os.path.exists(reviews_path):
            reviews_df = pd.read_csv(reviews_path)
            print("✅ Loaded reviews.csv")

        else:
            print("⚠️ Missing reviews.csv at:", reviews_path)

        if os.path.exists(business_path):
            businesses_df = pd.read_csv(business_path)
            print("✅ Loaded businesses.csv")

        else:
            print("⚠️ Missing businesses.csv at:", business_path)

    except Exception as e:
        print("❌ DATA LOAD ERROR:", str(e))


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
# RESPONSE WRAPPER
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
        "architecture": "Agentic Retrieval + User Modeling + RAG",
        "version": "2.0.0"
    })

# -------------------------------------------------
# SAMPLES (FIXED DATA ISSUE)
# -------------------------------------------------
@app.get("/samples")
def samples():
    try:
        load_data()

        import random

        if reviews_df is None or businesses_df is None:
            return safe_response(
                success=False,
                error="Data not loaded. Check /data/processed folder."
            )

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
        return safe_response(data=build_user_profile(req.user_id))
    except Exception as e:
        return safe_response(success=False, error=str(e))

# -------------------------------------------------
# REVIEW
# -------------------------------------------------
@app.post("/review")
def review(req: ReviewRequest):
    try:
        result = generate_review(req.user_id, req.business_id)
        return safe_response(data=result)
    except Exception as e:
        return safe_response(success=False, error=str(e))

# -------------------------------------------------
# RECOMMEND (CLEAN AGENT PIPELINE)
# -------------------------------------------------
@app.post("/recommend")
def recommend(req: RecommendRequest):

    try:
        # STEP 1: retrieval (ONLY ranking layer)
        retrieved = retrieve(
            query=req.query,
            user_id=req.user_id,
            k=5,
            use_llm=False
        )

        results = retrieved.get("results", [])

        if not results:
            return safe_response(data={
                "query": req.query,
                "user_id": req.user_id,
                "recommendations": [],
                "ai_response": {
                    "success": False,
                    "response": "No results found"
                }
            })

        # STEP 2: user context (NOT scoring)
        try:
            profile = build_user_profile(req.user_id)
        except:
            profile = {}

        # STEP 3: LLM reasoning layer (explanation only)
        try:
            ai_response = generate_response(
                req.query,
                results,
                req.user_id
            )
        except Exception as e:
            ai_response = {
                "success": False,
                "response": "LLM fallback mode - ranked results only",
                "error": str(e)
            }

        return safe_response(data={
            "query": req.query,
            "user_id": req.user_id,
            # "recommendations": results,
            "ai_response": ai_response,
            "profile_loaded": bool(profile)
        })

    except Exception as e:
        return safe_response(
            success=False,
            error=str(e),
            data={"trace": traceback.format_exc()}
        )

# -------------------------------------------------
# LLM STATUS
# -------------------------------------------------
@app.get("/llm-status")
def llm_status():

    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)

        if r.status_code != 200:
            return safe_response(data={
                "status": "unhealthy",
                "ollama_running": False
            })

        models = r.json().get("models", [])
        model_list = [m.get("name") for m in models]

        is_ready = OLLAMA_MODEL in model_list

        return safe_response(data={
            "status": "healthy" if is_ready else "degraded",
            "ollama_running": True,
            "model_loaded": is_ready,
            "expected_model": OLLAMA_MODEL,
            "available_models": model_list
        })

    except Exception as e:
        return safe_response(data={
            "status": "unhealthy",
            "error": str(e)
        })