import os
import json
import requests
import pandas as pd
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from models.user_profile import build_user_profile
from models.review_generator import generate_review

from retrieval.retrieve import retrieve
from rag.response_generator import generate_response

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI(
    title="LLM Agent Hackathon API",
    description="User Modeling + Review Generation + RAG API",
    version="1.0.0"
)

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

reviews_path = os.path.join(DATA_DIR, "reviews.csv")
businesses_path = os.path.join(DATA_DIR, "businesses.csv")

reviews_df = None
businesses_df = None


def load_data():
    global reviews_df, businesses_df

    if reviews_df is None:
        reviews_df = pd.read_csv(reviews_path)

    if businesses_df is None:
        businesses_df = pd.read_csv(businesses_path)


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
# ROOT
# -------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "LLM Agent API Running ✔",
        "features": [
            "User Modeling",
            "Review Generation",
            "RAG Recommendation System"
        ]
    }

# -------------------------------------------------
# SAMPLES
# -------------------------------------------------
import random

@app.get("/samples")
def samples():
    try:
        if reviews_df is None or businesses_df is None:
            load_data()

        random_user = random.choice(reviews_df["user_id"].tolist())
        random_business = businesses_df.sample(1).iloc[0]

        return {
            "random_user_id": random_user,
            "random_business": {
                "name": random_business["name"],
                "business_id": random_business["business_id"]
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# -------------------------------------------------
# PROFILE
# -------------------------------------------------
@app.post("/profile")
def profile(req: ProfileRequest):
    try:
        return {
            "success": True,
            "profile": build_user_profile(req.user_id)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# -------------------------------------------------
# REVIEW
# -------------------------------------------------
@app.post("/review")
def review(req: ReviewRequest):
    try:
        return generate_review(req.user_id, req.business_id)
    except Exception as e:
        return {"success": False, "error": str(e)}

# -------------------------------------------------
# ✅ FIXED RECOMMEND ENDPOINT
# -------------------------------------------------
@app.post("/recommend")
def recommend(req: RecommendRequest):

    try:
        # STEP 1: Retrieve results (NO LLM here)
        retrieved = retrieve(
            query=req.query,
            user_id=req.user_id,
            use_llm=False   # 🔥 IMPORTANT FIX
        )

        # STEP 2: Generate response using ONLY results
        response = generate_response(
            req.query,
            retrieved["results"],   # 🔥 CRITICAL FIX
            req.user_id
        )

        return {
            "success": True,
            "query": req.query,
            "user_id": req.user_id,
            "retrieved": retrieved["results"],
            "response": response
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# -------------------------------------------------
# OLLAMA STATUS
# -------------------------------------------------
@app.get("/llm-status")
def llm_status():

    try:
        r = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )

        return {
            "ollama_running": r.status_code == 200,
            "models": r.json() if r.status_code == 200 else None
        }

    except Exception as e:
        return {
            "ollama_running": False,
            "error": str(e)
        }