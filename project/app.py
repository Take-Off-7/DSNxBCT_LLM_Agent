import os
import json
import requests
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from models.user_profile import build_user_profile
from models.review_generator import generate_review

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="LLM Agent Hackathon API",
    description="User Modeling + Review Generation API",
    version="1.0.0"
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

reviews_path = os.path.join(
    DATA_DIR,
    "reviews.csv"
)

businesses_path = os.path.join(
    DATA_DIR,
    "businesses.csv"
)

# -------------------------------------------------
# Lazy-loaded DataFrames
# -------------------------------------------------
reviews_df = None
businesses_df = None

# -------------------------------------------------
# Load datasets
# -------------------------------------------------
def load_data():

    global reviews_df
    global businesses_df

    if reviews_df is None:

        reviews_df = pd.read_csv(
            reviews_path
        )

    if businesses_df is None:

        businesses_df = pd.read_csv(
            businesses_path
        )

# -------------------------------------------------
# Startup Event
# -------------------------------------------------
@app.on_event("startup")
def startup_event():

    load_data()

# -------------------------------------------------
# User mapping helpers
# -------------------------------------------------
def get_user_maps():

    user_ids = reviews_df[
        "user_id"
    ].unique()

    user_name_map = {
        f"user_{i}": uid
        for i, uid in enumerate(user_ids)
    }

    reverse_map = {
        uid: name
        for name, uid in user_name_map.items()
    }

    return user_name_map, reverse_map

# -------------------------------------------------
# Business mapping helper
# -------------------------------------------------
def get_business_map():

    return {

        str(name).strip().lower(): bid

        for name, bid in zip(
            businesses_df["name"],
            businesses_df["business_id"]
        )
    }

# -------------------------------------------------
# Resolve functions
# -------------------------------------------------
def resolve_user_id(user_name):

    user_map, _ = get_user_maps()

    return user_map.get(user_name)

def resolve_business_id(business_name):

    if not business_name:
        return None

    business_map = get_business_map()

    return business_map.get(
        business_name.strip().lower()
    )

# -------------------------------------------------
# Request Models
# -------------------------------------------------
class ReviewRequest(BaseModel):

    user_id: str
    business_id: str

class ReviewRequestByName(BaseModel):

    user_name: str
    business_name: str

class ProfileRequest(BaseModel):

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
            "Behavior Simulation",
            "Ollama Integration"
        ],

        "docs": "/docs",

        "available_endpoints": [
            "/samples",
            "/demo-input",
            "/profile",
            "/review",
            "/review-by-name",
            "/generate"
        ]
    }

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# -------------------------------------------------
# SAMPLE USERS + BUSINESSES
# -------------------------------------------------
@app.get("/samples")
def samples():

    load_data()

    user_map, _ = get_user_maps()

    return {

        "sample_users":
        list(user_map.keys())[:10],

        "sample_businesses":
        businesses_df[
            ["name", "business_id"]
        ].head(10).to_dict(
            orient="records"
        )
    }

# -------------------------------------------------
# DEMO INPUT
# -------------------------------------------------
@app.get("/demo-input")
def demo_input():

    load_data()

    try:

        merged = reviews_df.merge(
            businesses_df,
            on="business_id"
        )

        sample = merged.sample(5)

        return [

            {
                "user_id": row["user_id"],
                "business_name": row["name"]
            }

            for _, row in sample.iterrows()
        ]

    except Exception as e:

        return {
            "error": str(e)
        }

# -------------------------------------------------
# PROFILE ENDPOINT
# -------------------------------------------------
@app.post("/profile")
def profile(req: ProfileRequest):

    try:

        profile_data = build_user_profile(
            req.user_id
        )

        return {

            "success": True,
            "profile": profile_data
        }

    except Exception as e:

        return {

            "success": False,
            "error": str(e)
        }

# -------------------------------------------------
# REVIEW ENDPOINT (IDs)
# -------------------------------------------------
@app.post("/review")
def review(req: ReviewRequest):

    try:

        result = generate_review(
            req.user_id,
            req.business_id
        )

        return result

    except Exception as e:

        return {

            "success": False,
            "error": str(e)
        }

# -------------------------------------------------
# REVIEW ENDPOINT (Names)
# -------------------------------------------------
@app.post("/review-by-name")
def review_by_name(req: ReviewRequestByName):

    try:

        user_id = resolve_user_id(
            req.user_name
        )

        if not user_id:

            return {

                "success": False,
                "error": "User not found",
                "hint": "Use /samples"
            }

        business_id = resolve_business_id(
            req.business_name
        )

        if not business_id:

            return {

                "success": False,
                "error": "Business not found",
                "hint": "Use /samples"
            }

        result = generate_review(
            user_id,
            business_id
        )

        return result

    except Exception as e:

        return {

            "success": False,
            "error": str(e)
        }

# -------------------------------------------------
# GENERATE (Hackathon Demo Endpoint)
# -------------------------------------------------
@app.post("/generate")
def generate(req: ReviewRequestByName):

    """
    Cleaner demo-friendly endpoint
    for judges and evaluators.
    """

    try:

        user_id = resolve_user_id(
            req.user_name
        )

        if not user_id:

            return {

                "success": False,
                "error": "Unknown user",
                "hint": "Use /samples"
            }

        business_id = resolve_business_id(
            req.business_name
        )

        if not business_id:

            return {

                "success": False,
                "error": "Unknown business",
                "hint": "Use /samples"
            }

        result = generate_review(
            user_id,
            business_id
        )

        return {

            "success": True,

            "input": {

                "user_name": req.user_name,
                "business_name": req.business_name
            },

            "output": result
        }

    except Exception as e:

        return {

            "success": False,
            "error": str(e)
        }

# -------------------------------------------------
# OLLAMA CHECK
# -------------------------------------------------
@app.get("/llm-status")
def llm_status():

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )

        if response.status_code == 200:

            models = response.json()

            return {

                "success": True,
                "ollama_running": True,
                "models": models
            }

        return {

            "success": False,
            "ollama_running": False
        }

    except Exception as e:

        return {

            "success": False,
            "ollama_running": False,
            "error": str(e)
        }