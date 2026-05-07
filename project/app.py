import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from models.review_generator import generate_review
from models.user_profile import build_user_profile

app = FastAPI(title="LLM Agent Hackathon API")

# -------------------------------------------------
# Load dataset (safe path)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

reviews_path = os.path.join(BASE_DIR, "data/processed/reviews.csv")
businesses_path = os.path.join(BASE_DIR, "data/processed/businesses.csv")

reviews_df = pd.read_csv(reviews_path)
businesses_df = pd.read_csv(businesses_path)

# -------------------------------------------------
# Lookup maps
# -------------------------------------------------

# user_id → fake readable name (for testing)
user_ids = reviews_df["user_id"].unique()
user_name_map = {f"user_{i}": uid for i, uid in enumerate(user_ids)}

# business name → business_id (case-insensitive safe fix)
business_name_to_id = {
    str(name).lower(): bid
    for name, bid in zip(businesses_df["name"], businesses_df["business_id"])
}

# reverse lookup (optional)
user_name_reverse = {v: k for k, v in user_name_map.items()}

# -------------------------------------------------
# Resolver functions (IMPROVED)
# -------------------------------------------------

def resolve_user_id(user_name: str):
    return user_name_map.get(user_name)


def resolve_business_id(business_name: str):
    if not business_name:
        return None

    return business_name_to_id.get(business_name.lower())

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
    return {"message": "LLM Agent API running ✔"}

# -------------------------------------------------
# 🔥 NEW: HELPERS FOR TESTING (IMPORTANT FOR JUDGES)
# -------------------------------------------------

@app.get("/samples")
def get_samples():
    return {
        "sample_user_ids": list(user_name_map.values())[:5],
        "sample_businesses": businesses_df[["name", "business_id"]].head(5).to_dict(orient="records")
    }


@app.get("/demo-input")
def demo_input():
    sample = reviews_df.merge(businesses_df, on="business_id").sample(5)

    return [
        {
            "user_id": row["user_id"],
            "business_name": row["name"]
        }
        for _, row in sample.iterrows()
    ]

# -------------------------------------------------
# TASK A (ID-based)
# -------------------------------------------------
@app.post("/review")
def review_endpoint(req: ReviewRequest):
    return generate_review(req.user_id, req.business_id)

# -------------------------------------------------
# TASK A (NAME-based)
# -------------------------------------------------
@app.post("/review-by-name")
def review_by_name(req: ReviewRequestByName):

    user_id = resolve_user_id(req.user_name)
    business_id = resolve_business_id(req.business_name)

    if not user_id:
        return {
            "error": "User not found",
            "hint": "Call /samples to get valid user_ids"
        }

    if not business_id:
        return {
            "error": "Business not found",
            "hint": "Call /samples to get valid business names"
        }

    return generate_review(user_id, business_id)

# -------------------------------------------------
# USER PROFILE
# -------------------------------------------------
@app.post("/profile")
def profile_endpoint(req: ProfileRequest):
    return build_user_profile(req.user_id)