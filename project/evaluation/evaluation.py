import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from retrieval.retrieve import retrieve
from models.user_profile import build_user_profile

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
reviews_path = os.path.join(BASE_DIR, "data", "processed", "reviews.csv")
reviews = pd.read_csv(reviews_path)

# -------------------------------------------------
# NDCG@K
# -------------------------------------------------
def ndcg_at_k(recommended, relevant, k=10):

    dcg = 0.0

    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum([1 / np.log2(i + 2) for i in range(ideal_hits)])

    return dcg / idcg if idcg > 0 else 0.0


# -------------------------------------------------
# HIT RATE@K
# -------------------------------------------------
def hit_rate_at_k(recommended, relevant, k=10):
    return int(any(i in relevant for i in recommended[:k]))


# -------------------------------------------------
# RMSE (IMPROVED BASELINE)
# -------------------------------------------------
def compute_rmse():

    if "stars" not in reviews.columns:
        return None

    global_mean = reviews["stars"].mean()

    y_true = reviews["stars"].values
    y_pred = np.full_like(y_true, global_mean, dtype=float)

    return np.sqrt(mean_squared_error(y_true, y_pred))


# -------------------------------------------------
# EVALUATION CORE
# -------------------------------------------------
def evaluate_system(sample_users=50, k=10):

    users = reviews["user_id"].dropna().unique()
    users = np.random.choice(users, min(sample_users, len(users)), replace=False)

    ndcg_scores = []
    hit_scores = []

    for user_id in users:

        user_data = reviews[reviews["user_id"] == user_id]

        if len(user_data) == 0:
            continue

        # -------------------------------------------------
        # REALISTIC GROUND TRUTH
        # -------------------------------------------------
        relevant = set(user_data[user_data["stars"] >= 4]["business_id"])

        if len(relevant) == 0:
            continue

        # -------------------------------------------------
        # QUERY SIMULATION (agentic evaluation)
        # -------------------------------------------------
        try:
            query = "recommend places"  # neutral evaluation query

            result = retrieve(
                query=query,
                user_id=user_id,
                k=k
            )

            recommended = [
                r["business_id"]
                for r in result.get("results", [])
            ]

            ndcg_scores.append(ndcg_at_k(recommended, relevant, k))
            hit_scores.append(hit_rate_at_k(recommended, relevant, k))

        except Exception as e:
            print(f"⚠️ User {user_id} failed:", str(e))
            continue

    return {
        "NDCG@10": round(np.mean(ndcg_scores), 4) if ndcg_scores else 0,
        "HitRate@10": round(np.mean(hit_scores), 4) if hit_scores else 0,
        "RMSE": round(compute_rmse(), 4) if compute_rmse() else None
    }


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":

    print("🚀 Running improved evaluation...\n")

    results = evaluate_system(sample_users=50, k=10)

    print("\n📊 FINAL RESULTS:")
    for k, v in results.items():
        print(f"{k}: {v}")