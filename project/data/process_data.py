import pandas as pd
import json

def load_sample(path, n=100000):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break

            try:
                data.append(json.loads(line))  # ✅ SAFE PARSING
            except json.JSONDecodeError:
                continue  # skip broken lines instead of crashing

    return pd.DataFrame(data)


# Load datasets (sampled to avoid memory crash)
reviews = load_sample("data/raw/review.json", 100000)
businesses = load_sample("data/raw/business.json", 50000)
users = load_sample("data/raw/user.json", 50000)

# Keep only needed columns (with safety checks)
reviews = reviews[["user_id", "business_id", "stars", "text"]]

businesses = businesses[["business_id", "name", "categories"]]

users = users[["user_id", "average_stars", "review_count"]]

# Save processed data
reviews.to_csv("data/processed/reviews.csv", index=False)
businesses.to_csv("data/processed/businesses.csv", index=False)
users.to_csv("data/processed/users.csv", index=False)

print("Done processing sample data ✔")