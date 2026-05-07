import sys
import os
import pandas as pd

# 🔥 FORCE project root into Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.user_profile import build_user_profile

df = pd.read_csv("data/processed/reviews.csv")

user_id = df["user_id"].sample(1).values[0]

print("Testing user:", user_id)

profile = build_user_profile(user_id)

print(profile)