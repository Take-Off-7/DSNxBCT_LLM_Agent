import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed")


def load_reviews():
    path = os.path.join(DATA_PATH, "reviews.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def load_businesses():
    path = os.path.join(DATA_PATH, "businesses.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df