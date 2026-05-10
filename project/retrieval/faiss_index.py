import os
import faiss
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "review_embeddings.npy")
META_PATH = os.path.join(BASE_DIR, "data", "embeddings", "reviews_with_ids.csv")

INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "faiss.index")
BUSINESS_PATH = os.path.join(BASE_DIR, "data", "processed", "businesses.csv")

# -------------------------------------------------
# NORMALIZATION (CRITICAL FOR COSINE)
# -------------------------------------------------
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors / norms


# -------------------------------------------------
# BUILD REVIEW INDEX (CORE RETRIEVAL LAYER)
# -------------------------------------------------
def build_review_index():

    embeddings = np.load(EMB_PATH).astype("float32")
    embeddings = normalize(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print(f"[OK] Review FAISS index built: {index.ntotal} vectors")


# -------------------------------------------------
# BUILD BUSINESS INDEX (🔥 INTELLIGENCE UPGRADE)
# -------------------------------------------------
def build_business_index():

    businesses = pd.read_csv(BUSINESS_PATH)

    businesses["business_id"] = businesses["business_id"].astype(str).str.strip()

    # combine semantic signals
    business_texts = (
        businesses["name"].fillna("") + " " +
        businesses["categories"].fillna("")
    ).values

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(business_texts, convert_to_numpy=True).astype("float32")
    embeddings = normalize(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    business_index_path = os.path.join(BASE_DIR, "data", "embeddings", "business_faiss.index")

    faiss.write_index(index, business_index_path)

    print(f"[OK] Business FAISS index built: {index.ntotal} vectors")


# -------------------------------------------------
# LOAD INDEX
# -------------------------------------------------
def load_index(path=INDEX_PATH):
    return faiss.read_index(path)


# -------------------------------------------------
# HYBRID BUILD (RECOMMENDED)
# -------------------------------------------------
def build_all():

    print("Building review index...")
    build_review_index()

    print("Building business index...")
    build_business_index()

    print("DONE ✔")


# -------------------------------------------------
# ENTRY
# -------------------------------------------------
if __name__ == "__main__":
    build_all()