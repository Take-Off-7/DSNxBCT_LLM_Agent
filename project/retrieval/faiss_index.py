import os
import faiss
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "review_embeddings.npy")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "faiss.index")

def build_index():

    embeddings = np.load(EMB_PATH)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity (normalized)

    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print(f"FAISS index built with {index.ntotal} vectors ✔")


def load_index():

    return faiss.read_index(INDEX_PATH)


if __name__ == "__main__":
    build_index()