import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

def build_faiss_index(docs_dir: str, index_path: str, embed_model: str = "all-MiniLM-L6-v2"):
    """
    Reads all .md/.txt in docs_dir, embeds them, and builds + saves a FAISS index.
    """
    model = SentenceTransformer(embed_model)
    texts, paths = [], []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if fn.lower().endswith((".md", ".txt")):
                path = os.path.join(root, fn)
                with open(path, encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
                    paths.append(path)
    embeddings = model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(index_path + ".meta", "wb") as f:
        pickle.dump(paths, f)
    print(f"Built FAISS index at {index_path}")
