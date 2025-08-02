import os, pickle, faiss
from sentence_transformers import SentenceTransformer

def build_faiss_index(docs_dir, index_path, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts, paths = [], []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if fn.lower().endswith((".md", ".txt")):
                p = os.path.join(root, fn)
                with open(p, errors="ignore") as f:
                    texts.append(f.read()); paths.append(p)
    embs = model.encode(texts, convert_to_numpy=True)
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    faiss.write_index(idx, index_path)
    with open(index_path + ".meta", "wb") as f:
        pickle.dump(paths, f)
