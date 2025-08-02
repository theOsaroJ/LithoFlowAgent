import faiss
import pickle
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path: str, embed_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)
        self.index = faiss.read_index(index_path)
        with open(index_path + ".meta", "rb") as f:
            self.paths = pickle.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(emb, top_k)
        return [(self.paths[i], float(D[0][j])) for j, i in enumerate(I[0])]
