import faiss, pickle
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.paths = pickle.load(open(index_path + ".meta","rb"))
    def retrieve(self, q, top_k=5):
        emb = self.model.encode([q], convert_to_numpy=True)
        D,I = self.index.search(emb, top_k)
        return [(self.paths[i], float(D[0][j])) for j,i in enumerate(I[0])]
