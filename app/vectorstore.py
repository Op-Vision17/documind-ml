import faiss
import numpy as np
import pickle, os

class VectorStore:
    def __init__(self):
        self.dim = 384  # MiniLM dim
        self.index_file = "index.faiss"
        self.meta_file = "meta.pkl"

        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            self.metadata = pickle.load(open(self.meta_file, "rb"))
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []

    def add(self, vectors, meta_list):
        self.index.add(vectors.astype("float32"))
        self.metadata.extend(meta_list)
        self.save()

    def search(self, vector, k=4):
        D, I = self.index.search(np.array([vector]).astype("float32"), k)
        results = []
        for idx in I[0]:
            if idx == -1: continue
            results.append(self.metadata[idx])
        return results

    def save(self):
        faiss.write_index(self.index, self.index_file)
        pickle.dump(self.metadata, open(self.meta_file, "wb"))
