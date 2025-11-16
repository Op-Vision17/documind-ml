# ml-service/app/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class Embedder:
    def __init__(self, model_name: str = None):
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.model_name = model_name
        # load model once
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list) -> np.ndarray:
        """
        Embed a list of texts and return numpy array shape (n, dim)
        """
        if not texts:
            return np.asarray([])
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(embeddings)

    def embed_query(self, text: str):
        """
        Embed single query text and return numpy array (dim,)
        """
        emb = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(emb[0])
