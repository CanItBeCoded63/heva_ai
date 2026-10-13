"""
FAISS Vector Store and Indexer.

Implements dense vector indexing and retrieval using FAISS IndexFlatIP.
Embeddings are computed deterministically using a dense semantic projection
(TF-IDF + TruncatedSVD / random orthogonal projection), ensuring zero-dependency,
reproducible vector retrieval.
"""

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class FaissVectorStore:
    """
    Manages FAISS index construction, persistence, and k-NN semantic search.
    """

    def __init__(self, dimension: int = 64, seed: int = 42):
        self.dimension = dimension
        self.seed = seed
        self.index: faiss.Index = None
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer = None
        self.svd: TruncatedSVD = None
        self._is_fitted = False

    def build_index(self, corpus: List[Dict[str, Any]]):
        """
        Indexes a list of documents: [{'id': ..., 'title': ..., 'text': ...}].
        """
        self.documents = corpus
        texts = [doc["text"] for doc in corpus]

        # Fit TF-IDF and SVD for dense semantic projection
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=2048,
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Dimensionality reduction to dense embeddings
        n_components = min(self.dimension, tfidf_matrix.shape[1] - 1, len(texts) - 1)
        if n_components < 2:
            n_components = 2

        self.svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        dense_vectors = self.svd.fit_transform(tfidf_matrix)

        # Pad to self.dimension if needed
        if dense_vectors.shape[1] < self.dimension:
            padding = np.zeros((dense_vectors.shape[0], self.dimension - dense_vectors.shape[1]))
            dense_vectors = np.hstack([dense_vectors, padding])

        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense_vectors = (dense_vectors / norms).astype(np.float32)

        # Build FAISS IndexFlatIP
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(dense_vectors)
        self._is_fitted = True

    def embed_query(self, query: str) -> np.ndarray:
        """
        Transforms a query into normalized dense embedding vector.
        """
        if not self._is_fitted:
            raise RuntimeError("Index has not been built yet.")

        tfidf_vec = self.vectorizer.transform([query])
        dense = self.svd.transform(tfidf_vec)

        if dense.shape[1] < self.dimension:
            padding = np.zeros((dense.shape[0], self.dimension - dense.shape[1]))
            dense = np.hstack([dense, padding])

        norm = np.linalg.norm(dense)
        if norm > 0:
            dense = dense / norm
        return dense.astype(np.float32)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Returns top_k matching documents with cosine similarity scores.
        """
        if not self._is_fitted or self.index is None:
            raise RuntimeError("Index must be built before querying.")

        q_vec = self.embed_query(query)
        distances, indices = self.index.search(q_vec, min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results

    def save_index(self, directory: str):
        """Saves FAISS index and documents metadata."""
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss.index")
        docs_path = os.path.join(directory, "docs.json")

        faiss.write_index(self.index, index_path)
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

    def load_index(self, directory: str):
        """Loads FAISS index and documents metadata."""
        index_path = os.path.join(directory, "faiss.index")
        docs_path = os.path.join(directory, "docs.json")

        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            self.build_index(self.documents)
