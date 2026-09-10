"""Local TF-IDF embeddings for RAG (no cloud embedding API required)."""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EMBEDDING_MODEL, PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "embeddings_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_VECTORIZER: Optional[TfidfVectorizer] = None
_VECTORIZER_PATH = CACHE_DIR / "tfidf_vectorizer.pkl"


def _get_vectorizer() -> TfidfVectorizer:
    global _VECTORIZER
    if _VECTORIZER is not None:
        return _VECTORIZER
    if _VECTORIZER_PATH.exists():
        with open(_VECTORIZER_PATH, "rb") as f:
            _VECTORIZER = pickle.load(f)
        return _VECTORIZER
    _VECTORIZER = TfidfVectorizer(
        max_features=8192,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    return _VECTORIZER


def fit_vectorizer(texts: List[str]) -> TfidfVectorizer:
    """Fit (or refit) the shared TF-IDF vectorizer on a corpus and persist it."""
    global _VECTORIZER
    vectorizer = TfidfVectorizer(
        max_features=8192,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    vectorizer.fit(texts)
    _VECTORIZER = vectorizer
    with open(_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    return vectorizer


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Embed a single string with the fitted TF-IDF vectorizer."""
    vectorizer = _get_vectorizer()
    if not hasattr(vectorizer, "vocabulary_") or not vectorizer.vocabulary_:
        raise RuntimeError("TF-IDF vectorizer is not fitted. Build the RAG index first.")
    vec = vectorizer.transform([text]).toarray()[0]
    return vec.astype(float).tolist()


def get_embeddings_batch(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 100,
) -> List[List[float]]:
    """Embed a batch of strings."""
    del model, batch_size  # unused; kept for API compatibility
    vectorizer = _get_vectorizer()
    if not hasattr(vectorizer, "vocabulary_") or not vectorizer.vocabulary_:
        fit_vectorizer(texts)
        vectorizer = _get_vectorizer()
    matrix = vectorizer.transform(texts).toarray()
    return matrix.astype(float).tolist()


def build_embedding_index(
    texts: List[str], metadata: List[Dict]
) -> Tuple[np.ndarray, List[Dict]]:
    """Fit TF-IDF on corpus texts and return dense embedding matrix + metadata."""
    if len(texts) != len(metadata):
        raise ValueError("Texts and metadata must have the same length.")
    fit_vectorizer(texts)
    embeddings = get_embeddings_batch(texts)
    return np.array(embeddings, dtype=np.float32), metadata


def search_similar(
    query_embedding: List[float],
    embedding_matrix: np.ndarray,
    metadata: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """Search for similar embeddings using cosine similarity."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query, embedding_matrix)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_indices:
        result = metadata[int(idx)].copy()
        result["similarity_score"] = float(sims[int(idx)])
        results.append(result)
    return results
