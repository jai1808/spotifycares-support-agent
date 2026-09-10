import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from src.embeddings import get_embedding, get_embeddings_batch, build_embedding_index, search_similar

def build_rag_corpus(conversations_df) -> List[Dict]:
    """
    Build RAG corpus from a pandas DataFrame of brand conversations.
    Assumes columns: 'user_tweet', 'brand_reply', 'thread_id'.
    """
    corpus = []
    for _, row in conversations_df.iterrows():
        corpus.append({
            "user_query": str(row.get('user_tweet', '')),
            "brand_response": str(row.get('brand_reply', '')),
            "thread_id": str(row.get('thread_id', ''))
        })
    return corpus

def save_rag_corpus(corpus: List[Dict], path: str) -> None:
    """Save the RAG corpus to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

def load_rag_corpus(path: str) -> List[Dict]:
    """Load the RAG corpus from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

class RAGRetriever:
    def __init__(self, corpus: Optional[List[Dict]] = None):
        """Initialize the RAG retriever with an optional corpus."""
        self.corpus = corpus or []
        self.embedding_matrix: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        
    def build_index(self) -> None:
        """Embed all user queries and build the numpy index."""
        if not self.corpus:
            raise ValueError("Corpus is empty. Cannot build index.")
        
        texts = [item['user_query'] for item in self.corpus]
        self.embedding_matrix, self.metadata = build_embedding_index(texts, self.corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top_k similar historical conversations for a given query."""
        if self.embedding_matrix is None or not self.metadata:
            raise ValueError("Index not built. Call build_index() or load() first.")
            
        query_embedding = get_embedding(query)
        return search_similar(query_embedding, self.embedding_matrix, self.metadata, top_k)

    def save(self, path: str) -> None:
        """Persist index and metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(f"{path}_matrix.npy", self.embedding_matrix)
        with open(f"{path}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
    @classmethod
    def load(cls, path: str) -> 'RAGRetriever':
        """Load index and metadata from disk."""
        instance = cls()
        instance.embedding_matrix = np.load(f"{path}_matrix.npy")
        with open(f"{path}_metadata.json", 'r', encoding='utf-8') as f:
            instance.metadata = json.load(f)
        instance.corpus = instance.metadata  # Corpus is essentially the metadata
        return instance
