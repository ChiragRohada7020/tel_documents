"""
Embedding service - generates vector embeddings for text chunks.
"""

import logging
from functools import lru_cache
from typing import List, Any

import numpy as np
from config import Config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load each embedding model once per process."""
    # Importing sentence-transformers imports Torch, which exceeds small cloud
    # instances even if semantic search is disabled. Import only on demand.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""

    def __init__(self) -> None:
        # Loading may download a model on first use. Keep construction cheap so it
        # never blocks Telegram's update handler before processing begins.
        self.model = None
        self.dimension = Config.EMBEDDING_DIM

    def _get_model(self) -> Any:
        if self.model is None:
            self.model = _load_model(Config.EMBEDDING_MODEL)
        return self.model

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text string."""
        embedding = self._get_model().encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings."""
        embeddings = self._get_model().encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for chunks and return as a numpy array."""
        embeddings = self._get_model().encode(chunks, convert_to_numpy=True)
        return embeddings
