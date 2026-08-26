"""
Search service - performs semantic search over indexed document chunks.
Falls back to local cosine-similarity search, then text search.
"""

import logging
from typing import List, Dict, Any

import numpy as np

from database.mongo import get_db
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SearchService:
    """Service for performing semantic search over document embeddings."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.db = get_db()

    def search(self, query: str, user_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic search for the query across the user's documents.
        Tries Atlas vector search first, then local cosine-similarity fallback,
        then keyword text search as a last resort.

        Args:
            query: The search query string.
            user_id: The Telegram user ID to scope the search.
            top_k: Number of top results to return.

        Returns:
            A list of matching document chunks with scores.
        """
        # Try Atlas vector search first
        results = self._vector_search(query, user_id, top_k)
        if results:
            return results

        # Fall back to local cosine-similarity search (works without Atlas)
        logger.info("Atlas vector search unavailable, trying local cosine-similarity search.")
        results = self._local_vector_search(query, user_id, top_k)
        if results:
            return results

        # Last resort: keyword text search
        logger.info("Local vector search returned no results, falling back to text search.")
        return self._text_search(query, user_id, top_k)

    def _vector_search(self, query: str, user_id: int, top_k: int) -> List[Dict[str, Any]]:
        """Perform a vector (semantic) search via MongoDB Atlas $vectorSearch."""
        try:
            query_embedding = self.embedding_service.embed_text(query)
        except Exception as e:
            logger.warning(f"Could not generate query embedding for Atlas search: {e}")
            return []

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "document_chunks_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": top_k,
                }
            },
            {
                "$match": {
                    "user_id": user_id,
                }
            },
            {
                "$project": {
                    "content": 1,
                    "source": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            results = list(self.db.document_chunks.aggregate(pipeline))
            logger.info(f"Vector search for '{query}' returned {len(results)} results.")
            return results
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def _local_vector_search(self, query: str, user_id: int, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform a local cosine-similarity search using stored embeddings.
        This works without MongoDB Atlas - fetches all chunks for the user,
        computes cosine similarity with numpy, and returns the best matches.
        """
        try:
            query_embedding = np.array(self.embedding_service.embed_text(query), dtype=np.float32)
        except Exception as e:
            logger.warning(f"Could not generate query embedding: {e}")
            return []

        # Fetch all chunks for this user that have embeddings
        try:
            chunks = list(
                self.db.document_chunks.find(
                    {"user_id": user_id, "embedding": {"$exists": True, "$ne": None}},
                    {"content": 1, "source": 1, "embedding": 1},
                )
            )
        except Exception as e:
            logger.warning(f"Could not fetch chunks for local search: {e}")
            return []

        if not chunks:
            return []

        results: List[Dict[str, Any]] = []
        norm_q = float(np.linalg.norm(query_embedding))
        if norm_q == 0:
            return []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if not embedding:
                continue
            try:
                chunk_vec = np.array(embedding, dtype=np.float32)
                # Cosine similarity
                dot = float(np.dot(query_embedding, chunk_vec))
                norm_c = float(np.linalg.norm(chunk_vec))
                if norm_q == 0 or norm_c == 0:
                    continue
                score = dot / (norm_q * norm_c)
                results.append({
                    "content": chunk.get("content", ""),
                    "source": chunk.get("source", "unknown"),
                    "score": score,
                })
            except Exception:
                continue

        # Sort by similarity score (descending) and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]
        logger.info(f"Local vector search for '{query}' returned {len(top_results)} results.")
        return top_results

    def _text_search(self, query: str, user_id: int, top_k: int) -> List[Dict[str, Any]]:
        """Perform a text-based (keyword) search as a fallback."""
        try:
            results = list(
                self.db.document_chunks.find(
                    {"user_id": user_id, "$text": {"$search": query}},
                    {"content": 1, "source": 1, "score": {"$meta": "textScore"}},
                ).sort([("score", {"$meta": "textScore"})]).limit(top_k)
            )
            logger.info(f"Text search for '{query}' returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during text search: {e}")
            return []

    def list_documents(self, user_id: int) -> List[Dict[str, Any]]:
        """
        List all unique documents for a user.

        Args:
            user_id: The Telegram user ID.

        Returns:
            A list of dicts with 'source', 'chunk_count', and 'total_chars'.
        """
        pipeline = [
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": "$source",
                    "chunk_count": {"$sum": 1},
                    "total_chars": {"$sum": {"$strLenCP": "$content"}},
                }
            },
            {"$sort": {"chunk_count": -1}},
        ]

        try:
            results = list(self.db.document_chunks.aggregate(pipeline))
            logger.info(f"Listed {len(results)} documents for user {user_id}.")
            return results
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
