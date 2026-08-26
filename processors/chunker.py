"""
Text chunker - splits text into manageable chunks for embedding and indexing.
"""

import logging
import re
from typing import List

from config import Config

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into chunks of a specified size with optional overlap."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        self.chunk_size = Config.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = Config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentence boundaries.

        Args:
            text: The input text to chunk.

        Returns:
            A list of text chunks.
        """
        if not text.strip():
            return []

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence exceeds chunk size, save current chunk
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = self._get_overlap(chunks[-1]) + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        logger.info(f"Split text into {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def _get_overlap(self, text: str) -> str:
        """Get the last N characters of text for overlap."""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]

    def chunk_by_chars(self, text: str) -> List[str]:
        """
        Split text into fixed-size character chunks (no sentence awareness).

        Args:
            text: The input text to chunk.

        Returns:
            A list of text chunks.
        """
        if not text.strip():
            return []

        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                # Try to break at a sentence boundary
                last_period = text.rfind(".", start, end)
                if last_period > start:
                    end = last_period + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap if end < len(text) else end

        logger.info(f"Split text into {len(chunks)} character-based chunks")
        return chunks
