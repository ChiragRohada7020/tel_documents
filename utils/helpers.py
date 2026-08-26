"""
Utility helpers - common helper functions used across the application.
"""

import logging
import os
import re
import hashlib
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing unsafe characters.

    Args:
        filename: The original filename.

    Returns:
        A safe filename string.
    """
    # Remove path separators
    filename = os.path.basename(filename)
    # Replace unsafe characters
    filename = re.sub(r"[^\w\s\-\.]", "_", filename)
    # Collapse multiple underscores
    filename = re.sub(r"_+", "_", filename)
    return filename


def generate_file_hash(file_path: str) -> str:
    """
    Generate an MD5 hash of a file's contents.

    Args:
        file_path: Path to the file.

    Returns:
        The MD5 hash as a hex string.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
    except Exception as e:
        logger.error(f"Error generating file hash for {file_path}: {e}")
    return hash_md5.hexdigest()


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length, adding an ellipsis if truncated.

    Args:
        text: The input text.
        max_length: Maximum number of characters.

    Returns:
        The truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format a datetime as a human-readable string.

    Args:
        dt: The datetime to format. Defaults to now.

    Returns:
        A formatted timestamp string.
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: The directory path.

    Returns:
        The directory path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_file_size_mb(file_path: str) -> float:
    """
    Get the size of a file in megabytes.

    Args:
        file_path: Path to the file.

    Returns:
        File size in MB.
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0.0


def is_valid_file_type(filename: str, allowed_extensions: list) -> bool:
    """
    Check if a file has an allowed extension.

    Args:
        filename: The filename to check.
        allowed_extensions: List of allowed extensions (e.g., ['.pdf', '.docx']).

    Returns:
        True if the file type is allowed, False otherwise.
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() in [e.lower() for e in allowed_extensions]
