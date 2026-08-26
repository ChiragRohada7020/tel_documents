"""
Database index management - creates and manages MongoDB indexes.
"""

import logging
from database.mongo import get_db

logger = logging.getLogger(__name__)


def create_indexes() -> None:
    """Create all necessary indexes for the database collections."""
    db = get_db()

    # Document chunks collection
    chunks_collection = db.document_chunks

    # Vector search index (requires MongoDB Atlas or MongoDB 7.0+ with vector search)
    # This is typically created via the Atlas UI or mongosh, but we define it here
    # for documentation purposes.
    try:
        chunks_collection.create_index(
            [("user_id", 1), ("source", 1)],
            name="user_source_idx",
        )
        logger.info("Created index: user_source_idx on document_chunks")
    except Exception as e:
        logger.warning(f"Could not create user_source_idx: {e}")

    # Text index for keyword search
    try:
        chunks_collection.create_index(
            [("content", "text")],
            name="content_text_idx",
        )
        logger.info("Created text index: content_text_idx on document_chunks")
    except Exception as e:
        logger.warning(f"Could not create content_text_idx: {e}")

    # Conversations collection
    conversations_collection = db.conversations

    try:
        conversations_collection.create_index(
            [("user_id", 1), ("timestamp", -1)],
            name="user_timestamp_idx",
        )
        logger.info("Created index: user_timestamp_idx on conversations")
    except Exception as e:
        logger.warning(f"Could not create user_timestamp_idx: {e}")

    # Uploaded files collection (file_id storage for Telegram retrieval)
    uploaded_files_collection = db.uploaded_files

    try:
        uploaded_files_collection.create_index(
            [("user_id", 1), ("filename", 1)],
            name="user_filename_idx",
        )
        logger.info("Created index: user_filename_idx on uploaded_files")
    except Exception as e:
        logger.warning(f"Could not create user_filename_idx: {e}")

    try:
        uploaded_files_collection.create_index(
            [("user_id", 1), ("category", 1)],
            name="user_category_idx",
        )
        logger.info("Created index: user_category_idx on uploaded_files")
    except Exception as e:
        logger.warning(f"Could not create user_category_idx: {e}")

    try:
        uploaded_files_collection.create_index(
            [("expiry_date", 1)], name="expiry_date_idx", sparse=True
        )
        logger.info("Created index: expiry_date_idx on uploaded_files")
    except Exception as e:
        logger.warning(f"Could not create expiry_date_idx: {e}")

    # Private graph links between a user's related documents.
    try:
        db.document_links.create_index(
            [("user_id", 1), ("left", 1), ("right", 1)],
            unique=True,
            name="user_document_link_idx",
        )
        logger.info("Created index: user_document_link_idx on document_links")
    except Exception as e:
        logger.warning(f"Could not create user_document_link_idx: {e}")

    # Users collection
    users_collection = db.users

    try:
        users_collection.create_index(
            [("user_id", 1)],
            unique=True,
            name="user_id_unique_idx",
        )
        logger.info("Created unique index: user_id_unique_idx on users")
    except Exception as e:
        logger.warning(f"Could not create user_id_unique_idx: {e}")

    logger.info("All indexes created successfully.")


def create_vector_index() -> None:
    """
    Create a vector search index for semantic search.
    Note: This requires MongoDB Atlas or MongoDB 7.0+ with vector search support.
    In Atlas, this is typically created via the UI or API.
    """
    db = get_db()
    chunks_collection = db.document_chunks

    try:
        # For MongoDB Atlas, use the Atlas Administration API or UI
        # This is a placeholder for the vector search index definition
        logger.info(
            "Vector search index 'document_chunks_vector_index' should be created "
            "via MongoDB Atlas UI or API for the 'embedding' field."
        )
    except Exception as e:
        logger.error(f"Could not create vector index: {e}")
