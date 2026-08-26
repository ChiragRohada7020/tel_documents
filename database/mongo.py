"""
MongoDB connection and database management.
"""

import logging
from pymongo import MongoClient
from pymongo.database import Database

logger = logging.getLogger(__name__)

_client: MongoClient = None
_db: Database = None


def init_db(mongo_uri: str, database_name: str) -> Database:
    """Initialize the MongoDB client and return the database instance."""
    global _client, _db
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception:
        client.close()
        raise
    _client = client
    _db = _client[database_name]
    logger.info(f"Connected to MongoDB database: {database_name}")
    return _db


def get_db() -> Database:
    """Get the current database instance."""
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db


def get_client() -> MongoClient:
    """Get the MongoDB client instance."""
    if _client is None:
        raise RuntimeError("MongoDB client not initialized. Call init_db() first.")
    return _client


def close_db() -> None:
    """Close the MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed.")
