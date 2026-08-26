"""
Configuration module for the Telegram AI Assistant.
Loads environment variables and provides typed access to settings.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Centralized configuration loaded from environment variables."""

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    API_KEY: str = os.getenv("API_KEY", "")

    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "telegram_ai_assistant")

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    # Fast model used for intent classification (low latency, cheap)
    GROQ_CLASSIFIER_MODEL: str = os.getenv("GROQ_CLASSIFIER_MODEL", "openai/gpt-oss-20b")
    AI_MAX_TOKENS: int = int(os.getenv("AI_MAX_TOKENS", "4000"))
    AI_TEMPERATURE: float = float(os.getenv("AI_TEMPERATURE", "0.3"))

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
    # Disable on small cloud instances to avoid loading the Torch model; text
    # search, OCR, descriptions, and AI metadata remain available.
    ENABLE_LOCAL_EMBEDDINGS: bool = os.getenv("ENABLE_LOCAL_EMBEDDINGS", "true").lower() == "true"

    # Webhook / Polling
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    POLLING_MODE: bool = os.getenv("POLLING_MODE", "true").lower() == "true"

    # Google Cloud (for Vision API / OCR)
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    # OCR
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "en")
    ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "false").lower() == "true"

    # App environment
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # File upload limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
    DOCUMENT_PAGE_SIZE: int = int(os.getenv("DOCUMENT_PAGE_SIZE", "8"))
    ALLOW_GROUP_DOCUMENT_ACCESS: bool = os.getenv("ALLOW_GROUP_DOCUMENT_ACCESS", "false").lower() == "true"

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Conversation
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))

    # Expiry reminders (private-chat document vaults only)
    REMINDER_TIMEZONE: str = os.getenv("REMINDER_TIMEZONE", "Asia/Kolkata")
    REMINDER_HOUR: int = int(os.getenv("REMINDER_HOUR", "9"))
    EXPIRY_REMINDER_DAYS: tuple[int, ...] = tuple(
        int(value.strip()) for value in os.getenv("EXPIRY_REMINDER_DAYS", "30,7,1,0").split(",") if value.strip()
    )

    # Processing time limits: return a useful error instead of leaving uploads stuck.
    DOWNLOAD_TIMEOUT_SECONDS: int = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "120"))
    EMBEDDING_TIMEOUT_SECONDS: int = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180"))
    METADATA_TIMEOUT_SECONDS: int = int(os.getenv("METADATA_TIMEOUT_SECONDS", "60"))

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration values are set."""
        required = [
            ("TELEGRAM_BOT_TOKEN", cls.TELEGRAM_BOT_TOKEN),
            ("GROQ_API_KEY", cls.GROQ_API_KEY),
        ]
        missing = [name for name, value in required if not value]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        if cls.APP_ENV.lower() == "production" and not cls.API_KEY:
            raise ValueError("API_KEY must be set in production")
        if cls.MAX_FILE_SIZE_MB <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be greater than zero")
        if cls.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if not 0 <= cls.CHUNK_OVERLAP < cls.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be at least zero and smaller than CHUNK_SIZE")
        if cls.DOCUMENT_PAGE_SIZE < 1 or cls.DOCUMENT_PAGE_SIZE > 20:
            raise ValueError("DOCUMENT_PAGE_SIZE must be between 1 and 20")
        if cls.MAX_CONVERSATION_HISTORY < 1:
            raise ValueError("Conversation limits must be greater than zero")
        if not 0 <= cls.REMINDER_HOUR <= 23:
            raise ValueError("REMINDER_HOUR must be between 0 and 23")
        if any(days < 0 for days in cls.EXPIRY_REMINDER_DAYS):
            raise ValueError("EXPIRY_REMINDER_DAYS cannot contain negative values")
        if min(cls.DOWNLOAD_TIMEOUT_SECONDS, cls.EMBEDDING_TIMEOUT_SECONDS, cls.METADATA_TIMEOUT_SECONDS) < 1:
            raise ValueError("Processing timeout values must be greater than zero")

    @classmethod
    def setup_logging(cls) -> None:
        """Configure logging based on LOG_LEVEL."""
        level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        log_path = BASE_DIR / "bot.log"
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=level,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding="utf-8"),
            ],
        )
