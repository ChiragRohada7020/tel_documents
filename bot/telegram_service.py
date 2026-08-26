"""
Telegram service - handles low-level Telegram API interactions.
"""

import asyncio
import logging
from telegram import Bot, InputFile
from telegram.request import HTTPXRequest

from config import Config

logger = logging.getLogger(__name__)


class TelegramService:
    """Wrapper around the Telegram Bot API for sending messages and files."""

    def __init__(self) -> None:
        # Use generous timeouts so slow networks don't break file downloads/uploads.
        # In PTB v22, timeouts are configured on the HTTPXRequest objects, not on Bot.
        request = HTTPXRequest(
            connection_pool_size=8,
            read_timeout=60,
            write_timeout=60,
            connect_timeout=30,
            pool_timeout=30,
            media_write_timeout=120,
        )
        self.bot: Bot = Bot(token=Config.TELEGRAM_BOT_TOKEN, request=request)

    async def send_message(self, chat_id: int, text: str, **kwargs) -> None:
        """Send a text message to a chat."""
        await self.bot.send_message(chat_id=chat_id, text=text, **kwargs)

    async def send_typing(self, chat_id: int) -> None:
        """Send a 'typing' action to indicate the bot is processing."""
        await self.bot.send_chat_action(chat_id=chat_id, action="typing")

    async def download_file(self, file_id: str, dest_path: str, retries: int = 2) -> str:
        """
        Download a file from Telegram by file_id and save to dest_path.
        Retries automatically on transient network failures.
        """
        last_error: Exception = RuntimeError("Download failed")
        for attempt in range(1, retries + 2):
            try:
                file = await self.bot.get_file(file_id)
                await file.download_to_drive(dest_path)
                logger.info(f"Downloaded file {file_id} to {dest_path}")
                return dest_path
            except Exception as e:
                last_error = e
                logger.warning(f"Download attempt {attempt}/{retries + 1} failed for {file_id}: {e}")
                if attempt <= retries:
                    await asyncio.sleep(1.5 * attempt)  # brief backoff before retrying
        raise last_error

    async def send_document(self, chat_id: int, file_path: str, caption: str = "") -> None:
        """Send a document to a chat."""
        with open(file_path, "rb") as doc:
            await self.bot.send_document(chat_id=chat_id, document=doc, caption=caption)

    async def send_document_by_file_id(self, chat_id: int, file_id: str, caption: str = "") -> None:
        """Send a document to a chat using a Telegram file_id (no local file needed)."""
        await self.bot.send_document(chat_id=chat_id, document=file_id, caption=caption)

    async def send_photo(self, chat_id: int, photo_path: str, caption: str = "") -> None:
        """Send a photo to a chat."""
        with open(photo_path, "rb") as photo:
            await self.bot.send_photo(chat_id=chat_id, photo=photo, caption=caption)

    async def send_photo_by_file_id(self, chat_id: int, file_id: str, caption: str = "") -> None:
        """Send a photo to a chat using a Telegram file_id (no local file needed)."""
        await self.bot.send_photo(chat_id=chat_id, photo=file_id, caption=caption)