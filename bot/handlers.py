"""
Telegram bot handlers - registers all command and message handlers.
"""

import logging
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

from bot.commands import (
    start,
    help_command,
    handle_message,
    handle_document,
    handle_photo,
    handle_callback_query,
    error_handler,
)

logger = logging.getLogger(__name__)


def setup_handlers(application: Application) -> None:
    """Register all handlers with the application."""
    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    application.add_handler(MessageHandler(filters.Document.DOCX, handle_document))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Callback query handler (for inline keyboard buttons)
    application.add_handler(CallbackQueryHandler(handle_callback_query))

    # Error handler
    application.add_error_handler(error_handler)

    logger.info("All handlers registered successfully, including document-action callbacks.")
