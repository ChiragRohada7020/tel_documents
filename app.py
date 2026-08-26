"""
Telegram AI Assistant - Main Application Entry Point
"""

import atexit
import logging
import os
import sys
from datetime import time
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import Application

from config import Config
from bot.handlers import setup_handlers
from database.mongo import init_db, close_db
from database.indexes import create_indexes
from services.reminder_service import run_expiry_reminders

# Configure logging
Config.setup_logging()
logger = logging.getLogger(__name__)

# PID lock file to prevent multiple bot instances (which cause 409 Conflict)
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bot.pid")


def _acquire_pid_lock() -> None:
    """Ensure only one bot instance is running (prevents 409 Conflict)."""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                logger.error(
                    f"Another bot instance is already running (PID {old_pid}). "
                    f"Remove {PID_FILE} if this is stale."
                )
                sys.exit(1)
            except (ProcessLookupError, PermissionError, OSError):
                logger.warning(f"Removing stale PID file (PID {old_pid} no longer running).")
                os.remove(PID_FILE)
        except (ValueError, IOError):
            logger.warning("Removing corrupt PID file.")
            try:
                os.remove(PID_FILE)
            except OSError:
                pass
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    logger.info(f"Acquired PID lock (PID {os.getpid()}).")


def _release_pid_lock() -> None:
    """Remove the PID lock file on shutdown."""
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
            logger.info("Released PID lock.")
    except Exception as e:
        logger.warning(f"Could not remove PID file: {e}")


async def post_init(application: Application) -> None:
    """Runs when the bot starts."""
    if application.job_queue:
        application.job_queue.run_daily(
            run_expiry_reminders,
            time=time(hour=Config.REMINDER_HOUR, tzinfo=ZoneInfo(Config.REMINDER_TIMEZONE)),
            name="expiry-reminders",
        )
        logger.info("Scheduled daily expiry reminders.")
    else:
        logger.warning("JobQueue is unavailable; expiry reminders are disabled.")
    logger.info("Bot started successfully.")


async def post_shutdown(application: Application) -> None:
    """Runs when the bot shuts down."""
    logger.info("Bot shutting down...")
    close_db()
    _release_pid_lock()


def main() -> None:
    """Start the bot."""
    # Prevent duplicate instances (causes 409 Conflict on Telegram getUpdates)
    _acquire_pid_lock()
    # Register immediately: configuration or database startup may fail below.
    atexit.register(_release_pid_lock)

    try:
        Config.validate()
        init_db(Config.MONGODB_URI, Config.MONGODB_DATABASE)
        create_indexes()

        application = (
            Application.builder()
            .token(Config.TELEGRAM_BOT_TOKEN)
            .post_init(post_init)
            .post_shutdown(post_shutdown)
            .build()
        )
        setup_handlers(application)

        if Config.POLLING_MODE:
            logger.info("Starting bot in polling mode...")
            # Telegram remembers the previous allowed-update selection. Request
            # every update type explicitly so inline-keyboard taps
            # (``callback_query``) are delivered alongside text messages.
            application.run_polling(allowed_updates=Update.ALL_TYPES)
        else:
            webhook_url = Config.WEBHOOK_URL
            if not webhook_url:
                raise ValueError("WEBHOOK_URL must be set when POLLING_MODE is false")
            logger.info(f"Starting bot with webhook: {webhook_url}")
            application.run_webhook(
                listen="0.0.0.0",
                # Render supplies the listening port through PORT. Its HTTPS
                # proxy handles TLS, so the app only needs an internal port.
                port=int(os.getenv("PORT", "10000")),
                url_path=Config.TELEGRAM_BOT_TOKEN,
                webhook_url=webhook_url,
                allowed_updates=Update.ALL_TYPES,
            )
    except Exception:
        close_db()
        _release_pid_lock()
        raise


if __name__ == "__main__":
    main()
