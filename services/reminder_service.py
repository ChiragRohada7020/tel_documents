"""Daily expiry reminders for documents with an explicit expiry date."""

import asyncio
import logging
from datetime import date
from typing import Any, Dict, List

from config import Config
from database.mongo import get_db

logger = logging.getLogger(__name__)


class ReminderService:
    """Find expiring files and ensure each configured reminder is sent once."""

    def __init__(self) -> None:
        self.db = get_db()

    def claim_due_reminders(self, today: date | None = None) -> List[Dict[str, Any]]:
        """Atomically claim reminders due today, preventing duplicate sends."""
        today = today or date.today()
        due: List[Dict[str, Any]] = []
        records = self.db.uploaded_files.find(
            {"expiry_date": {"$type": "string", "$ne": ""}},
            {"user_id": 1, "filename": 1, "ai_title": 1, "expiry_date": 1, "reminder_sent_days": 1},
        )
        for record in records:
            try:
                expiry = date.fromisoformat(record["expiry_date"])
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid expiry date on {record.get('_id')}: {record.get('expiry_date')!r}")
                continue
            days_remaining = (expiry - today).days
            if days_remaining not in Config.EXPIRY_REMINDER_DAYS:
                continue
            result = self.db.uploaded_files.update_one(
                {"_id": record["_id"], "reminder_sent_days": {"$ne": days_remaining}},
                {"$addToSet": {"reminder_sent_days": days_remaining}},
            )
            if result.modified_count:
                record["days_remaining"] = days_remaining
                due.append(record)
        return due

    def release_claim(self, record: Dict[str, Any]) -> None:
        """Allow retry tomorrow if Telegram delivery failed after a claim."""
        self.db.uploaded_files.update_one(
            {"_id": record["_id"]}, {"$pull": {"reminder_sent_days": record["days_remaining"]}}
        )


async def run_expiry_reminders(context) -> None:
    """JobQueue entry point: send today's reminders to each file owner."""
    try:
        service = ReminderService()
        due = await asyncio.to_thread(service.claim_due_reminders)
    except Exception as e:
        logger.error(f"Could not prepare expiry reminders: {e}")
        return

    for record in due:
        title = record.get("ai_title") or record["filename"]
        days = record["days_remaining"]
        if days == 0:
            timing = "expires today"
        elif days == 1:
            timing = "expires tomorrow"
        else:
            timing = f"expires in {days} days"
        try:
            await context.bot.send_message(
                chat_id=record["user_id"],
                text=f"⏰ Reminder: {title} {timing} ({record['expiry_date']}).",
            )
        except Exception as e:
            logger.warning(f"Could not send expiry reminder for {record.get('_id')}: {e}")
            await asyncio.to_thread(service.release_claim, record)
