"""
Conversation service - manages per-user conversation history.
"""

import logging
from typing import List, Dict

from config import Config

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversation history per user."""

    def __init__(self) -> None:
        self.max_history = Config.MAX_CONVERSATION_HISTORY
        self._history: Dict[int, List[Dict[str, str]]] = {}

    def get_history(self, user_id: int) -> List[Dict[str, str]]:
        """Retrieve conversation history for a user."""
        return self._history.get(user_id, [])

    def add_message(self, user_id: int, role: str, content: str) -> None:
        """Add a message to the user's conversation history."""
        message = {"role": role, "content": content}
        self._history.setdefault(user_id, []).append(message)
        if len(self._history[user_id]) > self.max_history:
            self._history[user_id] = self._history[user_id][-self.max_history:]

    def clear_history(self, user_id: int) -> None:
        """Clear conversation history for a user."""
        if user_id in self._history:
            del self._history[user_id]
        logger.info(f"Cleared conversation history for user {user_id}")

    def get_formatted_history(self, user_id: int) -> str:
        """Return conversation history as a formatted string."""
        history = self.get_history(user_id)
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
