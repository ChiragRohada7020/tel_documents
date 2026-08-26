"""Memory service - manages in-process short-term and preference memory."""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryService:
    """Service for storing and retrieving user interaction memory."""

    def __init__(self) -> None:
        self._in_memory: Dict[str, List[str]] = {}

    def save_interaction(self, user_id: int, user_message: str, ai_response: str) -> None:
        """Save a user-AI interaction to short-term memory."""
        interaction = {
            "user": user_message,
            "assistant": ai_response,
            "timestamp": datetime.utcnow().isoformat(),
        }
        serialized = json.dumps(interaction)

        key = f"memory:{user_id}:interactions"
        self._in_memory.setdefault(key, []).insert(0, serialized)
        self._in_memory[key] = self._in_memory[key][:100]

    def get_recent_interactions(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent interactions from short-term memory."""
        key = f"memory:{user_id}:interactions"
        interactions = self._in_memory.get(key, [])[:limit]
        return [json.loads(item) for item in interactions]

    def save_preference(self, user_id: int, key: str, value: str) -> None:
        """Save a user preference until the bot process restarts."""
        pref_key = f"memory:{user_id}:preferences:{key}"
        self._in_memory[pref_key] = [value]

    def get_preference(self, user_id: int, key: str) -> Optional[str]:
        """Retrieve a user preference from long-term memory."""
        pref_key = f"memory:{user_id}:preferences:{key}"
        values = self._in_memory.get(pref_key, [])
        return values[0] if values else None

    def clear_memory(self, user_id: int) -> None:
        """Clear all memory for a user."""
        prefix = f"memory:{user_id}:"
        keys_to_remove = [k for k in self._in_memory if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._in_memory[key]
        logger.info(f"Cleared memory for user {user_id}")
