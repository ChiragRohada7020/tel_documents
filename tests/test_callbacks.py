"""Regression tests for the inline document-action buttons."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class CallbackActionTests(unittest.IsolatedAsyncioTestCase):
    def make_update(self, callback_data: str):
        message = SimpleNamespace(reply_text=AsyncMock())
        query = SimpleNamespace(
            data=callback_data,
            message=message,
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )
        update = SimpleNamespace(
            callback_query=query,
            effective_user=SimpleNamespace(id=5091119380),
            effective_chat=SimpleNamespace(id=5091119380, type="private"),
            effective_message=message,
        )
        return update, query, message

    async def test_document_button_delivers_stored_file(self):
        from bot.commands import handle_callback_query

        update, query, _ = self.make_update("get_doc:abc")
        file_info = {"filename": "passport.jpg", "file_id": "telegram-id", "file_type": "photo"}
        service = MagicMock()
        service.get_file_by_id.return_value = file_info

        with patch("bot.commands._get_doc_service", return_value=service), \
             patch("bot.commands._send_stored_file", new=AsyncMock()) as send_file:
            await handle_callback_query(update, SimpleNamespace())

        query.answer.assert_awaited_once()
        send_file.assert_awaited_once_with(update, file_info)

    async def test_summary_button_returns_summary(self):
        from bot.commands import handle_callback_query

        update, _, message = self.make_update("summary:abc")
        service = MagicMock()
        service.get_file_by_id.return_value = {"filename": "policy.pdf", "ai_title": "Policy"}
        service.get_document_text.return_value = "policy text"
        groq = MagicMock()
        groq.summarize = AsyncMock(return_value="Short summary")

        with patch("bot.commands._get_doc_service", return_value=service), \
             patch("bot.commands._get_groq_service", return_value=groq):
            await handle_callback_query(update, SimpleNamespace())

        self.assertEqual(message.reply_text.await_count, 2)
        self.assertIn("Short summary", message.reply_text.await_args_list[-1].args[0])

    async def test_delete_button_requests_confirmation(self):
        from bot.commands import handle_callback_query

        update, _, _ = self.make_update("delete_doc:abc")
        service = MagicMock()
        service.get_file_by_id.return_value = {"filename": "policy.pdf", "ai_title": "Policy"}

        with patch("bot.commands._get_doc_service", return_value=service), \
             patch("bot.commands._safe_edit", new=AsyncMock()) as edit:
            await handle_callback_query(update, SimpleNamespace())

        self.assertIn("Delete 'Policy'", edit.await_args.args[1])

