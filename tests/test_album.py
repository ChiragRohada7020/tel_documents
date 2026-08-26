"""Regression tests for the multi-image album -> combined PDF flow."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import bot.commands as commands


class AlbumBufferTests(unittest.IsolatedAsyncioTestCase):
    def make_context(self):
        """A context with NO ``job_queue`` -- mirroring the polling setup."""
        context = SimpleNamespace(user_data={}, chat_data={})
        context.bot = MagicMock()
        context.bot.send_message = AsyncMock()
        context.bot.send_chat_action = AsyncMock()
        context.bot.send_document = AsyncMock(
            return_value=SimpleNamespace(document=SimpleNamespace(file_name="combined.pdf"))
        )
        return context

    def make_message(self, index=0, group_id="grp-1", caption=None):
        return SimpleNamespace(
            from_user=SimpleNamespace(id=42),
            chat_id=777,
            media_group_id=group_id,
            photo=[SimpleNamespace(file_id=f"file-{index}")],
            caption=caption,
        )

    async def cleanup_tasks(self, context):
        for task in list(context.user_data.get("_album_tasks", {}).values()):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_buffering_survives_missing_job_queue(self):
        """Regression: 'JobQueue is unavailable' crashed multi-photo uploads."""
        context = self.make_context()
        self.assertIsNone(getattr(context, "job_queue", None))

        await commands._buffer_album_photo(context, self.make_message(index=1))

        group = context.user_data["_album_groups"]["grp-1"]
        self.assertEqual(group["file_ids"], ["file-1"])
        self.assertIn("Collecting images", context.bot.send_message.await_args.kwargs["text"])
        context.bot.send_document.assert_not_awaited()
        await self.cleanup_tasks(context)

    async def test_second_photo_extends_group_without_losing_state(self):
        context = self.make_context()

        await commands._buffer_album_photo(context, self.make_message(index=1))
        first_task = context.user_data["_album_tasks"]["grp-1"]
        await commands._buffer_album_photo(
            context, self.make_message(index=2, caption="my Aadhaar card")
        )

        group = context.user_data["_album_groups"]["grp-1"]
        self.assertEqual(group["file_ids"], ["file-1", "file-2"])
        self.assertEqual(group["caption"], "my Aadhaar card")
        # The old timer was replaced, so it must have been cancelled.
        await asyncio.sleep(0)
        self.assertTrue(first_task.cancelled())
        await self.cleanup_tasks(context)


class AlbumFlushTests(unittest.IsolatedAsyncioTestCase):
    def make_context(self):
        context = SimpleNamespace(user_data={}, chat_data={})
        context.bot = MagicMock()
        context.bot.send_message = AsyncMock()
        context.bot.send_chat_action = AsyncMock()
        context.bot.send_document = AsyncMock(
            return_value=SimpleNamespace(document=SimpleNamespace(file_name="combined.pdf"))
        )
        return context

    def make_message(self, index=0, group_id="grp-9"):
        return SimpleNamespace(
            from_user=SimpleNamespace(id=42),
            chat_id=777,
            media_group_id=group_id,
            photo=[SimpleNamespace(file_id=f"file-{index}")],
            caption=None,
        )

    async def test_flush_combines_images_into_indexed_pdf(self):
        context = self.make_context()
        telegram = MagicMock()
        telegram.download_file = AsyncMock(side_effect=lambda _fid, path: open(path, "wb").close())

        service = MagicMock()
        service.process_document = AsyncMock(
            return_value={"success": True, "stored_as": "aadhaar.pdf", "chunks": 3}
        )
        composer = MagicMock()
        # The mocked composer must actually create the PDF on disk.
        composer.return_value.combine_to_pdf.side_effect = (
            lambda _paths, out_path: open(out_path, "wb").write(b"%PDF-1.4\n")
        )

        with patch("bot.commands._get_telegram_service", return_value=telegram), \
             patch("bot.commands._get_doc_service", return_value=service), \
             patch("bot.commands.ImageComposer", composer), \
             patch.object(commands, "_ALBUM_FLUSH_DELAY", 0.05):
            await commands._buffer_album_photo(context, self.make_message(index=1))
            await commands._buffer_album_photo(context, self.make_message(index=2))
            # Wait for the quiet-window timer to fire and flush the album.
            await asyncio.gather(*list(context.user_data["_album_tasks"].values()))

        context.bot.send_document.assert_awaited_once()
        kwargs = context.bot.send_document.await_args.kwargs
        self.assertEqual(kwargs["filename"], "combined.pdf")
        composer.return_value.combine_to_pdf.assert_called_once()
        self.assertEqual(len(composer.return_value.combine_to_pdf.call_args.args[0]), 2)
        service.process_document.assert_awaited_once()
        texts = [call.kwargs.get("text", "") for call in context.bot.send_message.await_args_list]
        self.assertTrue(any("Combined 2 images" in t for t in texts))


class StoredFileDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_delivery_works_when_no_message_object_exists(self):
        """Regression: 'NoneType' object has no attribute 'reply_text'."""
        telegram = MagicMock()
        telegram.send_document_by_file_id = AsyncMock()
        update = SimpleNamespace(effective_chat=SimpleNamespace(id=99), callback_query=None)

        with patch("bot.commands._get_telegram_service", return_value=telegram):
            await commands._send_stored_file(
                update,
                {"filename": "doc.pdf", "file_id": "tg-id", "file_type": "pdf"},
            )

        telegram.send_document_by_file_id.assert_awaited_once_with(99, "tg-id", caption="📄 doc.pdf")


if __name__ == "__main__":
    unittest.main()
