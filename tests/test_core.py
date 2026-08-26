import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from processors.chunker import TextChunker


class ChunkerTests(unittest.TestCase):
    def test_zero_overlap_is_respected(self):
        chunks = TextChunker(chunk_size=5, chunk_overlap=0).chunk_by_chars("abcdefghij")
        self.assertEqual(chunks, ["abcde", "fghij"])

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=10, chunk_overlap=10)


class IntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_specific_document_request_is_not_misclassified_as_a_list(self):
        from bot.commands import _detect_intent_smart

        with patch("bot.commands._get_groq_service") as service:
            result = await _detect_intent_smart("send my insurance document", [])

        self.assertEqual(result, {"intent": "get_document", "query": "insurance"})
        service.assert_not_called()

    async def test_exact_browse_request_skips_classifier(self):
        from bot.commands import _detect_intent_smart

        with patch("bot.commands._get_groq_service") as service:
            result = await _detect_intent_smart("list my documents", [])

        self.assertEqual(result, {"intent": "list_documents", "query": ""})
        service.assert_not_called()


class ReminderTests(unittest.TestCase):
    def test_due_reminder_is_claimed_once(self):
        from services.reminder_service import ReminderService

        record = {
            "_id": "document-1",
            "user_id": 42,
            "filename": "policy.pdf",
            "expiry_date": "2026-09-02",
            "reminder_sent_days": [],
        }

        class Collection:
            def find(self, *_args, **_kwargs):
                return [record]

            def update_one(self, query, update):
                reminder_day = update["$addToSet"]["reminder_sent_days"]
                if reminder_day in record["reminder_sent_days"]:
                    return SimpleNamespace(modified_count=0)
                record["reminder_sent_days"].append(reminder_day)
                return SimpleNamespace(modified_count=1)

        database = SimpleNamespace(uploaded_files=Collection())
        with patch("services.reminder_service.get_db", return_value=database):
            service = ReminderService()
            first = service.claim_due_reminders(date(2026, 8, 26))
            second = service.claim_due_reminders(date(2026, 8, 26))

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["days_remaining"], 7)
        self.assertEqual(second, [])


class EmbeddingTests(unittest.TestCase):
    def test_model_is_loaded_lazily(self):
        from services.embedding_service import EmbeddingService

        model = MagicMock()
        model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2])
        with patch("services.embedding_service._load_model", return_value=model) as load_model:
            service = EmbeddingService()
            load_model.assert_not_called()
            self.assertEqual(service.embed_text("hello"), [0.1, 0.2])
            load_model.assert_called_once()


class ImageComposerTests(unittest.TestCase):
    def test_combine_images_to_pdf(self):
        from processors.image_composer import ImageComposer
        from PIL import Image
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(3):
                p = os.path.join(d, f"img_{i}.png")
                Image.new("RGB", (10, 10), (i * 60, 0, 0)).save(p)
                paths.append(p)
            out = os.path.join(d, "out.pdf")
            ImageComposer().combine_to_pdf(paths, out)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == "__main__":
    unittest.main()
