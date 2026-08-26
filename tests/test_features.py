"""Sprint regression tests: document rename + expiry listing (#1, #4)."""
from datetime import date
from types import SimpleNamespace

from bson import ObjectId

from services.document_service import DocumentService


# ---------------------------------------------------------------------------
# #4  date-aware expiry listing
# ---------------------------------------------------------------------------

class TestListExpiring:
    def _service(self, records):
        svc = object.__new__(DocumentService)

        class Files:
            def find(self, *_a, **_k):
                return iter(records)

        svc.db = SimpleNamespace(uploaded_files=Files())
        return svc

    def test_sorted_soonest_first_with_days(self):
        records = [
            {"_id": "b", "filename": "later.pdf", "expiry_date": "2026-12-31"},
            {"_id": "a", "filename": "soon.jpg", "ai_title": "Soon",
             "expiry_date": "2026-09-02"},
            {"_id": "c", "filename": "past.png", "expiry_date": "2026-08-01"},
        ]
        items = self._service(records).list_expiring(
            user_id=7, ref_date=date(2026, 8, 26))
        assert [i["days_remaining"] for i in items] == [-25, 7, 127]
        assert items[0]["filename"] == "past.png"
        assert items[1]["ai_title"] == "Soon"

    def test_invalid_dates_skipped(self):
        records = [
            {"_id": "x", "filename": "bad", "expiry_date": "coming-soon"},
            {"_id": "y", "filename": "good", "expiry_date": "2026-08-27"},
        ]
        items = self._service(records).list_expiring(user_id=7, ref_date=date(2026, 8, 26))
        assert [i["filename"] for i in items] == ["good"]

    def test_empty_vault_returns_empty(self):
        assert self._service([]).list_expiring(7) == []


class TestDetectExpiryQuery:
    def test_scopes(self):
        from bot.commands import _detect_expiry_query as d
        assert d("what expires this month") == "this_month"
        assert d("is anything overdue?") == "overdue"
        assert d("renewals next month") == "next_month"
        assert d("documents expiring in december") == "month:12"
        assert d("anything expiring soon") == "soon"
        assert d("what are my validity dates") == "all"

    def test_non_expiry_messages_ignored(self):
        from bot.commands import _detect_expiry_query as d
        assert d("hello there") is None
        assert d("send my pan card") is None
        assert d("give me all documents") is None


class TestScopeBounds:
    def test_this_month_matches_calendar_month(self):
        from bot.commands import _scope_bounds
        start, end, label = _scope_bounds("this_month", date(2026, 8, 26))
        assert start.day == 1 and (end.month, end.year) == (8, 2026)
        assert label == "August 2026"

    def test_named_month_rolls_forward_when_past(self):
        from bot.commands import _scope_bounds
        start, _, label = _scope_bounds("month:03", date(2026, 8, 26))
        assert start.year == 2027 and label == "March 2027"


# ---------------------------------------------------------------------------
# #1  rename_document keeps every reference consistent
# ---------------------------------------------------------------------------

class FakeFiles:
    """Handles ObjectId lookups AND filename-uniqueness checks."""

    def __init__(self, record=None, taken_names=()):
        self.record = record
        self.taken = set(taken_names)
        self.updates = []

    def find_one(self, query, projection=None):
        if "_id" in query:
            return dict(self.record) if self.record else None
        return {"_id": 1} if query.get("filename") in self.taken else None

    def update_one(self, query, update):
        self.updates.append((query, update))


class Recording:
    def __init__(self):
        self.calls = []

    def update_many(self, query, update):
        self.calls.append((query, update))


def rename_service(record, taken_names=()):
    svc = object.__new__(DocumentService)
    files = FakeFiles(record, taken_names)
    chunks, links = Recording(), Recording()
    svc.db = SimpleNamespace(uploaded_files=files,
                             document_chunks=chunks,
                             document_links=links)
    return svc, files, chunks, links


OID = "507f1f77bcf86cd799439011"


class TestRenameDocument:
    def _record(self):
        return {"_id": ObjectId(OID), "user_id": 7,
                "filename": "image-a1b2c3d4.jpg", "file_id": "tg-x",
                "file_type": "photo", "ai_title": ""}

    def test_renames_all_references_and_strips_hex(self):
        svc, files, chunks, links = rename_service(self._record())
        result = svc.rename_document(7, OID, "PAN Card Chirag")

        assert result["old"] == "image-a1b2c3d4.jpg"
        assert result["new"] == "pan-card-chirag.jpg"   # ext kept, hex gone
        assert result["title"] == "PAN Card Chirag"

        assert files.updates[0][1] == {"$set": {
            "filename": "pan-card-chirag.jpg",
            "ai_title": "PAN Card Chirag"}}
        assert chunks.calls[0] == (
            {"user_id": 7, "source": "image-a1b2c3d4.jpg"},
            {"$set": {"source": "pan-card-chirag.jpg"}})
        left_seen = any(c[0].get("left") == "image-a1b2c3d4.jpg" for c in links.calls)
        right_seen = any(c[0].get("right") == "image-a1b2c3d4.jpg" for c in links.calls)
        assert left_seen and right_seen

    def test_collision_gets_readability_counter_not_hex(self):
        svc, *_ = rename_service(self._record(), taken_names={"pan-card-chirag.jpg"})
        assert svc.rename_document(7, OID, "PAN Card Chirag")["new"] == \
            "pan-card-chirag-2.jpg"

    def test_missing_file_returns_none(self):
        svc, *_ = rename_service(None)
        assert svc.rename_document(7, OID, "X") is None

    def test_blank_title_rejected(self):
        svc, *_ = rename_service(self._record())
        assert svc.rename_document(7, OID, "   ") is None