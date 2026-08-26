"""Tests for document title resolution and human-friendly stored names."""
from services.document_service import DocumentService


class _FakeFiles:
    def __init__(self, existing=None):
        self.existing = set(existing or ())

    def find_one(self, query, projection=None):
        if query.get("filename") in self.existing:
            return {"_id": 1}
        return None


def make_service(existing=None):
    """Build a DocumentService without Mongo/Telegram/network side effects."""
    svc = object.__new__(DocumentService)
    svc.db = type("FakeDB", (), {"uploaded_files": _FakeFiles(existing)})()
    return svc


class TestGenericTitleDetection:
    def test_detects_useless_titles(self):
        assert DocumentService._is_generic_title("") is True
        assert DocumentService._is_generic_title(None) is True
        assert DocumentService._is_generic_title("Image") is True
        assert DocumentService._is_generic_title("photo") is True
        assert DocumentService._is_generic_title("Document") is True

    def test_detects_camera_filenames(self):
        assert DocumentService._is_generic_title("IMG_20240512") is True
        assert DocumentService._is_generic_title("IMG_20240512.jpg") is True
        assert DocumentService._is_generic_title("whatsapp image") is True
        assert DocumentService._is_generic_title("DSC00321.png") is True

    def test_accepts_real_titles(self):
        assert DocumentService._is_generic_title("Electricity Bill - MSEDCL") is False
        assert DocumentService._is_generic_title("PAN Card Chirag") is False
        assert DocumentService._is_generic_title("Rent Receipt June") is False


class TestFallbackTitle:
    def test_caption_wins_over_ocr(self):
        out = DocumentService._fallback_title_from_ocr(
            "MSEB Bill August", caption="my light bill")
        assert out == "my light bill"

    def test_skips_label_and_junk_lines(self):
        ocr = (
            "OCR text from image:\n"
            "-----\n"
            "1234567890\n"
            "MSEB Electricity Bill August 2026 Total Due INR 2450"
        )
        out = DocumentService._fallback_title_from_ocr(ocr)
        assert out.startswith("MSEB Electricity Bill")
        assert "INR 2450" not in out  # only first 8 words kept

    def test_empty_when_nothing_usable(self):
        assert DocumentService._fallback_title_from_ocr("") == ""
        assert DocumentService._fallback_title_from_ocr("12345\n---\n***") == ""


class TestSlugAndUniqueness:
    def test_slug_is_clean_and_lowercase(self):
        assert DocumentService._title_slug("PAN Card — Chirag!") == "pan-card-chirag"
        assert DocumentService._title_slug("   ") == "document"
        assert DocumentService._title_slug("Rent Receipt — June") .startswith("rent-receipt")

    def test_no_suffix_when_free(self):
        svc = make_service()
        assert svc._unique_filename(7, "electricity-bill.jpg") == "electricity-bill.jpg"

    def test_appends_readability_counters_on_collision(self):
        svc = make_service({"electricity-bill.jpg", "electricity-bill-2.jpg"})
        assert svc._unique_filename(7, "electricity-bill.jpg") == "electricity-bill-3.jpg"

    def test_collisions_scoped_per_user(self):
        # Same filename held by a different user must NOT force a suffix.
        class _OtherUserFiles(_FakeFiles):
            def find_one(self, query, projection=None):
                return {"_id": 1} if query.get("user_id") != 7 else None
        svc = object.__new__(DocumentService)
        svc.db = type("DB", (), {"uploaded_files": _OtherUserFiles(["x.jpg"])})()
        assert svc._unique_filename(7, "bill.jpg") == "bill.jpg"


class TestResolveTitle:
    def test_specific_ai_title_preferred(self):
        svc = make_service()
        got = svc._resolve_title({"title": "LIC Premium Receipt"}, "some text")
        assert got == "LIC Premium Receipt"

    def test_generic_ai_title_rejected_for_content_name(self):
        svc = make_service()
        got = svc._resolve_title(
            {"title": "Image"},
            "MSEB Electricity Bill August 2026",
        )
        assert got.startswith("MSEB Electricity Bill")

    def test_falls_back_to_original_stem(self):
        svc = make_service()
        got = svc._resolve_title(None, "", original_name="insurance-policy.pdf")
        assert got == "insurance-policy"

    def test_last_resort_never_blank(self):
        svc = make_service()
        assert svc._resolve_title({"title": "photo"}, "") == "photo"