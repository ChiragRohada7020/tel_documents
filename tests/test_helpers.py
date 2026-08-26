"""Tests for message-parsing / formatting helpers behind new bot features."""
from bot.commands import (
    _detect_expiry_query,
    _format_facts,
    _parse_rename_request,
)


class TestParseRenameRequest:
    def test_target_and_new_title(self):
        assert _parse_rename_request("rename this to MSEB Aug Bill") == \
            ("this", "MSEB Aug Bill")
        assert _parse_rename_request("Please rename my rc card to Car RC Chirag") == \
            ("my rc card", "Car RC Chirag")

    def test_alternative_separators(self):
        assert _parse_rename_request("rename bill as MSEB August") == \
            ("bill", "MSEB August")
        assert _parse_rename_request("rename old scan -> Clean Name") == \
            ("old scan", "Clean Name")

    def test_no_target_uses_active_document(self):
        assert _parse_rename_request("naam badlo Electricity Bill") == \
            ("", "Electricity Bill")
        assert _parse_rename_request("rename electricity bill august") == \
            ("", "electricity bill august")

    def test_missing_pieces(self):
        assert _parse_rename_request("rename this to") == ("this", None)
        assert _parse_rename_request("rename") == ("", "")
        # reminder: '' target + None title must still be 'handled' (usage msg)

    def test_not_a_rename(self):
        assert _parse_rename_request("say hello") == (None, None)
        assert _parse_rename_request("") == (None, None)
        assert _parse_rename_request(None) == (None, None)

    def test_quotes_are_stripped(self):
        assert _parse_rename_request('rename this to "PAN Card"') == \
            ("this", "PAN Card")


class TestDetectExpiryQueryEdgeCases:
    def test_case_and_spacing_insensitive(self):
        assert _detect_expiry_query("WHAT EXPIRES THIS MONTH?") == "this_month"
        assert _detect_expiry_query("   expiry   ") == "all"

    def test_hinglish_expiry_words_not_supported_but_safe(self):
        # Unknown Hinglish phrasing falls through to the LLM intent path.
        assert _detect_expiry_query("mera document kab khatam hoga") is None

    def test_named_month_beats_generic_soon(self):
        assert _detect_expiry_query("expiring soon in september") == "month:09"


class TestFormatFacts:
    def test_none_when_no_facts(self):
        assert _format_facts("Doc", None) is None
        assert _format_facts("Doc", []) is None

    def test_sheet_contains_labels_and_values(self):
        out = _format_facts("MSEB Bill", [
            {"label": "Bill Number:", "value": "2026-08/77341"},
            {"label": "Amount Due", "value": "INR 2450"},
            {"label": "", "value": "Unlabeled value kept as Detail"},
            {"label": "Empty Value", "value": ""},
        ])
        assert "Key details — MSEB Bill" in out
        assert "• Bill Number: 2026-08/77341" in out
        assert "• Amount Due: INR 2450" in out
        # Missing labels get a generic placeholder instead of being dropped.
        assert "• Detail: Unlabeled value" in out
        assert "Empty Value" not in out   # empty values still skipped
        assert "Empty Value" not in out

    def test_caps_at_25_entries(self):
        facts = [{"label": f"F{i}", "value": str(i)} for i in range(40)]
        out = _format_facts("Doc", facts)
        assert "• F24: 24" in out and "F25" not in out