"""
Telegram bot command and message handlers.
"""

import asyncio
import difflib
import logging
import os
import re
import tempfile
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from config import Config
from bot.telegram_service import TelegramService
from services.conversation_service import ConversationService
from services.groq_service import GroqService
from services.document_service import DocumentService
from services.search_service import SearchService
from services.memory_service import MemoryService
from processors.image_composer import ImageComposer

logger = logging.getLogger(__name__)

# Keywords that indicate the user wants to see/receive documents
_LIST_REQUESTS = {
    "list", "documents", "docs", "files", "my documents", "my docs",
    "list documents", "list my documents", "show all", "show all documents",
    "all documents", "what do you have", "kya kya documents hai",
}

# Keywords that indicate a search request
_SEARCH_KEYWORDS = ["search", "find", "look for", "looking for", "where is", "what is"]

# Common filler words stripped when extracting what document the user wants
_STOP_WORDS = {
    "give", "me", "my", "mine", "the", "a", "an", "send", "show", "please",
    "can", "could", "would", "you", "get", "find", "search", "for", "of",
    "to", "want", "need", "share", "all", "list", "have", "has", "do",
    "does", "what", "is", "are", "where", "it", "that", "this", "and",
    "or", "in", "on", "at", "with", "about", "documents", "document",
    "documts", "documtes", "docs", "doc", "file", "files", "pdf", "png",
    "jpg", "jpeg", "image", "photo", "picture", "pic",
}

_PUNCTUATION = """.,!?;:'"()"""

_NEWLINE = """
"""

_DOUBLE_NEWLINE = """


"""


# ---------------------------------------------------------------------------
# Cached service instances (created once, reused across all messages).
# This avoids reloading the embedding model / OCR check on every message.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_doc_service() -> DocumentService:
    return DocumentService()


@lru_cache(maxsize=1)
def _get_search_service() -> SearchService:
    return SearchService()


@lru_cache(maxsize=1)
def _get_groq_service() -> GroqService:
    return GroqService()


@lru_cache(maxsize=1)
def _get_conversation_service() -> ConversationService:
    return ConversationService()


@lru_cache(maxsize=1)
def _get_memory_service() -> MemoryService:
    return MemoryService()


@lru_cache(maxsize=1)
def _get_telegram_service() -> TelegramService:
    return TelegramService()


# High-signal words worth fuzzy-matching (catches typos like 'giev' or 'documtes')
_FUZZY_VOCAB = {
    "give", "send", "show", "list", "share",
    "documents", "document", "docs", "doc", "files", "file",
}

_DELETE_WORDS = {"delete", "remove", "erase", "discard", "hatao", "hatado"}
_SUMMARY_WORDS = {"summary", "summarize", "summarise", "explain"}


async def _detect_intent_smart(message: str, history: list) -> dict:
    """
    Smart intent detection combining three layers:

      1. Instant keyword fast-path (zero latency, high confidence)
      2. Fuzzy typo matching ('giev' -> 'give', 'documtes' -> 'documents')
      3. LLM classification fallback - understands Hinglish, natural
         questions and context-dependent follow-ups like 'send it'

    Returns {"intent": ..., "query": ...} where intent is one of:
      get_document | list_documents | search_knowledge | chat
    """
    msg = message.lower().strip()

    # Fast path 1: explicit list/browse requests
    if msg in _LIST_REQUESTS:
        return {"intent": "list_documents", "query": ""}

    # Fast path 2: fuzzy typo matching for document requests
    for word in msg.split():
        clean = word.strip(_PUNCTUATION)
        if len(clean) >= 4 and difflib.get_close_matches(clean, _FUZZY_VOCAB, n=1, cutoff=0.75):
            return {"intent": "get_document", "query": _extract_query(message)}

    # Smart path: LLM classification (Hinglish, follow-ups, natural language)
    result = await _get_groq_service().classify_intent(message, history)
    if result:
        return result

    # Final fallback: legacy keyword behavior
    if any(kw in msg for kw in _SEARCH_KEYWORDS):
        return {"intent": "search_knowledge", "query": _extract_query(message)}
    return {"intent": "chat", "query": ""}


def _extract_query(message: str) -> str:
    """Strip filler/keyword words to get the meaningful part of the request."""
    words = [w.strip(_PUNCTUATION) for w in message.lower().split()]
    filtered = [w for w in words if w and w not in _STOP_WORDS]
    return " ".join(filtered)


def _remember_document(context: ContextTypes.DEFAULT_TYPE, file_info: dict) -> None:
    """Remember the most recently discussed file for natural follow-ups."""
    if file_info.get("_id"):
        context.user_data["active_document_id"] = file_info["_id"]


def _active_document(user_id: int, context: ContextTypes.DEFAULT_TYPE):
    object_id = context.user_data.get("active_document_id")
    return _get_doc_service().get_file_by_id(user_id, object_id) if object_id else None


def _requested_document(user_id: int, context: ContextTypes.DEFAULT_TYPE, message: str):
    """Resolve a named document, or pronouns such as 'this' and 'it'."""
    query = _extract_query(message)
    if query:
        matches = _get_doc_service().find_documents(user_id, query, limit=1)
        if matches:
            return matches[0]
    return _active_document(user_id, context)


def _vault_access_allowed(update: Update) -> bool:
    """Keep a user's private documents and search results out of group chats."""
    chat = update.effective_chat
    return bool(chat and (chat.type == "private" or Config.ALLOW_GROUP_DOCUMENT_ACCESS))


async def _deny_group_vault_access(update: Update) -> None:
    await update.effective_message.reply_text(
        "For privacy, document uploads, searches, and file delivery are available only in a private chat with me."
    )


# ---------------------------------------------------------------------------
# Date-aware expiry queries ("what expires this month?")
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

_EXPIRY_KEYWORDS = (
    # Stem-based so expire/expires/expiring/expired/expiry all match.
    "expir", "renew",
    "valid till", "valid upto", "valid until", "validity", "due date",
    "overdue",
)


def _detect_expiry_query(message: str):
    """
    Return a scope when the message clearly asks about document expiry /
    renewal dates, otherwise None.

    Scopes: 'overdue' | 'this_month' | 'next_month' | 'soon' | 'all'
            | 'month:<MM>' for a named calendar month.
    """
    msg = (message or "").lower()
    if not any(keyword in msg for keyword in _EXPIRY_KEYWORDS):
        return None
    if "overdue" in msg or "already expired" in msg:
        return "overdue"
    if "next month" in msg:
        return "next_month"
    if "this month" in msg or "current month" in msg:
        return "this_month"
    for name, number in _MONTH_NAMES.items():
        if re.search(rf"\b{name}\b", msg):
            return f"month:{number:02d}"
    if any(word in msg for word in ("soon", "upcoming", "coming")):
        return "soon"
    return "all"


def _local_today() -> date:
    """Today's date in the timezone configured for reminders."""
    return datetime.now(ZoneInfo(Config.REMINDER_TIMEZONE)).date()


def _scope_bounds(scope: str, today: date):
    """Translate an expiry-query scope into (start_date, end_date, label)."""
    if scope == "overdue":
        return None, today, "already expired"
    if scope == "soon":
        return today, None, "expiring anytime"
    if scope == "this_month":
        start = today.replace(day=1)
        end = (start.replace(day=28) + timedelta(days=7)).replace(day=1) - timedelta(days=1)
        return start, end, start.strftime("%B %Y")
    if scope == "next_month":
        start = (today.replace(day=28) + timedelta(days=7)).replace(day=1)
        end = (start.replace(day=28) + timedelta(days=7)).replace(day=1) - timedelta(days=1)
        return start, end, start.strftime("%B %Y")
    if scope.startswith("month:"):
        number = int(scope.split(":", 1)[1])
        year = today.year
        start = today.replace(month=number, day=1, year=year)
        if number < today.month:  # earlier month this year rolls forward
            start = start.replace(year=year + 1)
        end = (start.replace(day=28) + timedelta(days=7)).replace(day=1) - timedelta(days=1)
        return start, end, start.strftime("%B %Y")
    return None, None, "with any recorded expiry date"


def _describe_days(days_remaining: int) -> str:
    if days_remaining == 0:
        return "⚠️ expires TODAY"
    if days_remaining < 0:
        return f"❌ expired {-days_remaining}d ago"
    return f"in {days_remaining}d"


async def _handle_expiry_query(update: Update, user_id: int, message: str) -> bool:
    """
    Answer questions like 'what expires this month?' straight from stored
    metadata. Returns True when the message was handled as an expiry query.
    """
    scope = _detect_expiry_query(message)
    if scope is None:
        return False

    today = _local_today()
    items = _get_doc_service().list_expiring(user_id, ref_date=today)

    start, end, label = _scope_bounds(scope, today)
    if start or end:
        filtered = []
        for item in items:
            parsed = DocumentService._parse_iso_date(item["expiry_date"])
            if parsed is None:
                continue
            if start and parsed < start:
                continue
            if end and parsed > end:
                continue
            filtered.append(item)
        items = filtered

    lines = [f"⏰ Documents {label}:", ""]
    shown = items[:15]
    if shown:
        for item in shown:
            name = item.get("ai_title") or item["filename"]
            lines.append(f"• {name} — {item['expiry_date']} ({_describe_days(item['days_remaining'])})")
        hidden = len(items) - len(shown)
        if hidden > 0:
            lines.append(f"…and {hidden} more.")
    else:
        lines.append("Nothing found.")
        lines.append("Upload a document whose OCR shows an explicit expiry /")
        lines.append("valid-till date and I will track it automatically.")

    if items and scope != "overdue":
        soonest = min((i["days_remaining"] for i in items), default=None)
        if soonest is not None and 0 <= soonest <= 30:
            lines.append("")
            lines.append("🔔 Daily reminders fire 30, 7, 1 days before — and on the day itself.")

    await update.message.reply_text(_NEWLINE.join(lines), reply_markup=_build_followup_keyboard())
    return True


# ---------------------------------------------------------------------------
# Structured fact sheets (📊 Key details button)
# ---------------------------------------------------------------------------

def _format_facts(header: str, facts) -> Optional[str]:
    """Render extracted facts as a simple readable sheet."""
    if not facts:
        return None
    lines = [f"📊 Key details — {header}:", ""]
    for fact in facts[:25]:
        label = str(fact.get("label", "")).strip().rstrip(":").rstrip() or "Detail"
        value = str(fact.get("value", "")).strip()
        if value:
            lines.append(f"• {label}: {value}")
    return _NEWLINE.join(lines)


# ---------------------------------------------------------------------------
# Natural-language rename ('rename this to ...')
# ---------------------------------------------------------------------------

_RENAME_PREFIX_RE = re.compile(
    r"^\s*(?:please\s+)?(?:rename|naam\s+badlo|naam\s+badal|change\s+the\s+name)\b[,:]?\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_RENAME_TO_SPLIT_RE = re.compile(r"\s+(?:as|=>|->)\s+|\s+to\s+|\s+to$", re.IGNORECASE)


def _parse_rename_request(message):
    """
    Parse 'rename [old-name] to [new-title]'.

    Returns (target_query, new_title) where either part may be '' meaning
    'unspecified'. Both parts None => the message is not a rename request.
    """
    match = _RENAME_PREFIX_RE.match(message or "")
    if not match:
        return None, None
    remainder = (match.group(1) or "").strip()
    if not remainder:
        return "", ""
    parts = _RENAME_TO_SPLIT_RE.split(remainder, maxsplit=1)
    if len(parts) == 2:
        target, new_title = parts[0].strip(), parts[1].strip(" \"'")
        return target, new_title or None
    return "", remainder.strip(" \"'")


async def _handle_rename_request(update: Update, user_id: int, message: str,
                                 context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Handle typed renames like:
      'rename this to Electricity Bill MSEB Aug'
      'rename pan card copy to PAN Card Chirag'
    Uses the currently-active document when no specific source is named.
    """
    target_query, new_title = _parse_rename_request(message)
    if new_title is None and target_query is None:
        return False
    if not new_title:
        await update.message.reply_text(
            "Tell me the new name too — e.g. ‘rename this to MSEB Aug bill’."
        )
        return True

    service = _get_doc_service()
    file_info = None
    if target_query:
        matches = service.find_documents(user_id, target_query, limit=1)
        file_info = matches[0] if matches else None
        if file_info is None:
            await update.message.reply_text(
                f"I could not find a document matching “{target_query}”. "
                "Open your 📋 list and tap the document first."
            )
            return True
    else:
        file_info = _active_document(user_id, context)
        if file_info is None:
            await update.message.reply_text(
                "Open the document first (tap it in your 📋 list), then say ‘rename this to …’."
            )
            return True

    _remember_document(context, file_info)
    result = await asyncio.to_thread(service.rename_document, user_id, file_info["_id"], new_title)
    if result:
        await update.message.reply_text(
            f"✏️ Renamed!\nWas: {result['old']}\nNow: {result['new']}",
            reply_markup=_build_followup_keyboard(),
        )
    else:
        await update.message.reply_text("⚠️ Renaming failed — please try again.")
    return True


async def _safe_reply(update: Update, text: str, reply_markup=None, parse_mode: str = None) -> None:
    """
    Reply with text; automatically falls back to plain text if Markdown
    parsing fails (dynamic AI titles/filenames often break Markdown).
    """
    try:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except BadRequest as e:
        logger.warning(f"Markdown send failed ({e}); retrying without formatting.")
        try:
            await update.message.reply_text(text, reply_markup=reply_markup)
        except Exception as e2:
            logger.error(f"Failed to send message: {e2}")
    except Exception as e:
        logger.error(f"Failed to send message: {e}")


async def _safe_edit(query, text: str, reply_markup=None, parse_mode: str = None) -> None:
    """
    Edit a callback message safely. Handles 'Message is not modified'
    (repeated button taps) and Markdown parse failures.
    """
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except BadRequest as e:
        if "message is not modified" in str(e).lower():
            return
        logger.warning(f"Markdown edit failed ({e}); retrying without formatting.")
        try:
            await query.edit_message_text(text, reply_markup=reply_markup)
        except BadRequest as e2:
            if "message is not modified" not in str(e2).lower():
                logger.error(f"Failed to edit message: {e2}")
        except Exception as e2:
            logger.error(f"Failed to edit message: {e2}")
    except Exception as e:
        logger.error(f"Failed to edit message: {e}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message with inline keyboard."""
    keyboard = [
        [InlineKeyboardButton("📚 Upload Document", callback_data="upload_doc")],
        [InlineKeyboardButton("📋 List My Documents", callback_data="list_docs")],
        [InlineKeyboardButton("🗂 Browse by Category", callback_data="categories")],
        [InlineKeyboardButton("🔍 Search Knowledge", callback_data="search")],
        [InlineKeyboardButton("💬 New Conversation", callback_data="new_chat")],
        [InlineKeyboardButton("❓ Help", callback_data="help")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        """👋 Welcome to your AI Assistant!

I can help you chat, search your documents, and process files.
Choose an option below or just start typing:""",
        reply_markup=reply_markup,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help information."""
    help_text = """🤖 Telegram AI Assistant

Available commands:
/start - Start the bot
/help - Show this help message

You can:
• Send text messages to chat with the AI
• Upload PDF, DOCX, or image files (stored safely in Telegram)
• Add a caption to images/documents to make them searchable
• Ask questions about your uploaded documents
• Say "give me my insurance document" to receive the actual file
• List all your uploaded documents with tappable buttons"""
    await update.message.reply_text(help_text)


def _build_followup_keyboard() -> InlineKeyboardMarkup:
    """Build an inline keyboard with follow-up action buttons."""
    keyboard = [
        [
            InlineKeyboardButton("❓ Ask Another Question", callback_data="ask_question"),
            InlineKeyboardButton("📚 Upload Document", callback_data="upload_doc"),
        ],
        [
            InlineKeyboardButton("📋 List My Documents", callback_data="list_docs"),
            InlineKeyboardButton("🗂 Categories", callback_data="categories"),
        ],
        [
            InlineKeyboardButton("🧹 End Conversation", callback_data="end_conversation"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _build_documents_keyboard(
    docs: list, page: int = 0, total: int = 0, category: str = ""
) -> InlineKeyboardMarkup:
    """Build a keyboard with one tappable button per stored document."""
    rows = []
    for doc in docs[:10]:
        icon = "🖼️" if doc.get("file_type") == "photo" else "📄"
        name = doc.get("ai_title") or doc["filename"]
        label = f"{icon} {name}"[:38]
        rows.append([
            InlineKeyboardButton(label, callback_data=f"get_doc:{doc['_id']}"),
            InlineKeyboardButton("📝 Summary", callback_data=f"summary:{doc['_id']}"),
        ])
        rows.append([
            InlineKeyboardButton("📊 Key details", callback_data=f"facts:{doc['_id']}"),
            InlineKeyboardButton("✏️ Rename", callback_data=f"rename_doc:{doc['_id']}"),
            InlineKeyboardButton("🗑 Delete", callback_data=f"delete_doc:{doc['_id']}"),
        ])
    if page > 0 or (page + 1) * Config.DOCUMENT_PAGE_SIZE < total:
        navigation = []
        if page > 0:
            callback = f"category_page:{category}:{page - 1}" if category else f"docs_page:{page - 1}"
            navigation.append(InlineKeyboardButton("◀ Previous", callback_data=callback))
        if (page + 1) * Config.DOCUMENT_PAGE_SIZE < total:
            callback = f"category_page:{category}:{page + 1}" if category else f"docs_page:{page + 1}"
            navigation.append(InlineKeyboardButton("Next ▶", callback_data=callback))
        rows.append(navigation)
    return InlineKeyboardMarkup(rows)


async def _show_categories(update: Update, user_id: int) -> None:
    """Show category filters for the user's uploaded documents."""
    categories = [
        category for category in _get_doc_service().list_categories(user_id)
        if len(category.encode("utf-8")) <= 45
    ]
    if not categories:
        await update.effective_message.reply_text("No categorized documents yet. Upload a document first.")
        return
    rows = [
        [InlineKeyboardButton(f"🗂 {category.title()}", callback_data=f"category:{category}")]
        for category in categories[:20]
    ]
    rows.append([InlineKeyboardButton("📋 All Documents", callback_data="list_docs")])
    await update.effective_message.reply_text("Choose a category:", reply_markup=InlineKeyboardMarkup(rows))


async def _reply_long(update: Update, text: str, reply_markup=None) -> None:
    """
    Send text to the chat, splitting it into multiple messages when it
    exceeds Telegram's 4096-character per-message limit.
    """
    max_len = 4000
    remaining = (text or "").strip() or "..."
    first = True
    while remaining:
        chunk = remaining[:max_len]
        if len(remaining) > max_len:
            # Prefer breaking at a newline inside the chunk
            cut = chunk.rfind(_NEWLINE)
            if cut > max_len // 2:
                chunk = remaining[:cut]
        try:
            await update.message.reply_text(chunk, reply_markup=reply_markup if first else None)
        except Exception as e:
            logger.error(f"Failed to send chunk: {e}")
        first = False
        remaining = remaining[len(chunk):].lstrip(_NEWLINE)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages with intelligent intent detection."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_message = update.message.text

    logger.info(f"Message from user {user_id}: {user_message}")

    # Show typing indicator while processing
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    if not _vault_access_allowed(update):
        await _chat_with_ai(update, user_id, user_message, include_document_context=False)
        return

    history = _get_conversation_service().get_history(user_id)
    words = set(re.findall(r"[a-z]+", user_message.lower()))

    # Conversational confirmations: "yes" after "delete this document".
    pending_delete = context.user_data.get("pending_delete_id")
    if pending_delete and user_message.lower().strip() in {"yes", "yes delete", "haan", "ha", "confirm"}:
        deleted = _get_doc_service().delete_file_by_id(user_id, pending_delete)
        context.user_data.pop("pending_delete_id", None)
        context.user_data.pop("active_document_id", None)
        await update.message.reply_text("✅ Document deleted." if deleted else "⚠️ That document was already deleted.")
        return

    # Let the user add facts to the document they most recently opened. This
    # makes a caption/editable note part of the same searchable knowledge.
    awaiting_note = context.user_data.get("awaiting_note_for")
    if awaiting_note:
        file_info = await asyncio.to_thread(
            _get_doc_service().append_user_note, user_id, awaiting_note, user_message
        )
        context.user_data.pop("awaiting_note_for", None)
        if file_info:
            await update.message.reply_text(
                f"✅ Added to {file_info.get('ai_title') or file_info['filename']}. I’ll use it in future searches and answers."
            )
        else:
            await update.message.reply_text("⚠️ I could not add that detail. Please open the document and try again.")
        return

    is_note_request = bool(re.search(r"\b(add|save|remember|update)\b.*\b(detail|details|note|info|information)\b", user_message.lower()))
    if is_note_request:
        file_info = _active_document(user_id, context)
        if not file_info:
            await update.message.reply_text("Open the document first, then tell me the detail you want to add.")
            return
        context.user_data["awaiting_note_for"] = file_info["_id"]
        await update.message.reply_text(
            f"What detail should I add to {file_info.get('ai_title') or file_info['filename']}? Send it in your next message."
        )
        return

    # Applying a rename started via the ✏️ Rename button: the next message
    # becomes the new human-readable title.
    awaiting_rename = context.user_data.get("awaiting_rename_for")
    if awaiting_rename:
        result = await asyncio.to_thread(
            _get_doc_service().rename_document, user_id, awaiting_rename, user_message
        )
        context.user_data.pop("awaiting_rename_for", None)
        if result:
            await update.message.reply_text(
                f"✏️ Renamed!\nNow: {result['new']}",
                reply_markup=_build_followup_keyboard(),
            )
        else:
            await update.message.reply_text("⚠️ I could not rename that document. Try again from the list.")
        return

    # Fast paths that never reach the LLM classifier:
    #   typed renames and date-aware expiry questions.
    if await _handle_rename_request(update, user_id, user_message, context):
        return
    if await _handle_expiry_query(update, user_id, user_message):
        return

    # Natural document actions work after viewing, receiving, or discussing a file.
    if words & _DELETE_WORDS:
        file_info = _requested_document(user_id, context, user_message)
        if not file_info:
            await update.message.reply_text("Which document should I delete? Send its name or open it from your document list first.")
            return
        _remember_document(context, file_info)
        context.user_data["pending_delete_id"] = file_info["_id"]
        name = file_info.get("ai_title") or file_info["filename"]
        await update.message.reply_text(f"Delete ‘{name}’? Reply YES to confirm.")
        return

    if words & _SUMMARY_WORDS and ("this" in words or "it" in words or "document" in words):
        file_info = _requested_document(user_id, context, user_message)
        if not file_info:
            await update.message.reply_text("Which document should I summarize? Send its name or open it from your document list first.")
            return
        _remember_document(context, file_info)
        text = _get_doc_service().get_document_text(user_id, file_info["filename"])
        if not text:
            await update.message.reply_text("I could not find readable text for that document.")
            return
        await update.message.reply_text("🧠 Preparing a concise summary…")
        summary = await _get_groq_service().summarize(text)
        await update.message.reply_text(summary or "I could not create a summary right now.")
        return

    # Smart intent detection: keywords -> fuzzy typos -> LLM (Hinglish aware)
    detected = await _detect_intent_smart(user_message, history)
    intent = detected["intent"]
    query = detected.get("query") or ""

    if intent == "list_documents":
        await _list_documents(update, user_id)
        return

    if intent == "get_document":
        if not query:
            query = _extract_query(user_message)
        await _handle_document_request(update, user_id, user_message, query=query, context=context)
        return

    if intent == "search_knowledge":
        # Prefer the LLM-extracted query; fall back to keyword stripping
        search_query = query or re.sub(
            r"^(search|find|look for|looking for|where is|what is)\s+",
            "",
            user_message.lower().strip(),
        )
        if not search_query:
            await update.message.reply_text(
                "🔍 What would you like to search for?",
                reply_markup=_build_followup_keyboard(),
            )
            return

        # Perform search
        results = await asyncio.to_thread(_get_search_service().search, search_query, user_id)

        if results:
            lines = [f'🔍 Search results for: "{search_query}":', ""]
            for i, doc in enumerate(results[:5], 1):
                content = doc.get("content", "")[:200]
                source = doc.get("source", "unknown")
                score = doc.get("score", 0)
                lines.append(f"{i}. 📄 {source}")
                lines.append(f"   {content}...")
                lines.append(f"   Match score: {score:.2f}")
                lines.append("")
            response = _NEWLINE.join(lines)
            await update.message.reply_text(response, reply_markup=_build_followup_keyboard())
        else:
            no_results = (
                f'🔍 No results found for "{search_query}".'
                + _NEWLINE
                + "Try uploading documents first, or ask me anything!"
            )
            await update.message.reply_text(no_results, reply_markup=_build_followup_keyboard())
        return

    # Default: chat with the AI
    await _chat_with_ai(update, user_id, user_message)


async def _handle_document_request(
    update: Update,
    user_id: int,
    user_message: str,
    query: str = "",
    context: ContextTypes.DEFAULT_TYPE = None,
) -> None:
    """
    Handle requests like 'give me my insurance document'.

    If exactly one document matches, send it directly. If multiple documents
    match (e.g. the same document uploaded several times), send the best
    match immediately and show ALL matches as tappable buttons so the user
    can receive any of them.

    Args:
        query: Pre-extracted query (e.g. from LLM intent detection).
               Falls back to keyword extraction from the message.
    """
    # Use provided query or extract meaningful keywords from the request
    if not query:
        query = _extract_query(user_message)
    matches = []
    if query:
        matches = _get_doc_service().find_documents(user_id, query)

    if not matches:
        # No specific match - list everything with tappable buttons
        await _list_documents(update, user_id)
        return

    if len(matches) == 1:
        if context:
            _remember_document(context, matches[0])
        await _send_stored_file(update, matches[0])
        return

    # Multiple matches: send the best one, then show all of them
    if context:
        _remember_document(context, matches[0])
    await _send_stored_file(update, matches[0])

    lines = [f'🔎 Found {len(matches)} documents matching "{query}":', ""]
    for i, m in enumerate(matches, 1):
        icon = "🖼️" if m.get("file_type") == "photo" else "📄"
        name = m.get("ai_title") or m["filename"]
        lines.append(f"{i}. {icon} {name}")
    listing = _NEWLINE.join(lines)

    await update.message.reply_text(
        listing,
        reply_markup=_build_documents_keyboard(matches),
    )


async def _send_stored_file(update: Update, file_info: dict) -> None:
    """Send a Telegram-hosted file back to the user using its file ID."""
    telegram_service = _get_telegram_service()
    chat_id = update.effective_chat.id
    filename = file_info["filename"]
    file_id = file_info["file_id"]
    display_name = file_info.get("ai_title") or filename
    # CallbackQuery updates do not populate ``update.message``. Their message
    # is exposed as ``effective_message`` / ``callback_query.message``.
    # The message object may be unavailable (e.g. the originating message was
    # edited/deleted), so guard before calling ``reply_text``.
    message = getattr(update, "effective_message", None)
    if message is None:
        query = getattr(update, "callback_query", None)
        message = getattr(query, "message", None)

    if message is not None:
        try:
            await message.reply_text(f"📤 Sending you: {display_name}")
        except Exception as e:
            logger.warning(f"Could not send 'sending' notice: {e}")
    try:
        if file_info.get("file_type") == "photo":
            await telegram_service.send_photo_by_file_id(chat_id, file_id, caption=f"📄 {display_name}")
        else:
            await telegram_service.send_document_by_file_id(chat_id, file_id, caption=f"📄 {display_name}")
    except Exception as e:
        logger.error(f"Telegram file-id send failed for {filename}: {e}")
        if message is not None:
            try:
                await message.reply_text(
                    f'⚠️ Telegram could not send "{display_name}". Please re-upload it.',
                    reply_markup=_build_followup_keyboard(),
                )
            except Exception as e2:
                logger.warning(f"Could not send error notice: {e2}")
        else:
            logger.warning("No message object available to send error notice.")


async def _chat_with_ai(
    update: Update, user_id: int, user_message: str, include_document_context: bool = True
) -> None:
    """Chat with the AI using conversation history and document context."""
    conversation_service = _get_conversation_service()
    history = conversation_service.get_history(user_id)
    if not history:
        # Short-term memory gives a small grace period if persistent history is unavailable.
        for interaction in reversed(_get_memory_service().get_recent_interactions(user_id)):
            history.extend([
                {"role": "user", "content": interaction["user"]},
                {"role": "assistant", "content": interaction["assistant"]},
            ])

    # Get relevant context from memory/search
    relevant_docs = (
        await asyncio.to_thread(_get_search_service().search, user_message, user_id)
        if include_document_context else []
    )

    # Build labeled context so the AI knows which file each excerpt came from
    context_text = ""
    if relevant_docs:
        parts = []
        for doc in relevant_docs[:3]:
            label = "[From " + str(doc.get("source", "document")) + "]"
            parts.append(label + _NEWLINE + doc["content"])
        # Add a small amount of context from files connected through shared
        # people, organizations, document types, and concepts. This lets the
        # assistant reason across related uploads instead of treating each one
        # as an isolated file.
        sources = [str(doc.get("source", "")) for doc in relevant_docs[:3] if doc.get("source")]
        related = _get_doc_service().related_document_context(user_id, sources)
        for doc in related:
            parts.append("[Related: " + doc["source"] + "]" + _NEWLINE + doc["content"])
        context_text = _DOUBLE_NEWLINE.join(parts)

    # Get AI response
    response = await _get_groq_service().chat(user_message, history, context_text)

    # Save to conversation history
    conversation_service.add_message(user_id, "user", user_message)
    conversation_service.add_message(user_id, "assistant", response)

    # Save to memory
    _get_memory_service().save_interaction(user_id, user_message, response)

    # Build reply keyboard: quick "Get file" buttons for matching docs + follow-ups
    keyboard = []
    seen_sources = set()
    for doc in relevant_docs[:2]:
        source = doc.get("source")
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        file_info = _get_doc_service().get_file_by_source(user_id, source)
        if file_info:
            icon = "🖼️" if file_info.get("file_type") == "photo" else "📄"
            name = file_info.get("ai_title") or source
            label = (icon + " Get " + name)[:60]
            keyboard.append([InlineKeyboardButton(label, callback_data="get_doc:" + file_info["_id"])])
    keyboard.extend(_build_followup_keyboard().inline_keyboard)

    await _reply_long(update, response, InlineKeyboardMarkup(keyboard))


async def _list_documents(update: Update, user_id: int, page: int = 0, category: str = "") -> None:
    """List all stored documents with tappable buttons to receive each file."""
    total = _get_doc_service().count_user_files(user_id, category=category or None)
    docs = _get_doc_service().list_user_files(
        user_id, skip=page * Config.DOCUMENT_PAGE_SIZE, limit=Config.DOCUMENT_PAGE_SIZE,
        category=category or None,
    )

    if not docs:
        empty_msg = (
            "📭 You haven't uploaded any documents yet."
            + _NEWLINE
            + "Upload a PDF, DOCX, or image to get started!"
        )
        await update.effective_message.reply_text(empty_msg, reply_markup=_build_followup_keyboard())
        return

    heading = f"{category.title()} Documents" if category else "Your Documents"
    header = f"📚 {heading} (page {page + 1} of {max(1, (total - 1) // Config.DOCUMENT_PAGE_SIZE + 1)})\nTap a document below to receive it, summarize it, or delete it:"
    lines = [header, ""]
    for i, doc in enumerate(docs, 1):
        icon = "🖼️" if doc.get("file_type") == "photo" else "📄"
        name = doc.get("ai_title") or doc["filename"]
        lines.append(f"{i}. {icon} {name}")
        if doc.get("description"):
            lines.append(f"   {doc['description'][:80]}")
        if doc.get("expiry_date"):
            lines.append(f"   Expires: {doc['expiry_date']}")
    listing = _NEWLINE.join(lines)

    await update.effective_message.reply_text(
        listing,
        reply_markup=_build_documents_keyboard(docs, page, total, category),
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads (PDF, DOCX)."""
    user_id = update.effective_user.id
    if not _vault_access_allowed(update):
        await _deny_group_vault_access(update)
        return
    document = update.message.document
    chat_id = update.effective_chat.id

    # Extract caption (user description) - this was previously dropped!
    caption = update.message.caption or ""

    # Validate file size before downloading
    file_size_mb = (document.file_size or 0) / (1024 * 1024)
    if file_size_mb > Config.MAX_FILE_SIZE_MB:
        await update.message.reply_text(
            f"❌ This file is too large ({file_size_mb:.1f}MB). "
            f"Maximum allowed size is {Config.MAX_FILE_SIZE_MB}MB."
        )
        return

    # Send typing indicator so the user knows the bot is working
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_document")

    if caption:
        await update.message.reply_text(
            f'📄 Processing "{caption[:100]}"... The first upload can take a couple of minutes while the search model loads.'
        )
    else:
        await update.message.reply_text(
            "📄 Processing your document... The first upload can take a couple of minutes while the search model loads."
        )

    is_image = (document.mime_type or "").startswith("image/")
    if is_image:
        result = await _get_doc_service().process_image(
            document, user_id, context, caption=caption, original_name=document.file_name or "image.jpg"
        )
    else:
        result = await _get_doc_service().process_document(document, user_id, context, caption=caption)

    if result["success"]:
        stored_name = result.get("stored_as") or result["filename"]
        success_msg = f"""✅ Document processed successfully!
📄 Saved as: {stored_name}
📊 {result['chunks']} chunks created
🧠 Indexed and ready for search.
💾 Stored in Telegram - ask me for it anytime!"""
        await update.message.reply_text(success_msg, reply_markup=_build_followup_keyboard())
    else:
        await update.message.reply_text(f"❌ Error processing document: {result['error']}")


# Seconds of quiet after the last photo in an album before combining.
_ALBUM_FLUSH_DELAY = 5


async def _buffer_album_photo(context: ContextTypes.DEFAULT_TYPE, message) -> None:
    """Buffer a photo that belongs to a media group (album).

    Telegram delivers each photo in an album as a separate message that shares
    the same ``media_group_id``. We collect them and, after a short quiet
    window, combine every photo into a single PDF.
    """
    # ``message`` is ``update.message`` (a PTB Message), so use Message
    # attributes -- it has no ``effective_user`` / ``effective_chat``.
    user_id = message.from_user.id
    group_id = message.media_group_id
    photo = message.photo[-1]
    chat_id = message.chat_id

    groups = context.user_data.setdefault("_album_groups", {})
    group = groups.setdefault(group_id, {"file_ids": [], "caption": ""})
    group["file_ids"].append(photo.file_id)
    if message.caption:
        group["caption"] = message.caption

    # Schedule (or reschedule) the combine so it only fires after the album is
    # complete. We use an asyncio task instead of the JobQueue because the
    # JobQueue may be disabled (see "JobQueue is unavailable" in the logs).
    tasks = context.user_data.setdefault("_album_tasks", {})
    previous = tasks.get(group_id)
    if previous and not previous.done():
        previous.cancel()
    task = asyncio.create_task(_delayed_flush(context, user_id, group_id, chat_id))
    tasks[group_id] = task

    if len(group["file_ids"]) == 1:
        await context.bot.send_message(
            chat_id=chat_id,
            text="🖼️ Collecting images from your album…",
        )


async def _delayed_flush(context: ContextTypes.DEFAULT_TYPE, user_id: int, group_id: str, chat_id: int) -> None:
    """Wait for the album quiet window, then combine the buffered photos."""
    await asyncio.sleep(_ALBUM_FLUSH_DELAY)
    await _flush_album(context, user_id, group_id, chat_id)


async def _flush_album(context: ContextTypes.DEFAULT_TYPE, user_id: int, group_id: str, chat_id: int) -> None:
    """Combine a buffered album of photos into a single PDF and store it."""
    groups = context.user_data.get("_album_groups", {})
    group = groups.pop(group_id, None)
    tasks = context.user_data.get("_album_tasks", {})
    tasks.pop(group_id, None)

    if not group or not group["file_ids"]:
        return

    file_ids = group["file_ids"]
    caption = group["caption"] or f"Combined PDF of {len(file_ids)} images"

    await context.bot.send_chat_action(chat_id=chat_id, action="upload_document")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_paths = []
            for index, file_id in enumerate(file_ids):
                path = os.path.join(tmp_dir, f"img_{index}.jpg")
                await _get_telegram_service().download_file(file_id, path)
                image_paths.append(path)

            pdf_path = os.path.join(tmp_dir, "combined.pdf")
            ImageComposer().combine_to_pdf(image_paths, pdf_path)

            # Send the combined PDF back to the user. The explicit ``filename``
            # makes Telegram report the file as ``combined.pdf`` so the vault
            # recognises it as a PDF.
            with open(pdf_path, "rb") as pdf_file:
                message = await context.bot.send_document(
                    chat_id=chat_id,
                    document=pdf_file,
                    caption=caption,
                    filename="combined.pdf",
                )

            # Store it in the vault by processing the Telegram-hosted PDF.
            document = getattr(message, "document", None)
            if document:
                result = await _get_doc_service().process_document(
                    document, user_id, context, caption=caption
                )
                if result.get("success"):
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"✅ Combined {len(file_ids)} images into a PDF and indexed it.\n"
                            f"📄 {result.get('stored_as') or result.get('filename')}\n"
                            f"📊 {result['chunks']} chunks"
                        ),
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"⚠️ Combined PDF sent, but indexing failed: {result.get('error')}",
                    )
            else:
                await context.bot.send_message(
                    chat_id=chat_id, text="✅ Combined PDF sent to your chat."
                )
    except Exception as e:
        logger.error(f"Failed to combine album into PDF: {e}")
        await context.bot.send_message(
            chat_id=chat_id, text=f"❌ Could not combine the images: {e}"
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image uploads. Uses caption as searchable text, falls back to OCR."""
    user_id = update.effective_user.id
    if not _vault_access_allowed(update):
        await _deny_group_vault_access(update)
        return

    # Media group (album) of multiple photos -> combine into a single PDF.
    if update.message.media_group_id:
        await _buffer_album_photo(context, update.message)
        return

    chat_id = update.effective_chat.id
    photo = update.message.photo[-1]  # Get the highest resolution photo
    caption = update.message.caption or ""

    file_size_mb = (photo.file_size or 0) / (1024 * 1024)
    if file_size_mb > Config.MAX_FILE_SIZE_MB:
        await update.message.reply_text(
            f"❌ This image is too large ({file_size_mb:.1f}MB). Maximum allowed size is {Config.MAX_FILE_SIZE_MB}MB."
        )
        return

    # Send upload photo action so the user knows the bot is working
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

    if caption:
        await update.message.reply_text(f'🖼️ Processing your image with caption: "{caption[:100]}"...')
    else:
        await update.message.reply_text(
            "🖼️ Processing your image... (No caption provided, will try OCR if available)"
        )

    result = await _get_doc_service().process_image(photo, user_id, context, caption=caption)

    if result["success"]:
        stored_name = result.get("stored_as") or result["filename"]
        success_msg = f"""✅ Image processed successfully!
🖼️ Saved as: {stored_name}
📊 {result['chunks']} chunks created
🧠 Indexed and ready for search.
💾 Stored in Telegram - ask me for it anytime!"""
        await update.message.reply_text(success_msg, reply_markup=_build_followup_keyboard())
    else:
        await update.message.reply_text(f"❌ Error processing image: {result['error']}")


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button callbacks."""
    query = update.callback_query
    callback_data = query.data
    user_id = update.effective_user.id
    logger.info(f"Callback query received: {callback_data}")

    # Answer the callback so the client stops showing the loading spinner.
    # Safe even if it fails (e.g. answered twice or network blip).
    try:
        await query.answer()
    except Exception as e:
        logger.warning(f"Could not answer callback query: {e}")

    if not _vault_access_allowed(update):
        await _deny_group_vault_access(update)
        return

    if callback_data.startswith("get_doc:"):
        # User tapped a specific document button - send the actual file
        object_id = callback_data.split(":", 1)[1]
        logger.info(f"Sending stored document {object_id} to user {user_id}")
        file_info = _get_doc_service().get_file_by_id(user_id, object_id)
        if not file_info:
            await _safe_edit(query, "⚠️ File not found. It may have been removed.")
            return

        _remember_document(context, file_info)

        # Use the same delivery routine as natural-language requests.  It gives
        # immediate feedback and falls back to the on-disk copy if Telegram no
        # longer accepts the stored file ID.
        await _send_stored_file(update, file_info)
        return

    if callback_data.startswith("delete_doc:"):
        object_id = callback_data.split(":", 1)[1]
        file_info = _get_doc_service().get_file_by_id(user_id, object_id)
        if not file_info:
            await _safe_edit(query, "⚠️ File not found. It may already have been deleted.")
            return
        _remember_document(context, file_info)
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("Yes, delete", callback_data=f"delete_confirm:{object_id}"),
            InlineKeyboardButton("Cancel", callback_data="list_docs"),
        ]])
        await _safe_edit(query, f"Delete '{file_info['ai_title'] or file_info['filename']}'? This cannot be undone.", keyboard)
        return

    if callback_data.startswith("facts:"):
        object_id = callback_data.split(":", 1)[1]
        file_info = _get_doc_service().get_file_by_id(user_id, object_id)
        if not file_info:
            await _safe_edit(query, "⚠️ File not found. It may have been deleted.")
            return
        _remember_document(context, file_info)
        display_name = file_info.get("ai_title") or file_info["filename"]
        text = await asyncio.to_thread(
            _get_doc_service().get_document_text, user_id, file_info["filename"]
        )
        if not text:
            await query.message.reply_text("I have no readable text for this file yet. Re-send it once so I can index it properly.")
            return
        status = await query.message.reply_text("📊 Pulling every detail…")
        facts = await _get_groq_service().extract_facts(text)
        rendered = _format_facts(display_name, facts)
        if not rendered:
            await status.edit_text(
                "I could not pull structured details from this file.\n"
                "Try ‘📝 Summary’, or ask me a question about it directly."
            )
            return
        max_len = 4000
        while rendered:
            chunk, rendered = rendered[:max_len], rendered[max_len:]
            try:
                await query.message.reply_text(chunk)
            except Exception as e:
                logger.error(f"Failed sending fact chunk: {e}")
                break
        try:
            await status.delete()
        except Exception:
            pass
        return

    if callback_data.startswith("rename_doc:"):
        object_id = callback_data.split(":", 1)[1]
        file_info = _get_doc_service().get_file_by_id(user_id, object_id)
        if not file_info:
            await _safe_edit(query, "⚠️ File not found. It may have been deleted.")
            return
        _remember_document(context, file_info)
        current = file_info.get("ai_title") or file_info["filename"]
        context.user_data["awaiting_rename_for"] = object_id
        await query.message.reply_text(
            f"✏️ Send the new name for “{current}”.\nExample: Electricity Bill — MSEB — August"
        )
        return

    if callback_data.startswith("summary:"):
        object_id = callback_data.split(":", 1)[1]
        file_info = _get_doc_service().get_file_by_id(user_id, object_id)
        if not file_info:
            await _safe_edit(query, "⚠️ File not found. It may have been deleted.")
            return
        _remember_document(context, file_info)
        text = await asyncio.to_thread(
            _get_doc_service().get_document_text, user_id, file_info["filename"]
        )
        if not text:
            await query.message.reply_text("I could not find searchable text for this file.")
            return
        await query.message.reply_text("🧠 Preparing a concise summary…")
        summary = await _get_groq_service().summarize(text)
        if summary:
            await query.message.reply_text(f"🧠 Summary of {file_info['ai_title'] or file_info['filename']}:\n\n{summary[:3900]}")
        else:
            await query.message.reply_text("I could not create a summary right now. Please try again.")
        return

    if callback_data.startswith("delete_confirm:"):
        object_id = callback_data.split(":", 1)[1]
        deleted = _get_doc_service().delete_file_by_id(user_id, object_id)
        await _safe_edit(query, "✅ Document deleted." if deleted else "⚠️ File was already deleted.")
        return

    if callback_data.startswith("docs_page:"):
        try:
            page = max(0, int(callback_data.split(":", 1)[1]))
        except ValueError:
            await _safe_edit(query, "⚠️ Invalid document page.")
            return
        # A callback message cannot be safely reused for long lists; send a fresh page.
        await _list_documents(update, user_id, page)
        return

    if callback_data.startswith("category_page:"):
        try:
            _, category, page_text = callback_data.split(":", 2)
            page = max(0, int(page_text))
        except ValueError:
            await _safe_edit(query, "⚠️ Invalid category page.")
            return
        if category not in _get_doc_service().list_categories(user_id):
            await _safe_edit(query, "⚠️ Category not found.")
            return
        await _list_documents(update, user_id, page, category)
        return

    if callback_data.startswith("category:"):
        category = callback_data.split(":", 1)[1]
        if category not in _get_doc_service().list_categories(user_id):
            await _safe_edit(query, "⚠️ Category not found.")
            return
        await _list_documents(update, user_id, category=category)
        return

    if callback_data == "upload_doc":
        await _safe_edit(query, "📎 Please upload a PDF or DOCX file to process.")
    elif callback_data == "list_docs":
        await _list_documents(update, user_id)
    elif callback_data == "categories":
        await _show_categories(update, user_id)
    elif callback_data == "search":
        await _safe_edit(query, "🔍 Type your search query and I'll find relevant information.")
    elif callback_data == "new_chat":
        _get_conversation_service().clear_history(user_id)
        _get_memory_service().clear_memory(user_id)
        await _safe_edit(
            query,
            "🆕 Conversation history cleared! Start a new chat.",
            reply_markup=_build_followup_keyboard(),
        )
    elif callback_data == "help":
        help_cb = """🤖 Telegram AI Assistant

• Send text to chat with AI
• Upload documents (PDF, DOCX) or images
• Files are stored in Telegram itself
• Say "give me my insurance document" to get the actual file
• Tap 📋 List My Documents to browse with buttons"""
        await _safe_edit(query, help_cb, reply_markup=_build_followup_keyboard())
    elif callback_data == "ask_question":
        await _safe_edit(query, "❓ Go ahead and ask your question!")
    elif callback_data == "end_conversation":
        _get_conversation_service().clear_history(user_id)
        _get_memory_service().clear_memory(user_id)
        await _safe_edit(query, "👋 Conversation ended. Send /start to begin again!")
    else:
        # Buttons sent by an older bot version can remain in the chat after a
        # deployment. Give a visible instruction instead of ignoring the tap.
        await query.message.reply_text(
            "⚠️ This is an older button. Tap ‘List My Documents’ again and use the new buttons."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a message to the user."""
    logger.error(f"Exception while handling an update: {context.error}")

    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ An unexpected error occurred. Please try again."
            )
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
