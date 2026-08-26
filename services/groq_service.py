"""
Groq service - handles AI chat completions and metadata generation via the Groq API.
"""

import json
import logging
from typing import List, Dict, Optional

from groq import AsyncGroq

from config import Config

logger = logging.getLogger(__name__)


class GroqService:
    """Service for interacting with the Groq API for LLM inference."""

    def __init__(self) -> None:
        self.client = AsyncGroq(api_key=Config.GROQ_API_KEY, timeout=Config.METADATA_TIMEOUT_SECONDS, max_retries=2)
        self.model = Config.GROQ_MODEL
        self.max_tokens = Config.AI_MAX_TOKENS
        self.temperature = Config.AI_TEMPERATURE

    async def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        context: str = "",
    ) -> str:
        """
        Generate a chat response from the LLM.

        Args:
            message: The current user message.
            history: Previous conversation messages as a list of dicts
                     with 'role' and 'content' keys.
            context: Additional context (e.g., retrieved documents) to prepend.

        Returns:
            The AI-generated response string.
        """
        messages: List[Dict[str, str]] = []

        # Keep the request small enough for Groq's token-per-minute limits
        # (e.g. free tier allows ~8000 TPM). Trim history and context.
        max_history_messages = 8
        max_history_chars = 600
        max_context_chars = 2500

        trimmed_history = (history or [])[-max_history_messages:]

        # Base persona - always present so the bot feels consistent & smart
        messages.append({
            "role": "system",
            "content": (
                "You are a smart, friendly personal assistant running inside a Telegram "
                "document vault. The user stores their personal documents with you. "
                "Rules: "
                "(1) Reply in the same language/style the user writes in - English, "
                "Hindi, or Hinglish (e.g. 'mera aadhar card kaha hai' gets a Hinglish reply). "
                "(2) Use conversation history to understand follow-up questions like "
                "'send it', 'and what about the date?', 'wo wala do'. "
                "(3) Be concise, warm and helpful. Never invent facts."
            ),
        })

        # Add document context if provided
        if context:
            context = context[:max_context_chars]
            messages.append({
                "role": "system",
                "content": (
                    "CONTEXT below holds excerpts retrieved from the user's own files, "
                    "each excerpt labeled with its source file name. Rules: "
                    "(1) If the CONTEXT contains information relevant to the question, "
                    "answer using it and mention the source file name(s). "
                    "(2) Never say you have no documents or no details when CONTEXT is non-empty. "
                    "(3) If CONTEXT is empty or unrelated to the question, briefly ask what you would need. "
                    "CONTEXT: " + context
                ),
            })

        # Add conversation history (trimmed)
        for msg in trimmed_history:
            content = str(msg.get("content", ""))[:max_history_chars]
            messages.append({"role": msg["role"], "content": content})

        # Add current user message
        messages.append({"role": "user", "content": message})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return "Sorry, I encountered an error while processing your request."

    async def classify_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Use a fast LLM to classify the user's intent.

        Understands English, Hinglish, typos and context-dependent follow-ups
        (e.g. "send it" right after listing documents).

        Returns:
            {"intent": <one of get_document|list_documents|search_knowledge|chat>,
             "query": <extracted search/document query or "">}
            None if classification fails.
        """
        recent = (history or [])[-4:]
        history_text = chr(10).join(
            f"{m.get('role', 'user')}: {str(m.get('content', ''))[:150]}" for m in recent
        ) or "(none)"

        prompt = f"""You are an intent classifier for a personal document assistant Telegram bot.
The user stores documents (PDFs, images) and can ask for them back, search them, or just chat.

Classify the user's message into EXACTLY ONE intent:
- "get_document": user wants to RECEIVE/see/download a specific file they uploaded.
  Examples: "give me my aadhar card", "send my insurance pdf", "mera pan card dikhao",
  "aadhar card do", "show me my degree certificate", "send it", "wo wala do"
- "list_documents": user wants to see ALL their documents / browse the list.
  Examples: "list my documents", "what files do you have", "mere documents dikhao",
  "show all docs", "kya kya documents hai"
- "search_knowledge": user asks a question whose answer is inside their documents,
  or explicitly asks to search. Examples: "what is my PAN number?",
  "find my rent agreement amount", "mera aadhaar number kya hai", "search insurance"
- "chat": general conversation, greetings, questions not about their documents.
  Examples: "hello", "how are you", "tell me a joke", "what can you do"

CONVERSATION HISTORY (for follow-up context):
{history_text}

USER MESSAGE: "{message}"

Respond with ONLY valid JSON, no other text:
{{"intent": "<intent>", "query": "<the specific document/topic the user wants, empty string for chat/list>"}}"""

        try:
            response = await self.client.chat.completions.create(
                model=Config.GROQ_CLASSIFIER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            raw = response.choices[0].message.content or ""
            data = self._parse_json(raw)
            if not data:
                return None
            intent = str(data.get("intent", "")).strip()
            valid = {"get_document", "list_documents", "search_knowledge", "chat"}
            if intent not in valid:
                return None
            return {
                "intent": intent,
                "query": str(data.get("query", "")).strip(),
            }
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")
            return None

    async def generate_metadata(self, user_description: str, document_text: str) -> Optional[dict]:
        """
        Analyze a document (description + content) and generate rich searchable
        metadata as JSON: title, description, category, document_type, tags,
        search_aliases (abbreviations / Hinglish / typos) and entities.

        Returns a dict on success, None on failure.
        """
        desc = (user_description or "").strip() or "(none provided)"
        doc_excerpt = (document_text or "").strip()[:4000] or "(no text extracted)"

        prompt = f"""You are an intelligent personal document organizer.

The user uploads a document and may provide a short description.

Your job is to deeply understand what the document is and generate rich metadata so that the document can be found later even if the user uses:
- Different names
- Abbreviations
- Common spelling mistakes
- Typing mistakes
- Singular/plural variations
- Hinglish words
- Informal language
- Related concepts
- Partial names

USER DESCRIPTION:
{desc}

DOCUMENT CONTENT:
{doc_excerpt}

Analyze both the user's description and document content.

Return ONLY valid JSON in exactly this format:

{{
  "title": "Clear human-readable document name",
  "description": "A short description explaining what this document is and what important information it contains.",
  "category": "one of identity, financial, insurance, education, medical, legal, property, employment, travel, utilities, or other",
  "document_type": "specific document type",
  "expiry_date": "YYYY-MM-DD if an explicit expiry/valid-until date is present, otherwise an empty string",
  "tags": ["important keyword 1", "important keyword 2"],
  "search_aliases": ["alternative name 1", "common spelling variation", "abbreviation", "informal name", "hinglish variation", "likely typing mistake"],
  "entities": {{"people": [], "organizations": [], "locations": [], "important_numbers": []}}
}}

METADATA RULES:

1. Understand the actual meaning of the document. Do not rely only on the filename.
2. Create a clear, SPECIFIC title a normal person would recognise at a glance. Prefer the pattern "<Document Type> - <Organization/Person> - <Period/Number>", e.g. "Electricity Bill - MSEDCL - August 2026", "PAN Card - Chirag Rohada", "Rent Receipt - Sharma PG - June". NEVER output generic titles such as "Image", "Photo", "Screenshot", "Document", "File" or camera names like "IMG_20240512".
2b. Capture EVERY useful detail present in the content so nothing searchable is lost: invoice/receipt/bill numbers, amounts and currencies, issue/due/expiry dates, policy or registration numbers (without breaking rule 10), organization names, persons, locations, subject matter, months/years. Put identifiers into entities.important_numbers and topical words into tags.
3. Generate useful tags that describe document type, purpose, related concepts, important subjects, organizations, relevant categories.
4. Generate search_aliases that help find the document even when the user writes the name differently.
5. Include common spelling variations ONLY when they are realistic and useful.
6. Include common abbreviations where appropriate.
7. Include natural Hinglish variations if the document is likely to be searched by an Indian user.
8. Do NOT generate random or meaningless spelling mistakes.
9. If the user description conflicts with the actual document content, prioritize the actual document content but consider the user's description as additional context.
10. Do not expose sensitive information unnecessarily. Do not put full Aadhaar numbers, PAN numbers, bank account numbers, passwords, or other highly sensitive identifiers into tags or search_aliases.
11. Keep tags and aliases useful for searching. Avoid duplicate values.
12. Generate between 10 and 25 combined tags and search_aliases when appropriate.
13. Use the category list exactly; select "other" if none fits.
14. Only set expiry_date when the document explicitly gives an expiry, valid-until, renewal-due, or end date. Never guess dates.

Return ONLY valid JSON."""

        # NOTE: reasoning models (e.g. openai/gpt-oss-*) spend completion tokens
        # on chain-of-thought before emitting the JSON; a small budget truncated
        # the object mid-way and made metadata generation fail silently.
        meta_max_tokens = max(self.max_tokens, 2600)

        async def _create(extra_kwargs: Optional[dict]) -> object:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": meta_max_tokens,
            }
            payload.update(extra_kwargs or {})
            return await self.client.chat.completions.create(**payload)

        try:
            try:
                # Lower reasoning effort keeps thinking short so real JSON fits.
                response = await _create({"reasoning_effort": "low"})
            except Exception as effort_err:
                if "reasoning_effort" not in str(effort_err).lower():
                    raise
                response = await _create(None)  # model without that parameter
            raw = response.choices[0].message.content or ""
            metadata = self._parse_json(raw)
            if metadata is None:
                logger.warning("Metadata generation returned unparseable output.")
            return metadata
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            return None

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Extract a JSON object from model output (tolerates code fences / think blocks)."""
        text = (raw or "").strip()
        # Some models emit <think>...</think> before the actual answer.
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    async def extract_facts(self, document_text: str) -> Optional[List[Dict[str, str]]]:
        """
        Pull every concrete detail out of a document's text as a structured
        fact sheet: names, IDs/numbers, amounts, dates, addresses, statuses...

        Returns a list of {"label": ..., "value": ...} dicts, or None on failure.
        """
        excerpt = (document_text or "").strip()[:6000]
        if not excerpt:
            return None

        prompt = f"""You are a precise information-extraction engine reading an OCR'd personal document.

DOCUMENT TEXT:
{excerpt}

Extract EVERY concrete detail present in the text. Include things like:
person names, organization names, ID/account/policy/registration/bill/invoice numbers,
amounts with currency, unit counts, issue/due/expiry/renewal dates, addresses,
phone numbers, vehicle/model references, status flags, periods covered, grades/positions.

Return ONLY valid JSON exactly in this shape (6-18 entries is typical):

{{"facts": [
  {{"label": "Consumer Name", "value": "Chirag Rohada"}},
  {{"label": "Bill Number", "value": "2026-08/77341"}},
  {{"label": "Amount Due", "value": "INR 2450"}},
  {{"label": "Due Date", "value": "2026-09-12"}}
]}}

RULES:
1. Every entry must come from the document text itself - never invent values.
2. Keep each value short and copy verbatim where possible.
3. Use clear human labels ("Bill Number", not "field_4").
4. Skip headers, page numbers, decorations, and OCR artifacts.
5. If the text has almost nothing extractable, return {{"facts": []}}."""

        meta_max_tokens = max(self.max_tokens, 2600)

        async def _create(extra_kwargs: Optional[dict]) -> object:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": meta_max_tokens,
            }
            payload.update(extra_kwargs or {})
            return await self.client.chat.completions.create(**payload)

        try:
            try:
                response = await _create({"reasoning_effort": "low"})
            except Exception as effort_err:
                if "reasoning_effort" not in str(effort_err).lower():
                    raise
                response = await _create(None)
            raw = response.choices[0].message.content or ""
            data = self._parse_json(raw)
            if not isinstance(data, dict):
                logger.warning("Fact extraction returned unparseable output.")
                return None
            facts = data.get("facts")
            if not isinstance(facts, list):
                return []
            cleaned: List[Dict[str, str]] = []
            for entry in facts:
                if not isinstance(entry, dict):
                    continue
                label = str(entry.get("label", "")).strip().rstrip(": ")
                value = str(entry.get("value", "")).strip()
                if label and value and len(cleaned) < 25:
                    cleaned.append({"label": label, "value": value})
            return cleaned
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return None

    async def summarize(self, text: str, max_tokens: int = 500) -> str:
        """Summarize a given text using the LLM."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Summarize the following text concisely:"},
                    {"role": "user", "content": text},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return ""
