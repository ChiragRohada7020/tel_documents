"""
Document service - orchestrates document processing, chunking, and indexing.
Stores file_ids in MongoDB for later retrieval via Telegram API.
Generates rich AI metadata (title, tags, aliases, entities) for every upload.
Files are retained by Telegram; local downloads exist only during processing.
"""

import asyncio
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from bson import ObjectId

from config import Config
from bot.telegram_service import TelegramService
from database.mongo import get_db
from services.embedding_service import EmbeddingService
from services.groq_service import GroqService
from processors.pdf_processor import PDFProcessor
from processors.docx_processor import DocxProcessor
from processors.image_processor import ImageProcessor
from processors.chunker import TextChunker
from utils.helpers import sanitize_filename

logger = logging.getLogger(__name__)

class DocumentService:
    """Service for processing Telegram-hosted documents and indexing their text."""

    def __init__(self) -> None:
        self.telegram_service = TelegramService()
        self.embedding_service = EmbeddingService()
        self.groq_service = GroqService()
        self.chunker = TextChunker()
        self.db = get_db()

        self.pdf_processor = PDFProcessor()
        self.docx_processor = DocxProcessor()
        self.image_processor = ImageProcessor() if Config.ENABLE_OCR else None

    # Titles that carry no information and should be replaced by content
    # derived from the caption / extracted text.
    _GENERIC_TITLE_WORDS = {
        "image", "images", "img", "photo", "photograph", "pic", "pics",
        "picture", "screenshot", "document", "docs", "doc", "file", "scan",
        "scanned", "upload", "pdf", "jpg", "jpeg", "png", "webp", "whatsapp",
    }

    @classmethod
    def _is_generic_title(cls, title: str) -> bool:
        """True when a title is empty or something useless like 'image' / 'IMG_20240512.jpg'."""
        t = (title or "").strip().lower()
        if not t:
            return True
        t = re.sub(r"\.(jpg|jpeg|png|webp|pdf|docx?)$", "", t)
        t = re.sub(r"[_\-\s]+", " ", t).strip()
        words = [w for w in t.split() if w]
        if not words:
            return True
        # Pure camera/photo filename patterns: IMG_20240512, DSC00321, ...
        joined = "".join(words)
        if re.fullmatch(r"(img|image|photo|pic|dsc|dcim|screenshot|whatsapp)[0-9]*", joined):
            return True
        return all(w in cls._GENERIC_TITLE_WORDS for w in words)

    @staticmethod
    def _fallback_title_from_ocr(ocr_text: str, caption: str = "") -> str:
        """
        Derive a short human-readable name from the user's caption or from the
        first meaningful line of extracted/OCR text. Returns '' when nothing
        usable is found.
        """
        if caption and caption.strip():
            return re.sub(r"\s+", " ", caption.strip())[:80].strip(" -:,.")
        for raw in (ocr_text or "").splitlines():
            line = raw.strip()
            if not line or line.lower().startswith("ocr text from image"):
                continue
            letters = sum(ch.isalpha() for ch in line)
            digits = sum(ch.isdigit() for ch in line)
            if letters < 3 or letters <= digits:
                continue  # skip numbers-only / symbol junk lines
            snippet = " ".join(line.split()[:8])
            if len(snippet) >= 4:
                return snippet[:80].strip(" -:,.")
        return ""

    @staticmethod
    def _title_slug(title: str) -> str:
        """Turn any title into a clean lowercase filename base ('Electricity Bill - MSEB' -> 'electricity-bill-mseb')."""
        base = re.sub(r"[^A-Za-z0-9]+", "-", (title or "")).strip("-").lower()
        return base[:60].strip("-") or "document"

    def _unique_filename(self, user_id: int, filename: str) -> str:
        """
        Return a filename that does not collide with this user's existing
        documents by appending -2, -3 ... only when needed. This replaces the
        old random hex suffix so stored names stay human-friendly.
        """
        root = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        candidate = filename
        counter = 2
        try:
            while self.db.uploaded_files.find_one(
                {"user_id": user_id, "filename": candidate}, {"_id": 1}
            ):
                candidate = f"{root}-{counter}{ext}"
                counter += 1
                if counter > 50:
                    # Practically unreachable; bail out with a unique suffix.
                    return f"{root}-{uuid.uuid4().hex[:6]}{ext}"
        except Exception as e:
            logger.warning(f"Uniqueness check failed for {filename}: {e}")
        return candidate

    def _resolve_title(self, ai_metadata: Optional[Dict[str, Any]], ocr_text: str,
                       caption: str = "", original_name: str = "") -> str:
        """
        Pick the best available document title:
        specific AI title > caption > first meaningful content line > original filename stem.
        Generic AI titles like 'image' are rejected and replaced.
        """
        ai_title = ((ai_metadata or {}).get("title") or "").strip()
        if ai_title and not self._is_generic_title(ai_title):
            return ai_title
        fallback = self._fallback_title_from_ocr(ocr_text, caption)
        if fallback:
            return fallback
        stem = os.path.splitext(original_name)[0] if original_name else ""
        if stem and not self._is_generic_title(stem):
            return stem
        return ai_title or stem or "document"

    @staticmethod
    def _make_stored_name(title: str, unique: str, ext: str) -> str:
        """
        Backward-compatible shim: build a stored filename from a title.
        'PAN Card' + '.jpg'  ->  'pan-card.jpg'  (suffix kept empty for new code).
        """
        base = DocumentService._title_slug(title)
        suffix = f"-{unique}" if unique else ""
        return f"{base}{suffix}{ext}"

    async def process_document(self, document, user_id: int, context, caption: str = "") -> Dict[str, Any]:
        """
        Process an uploaded document (PDF, DOCX).
        Generates AI metadata, stores the Telegram file_id, and indexes the text.

        Args:
            document: The Telegram Document object.
            user_id: The Telegram user ID.
            context: The bot context.
            caption: Optional user-provided description (caption) for the document.
        """
        stored_name: Optional[str] = None
        metadata_stored = False
        try:
            original_name = document.file_name or "document"
            file_id = document.file_id
            unique = uuid.uuid4().hex[:8]
            orig_ext = os.path.splitext(original_name)[1].lower() or ".bin"

            # Download the file
            with tempfile.TemporaryDirectory() as tmp_dir:
                safe_temp_name = sanitize_filename(os.path.basename(original_name)) or "document"
                tmp_path = os.path.join(tmp_dir, safe_temp_name)
                await asyncio.wait_for(
                    self.telegram_service.download_file(file_id, tmp_path),
                    timeout=Config.DOWNLOAD_TIMEOUT_SECONDS,
                )

                # Extract text based on file type
                if original_name.lower().endswith(".pdf"):
                    text = await asyncio.to_thread(self.pdf_processor.extract_text, tmp_path)
                    if not text.strip() and self.image_processor and self.image_processor.is_available():
                        pages = await asyncio.to_thread(self.pdf_processor.render_pages_for_ocr, tmp_path)
                        ocr_pages = []
                        for page in pages:
                            ocr_pages.append(
                                await asyncio.to_thread(self.image_processor.extract_text_from_image, page)
                            )
                        text = "\n".join(ocr_pages)
                elif original_name.lower().endswith(".docx"):
                    text = await asyncio.to_thread(self.docx_processor.extract_text, tmp_path)
                else:
                    return {"success": False, "error": "Unsupported file type"}

                # The user's description is valuable searchable context too. It
                # supplements extracted/OCR text rather than replacing it.
                extracted_text = text.strip()
                text = "\n\n".join(part for part in [
                    f"User description: {caption.strip()}" if caption and caption.strip() else "",
                    f"Extracted document text: {extracted_text}" if extracted_text else "",
                ] if part)
                if not text.strip():
                    return {"success": False, "error": "No text or description was available for this document"}

                # Generate rich AI metadata (title, tags, aliases, entities)
                # Use the user's caption as the description for better metadata
                try:
                    ai_metadata = await asyncio.wait_for(
                        self.groq_service.generate_metadata(caption or "", text),
                        timeout=Config.METADATA_TIMEOUT_SECONDS,
                    )
                except Exception as meta_err:
                    logger.warning(f"Metadata generation failed for {original_name}: {meta_err}")
                    ai_metadata = None

                # Prefer a specific AI title; otherwise derive one from the
                # caption / actual document content instead of 'IMG_2024'.
                resolved_title = self._resolve_title(ai_metadata, extracted_text, caption, original_name)

                # Use a stable generated name for metadata/indexing. The file itself
                # remains on Telegram; the temporary download is deleted on exit.
                stored_name = sanitize_filename(self._unique_filename(user_id, self._title_slug(resolved_title) + orig_ext))

                # Store metadata + index chunks under the final stored name
                self._store_file_metadata(
                    user_id, stored_name, file_id, "document", metadata=ai_metadata
                )
                metadata_stored = True

                chunks = self.chunker.chunk(text)
                embeddings = await self._embeddings_for_chunks(chunks)
                await asyncio.to_thread(self._index_chunks, user_id, stored_name, chunks, embeddings)
                self._link_document_to_existing(user_id, stored_name, ai_metadata or {})

                return {
                    "success": True,
                    "filename": original_name,
                    "stored_as": stored_name,
                    "chunks": len(chunks),
                }

        except Exception as e:
            if isinstance(e, TimeoutError):
                e = TimeoutError("Processing took too long. Please try again in a moment.")
            logger.error(f"Error processing document: {e}")
            if stored_name and metadata_stored:
                self._remove_partial_upload(user_id, stored_name)
            return {"success": False, "error": str(e)}

    async def process_image(
        self,
        photo,
        user_id: int,
        context,
        caption: Optional[str] = None,
        original_name: str = "image.jpg",
    ) -> Dict[str, Any]:
        """
        Process an uploaded image.
        Uses the caption as searchable text if provided; falls back to OCR.
        Generates AI metadata, stores the Telegram file_id, and indexes the text.
        """
        stored_name: Optional[str] = None
        metadata_stored = False
        try:
            file_id = photo.file_id
            unique = uuid.uuid4().hex[:8]
            extension = os.path.splitext(original_name)[1].lower()
            if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
                extension = ".jpg"

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, f"upload_{unique}{extension}")
                await asyncio.wait_for(
                    self.telegram_service.download_file(file_id, tmp_path),
                    timeout=Config.DOWNLOAD_TIMEOUT_SECONDS,
                )

                # Always run OCR when available, even with a caption. The
                # caption provides user context; OCR contributes details seen
                # inside the image, and both are indexed together.
                ocr_text = ""
                if self.image_processor and self.image_processor.is_available():
                    ocr_text = await asyncio.to_thread(self.image_processor.extract_text, tmp_path)
                text = "\n\n".join(part for part in [
                    f"User description: {caption.strip()}" if caption and caption.strip() else "",
                    f"OCR text from image: {ocr_text.strip()}" if ocr_text and ocr_text.strip() else "",
                ] if part)

                if not text.strip():
                    return {
                        "success": False,
                        "error": "No text was detected. Please provide a caption/description for the image.",
                    }

                # Generate rich AI metadata using caption + OCR/content text
                try:
                    ai_metadata = await asyncio.wait_for(
                        self.groq_service.generate_metadata(caption or "", text),
                        timeout=Config.METADATA_TIMEOUT_SECONDS,
                    )
                except Exception as meta_err:
                    logger.warning(f"Metadata generation failed: {meta_err}")
                    ai_metadata = None

                # Never settle for 'image': resolve a meaningful name from the
                # caption or whatever OCR managed to read off the photo.
                resolved_title = self._resolve_title(ai_metadata, ocr_text, caption)

                # Store only metadata and the Telegram file ID; no local copy is kept.
                stored_name = sanitize_filename(self._unique_filename(user_id, self._title_slug(resolved_title) + extension))

                # Store metadata + index chunks under the final stored name
                self._store_file_metadata(
                    user_id, stored_name, file_id, "photo",
                    description=caption or "", metadata=ai_metadata,
                )
                metadata_stored = True

                chunks = self.chunker.chunk(text)
                embeddings = await self._embeddings_for_chunks(chunks)
                await asyncio.to_thread(self._index_chunks, user_id, stored_name, chunks, embeddings)
                self._link_document_to_existing(user_id, stored_name, ai_metadata or {})

                return {
                    "success": True,
                    "filename": stored_name,
                    "stored_as": stored_name,
                    "chunks": len(chunks),
                }

        except Exception as e:
            if isinstance(e, TimeoutError):
                e = TimeoutError("Processing took too long. Please try again in a moment.")
            logger.error(f"Error processing image: {e}")
            if stored_name and metadata_stored:
                self._remove_partial_upload(user_id, stored_name)
            return {"success": False, "error": str(e)}

    def _remove_partial_upload(self, user_id: int, filename: str) -> None:
        """Remove partial metadata and search chunks after a failed upload."""
        try:
            self.db.uploaded_files.delete_one({"user_id": user_id, "filename": filename})
            self.db.document_chunks.delete_many({"user_id": user_id, "source": filename})
        except Exception as cleanup_error:
            logger.error(f"Could not clean partial upload {filename}: {cleanup_error}")

    def _store_file_metadata(
        self,
        user_id: int,
        filename: str,
        file_id: str,
        file_type: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store file metadata in MongoDB for later retrieval."""
        doc: Dict[str, Any] = {
            "user_id": user_id,
            "filename": filename,
            "file_id": file_id,
            "file_type": file_type,
            "description": description,
        }
        if metadata:
            doc.update({
                "ai_title": metadata.get("title", ""),
                "ai_description": metadata.get("description", ""),
                "category": metadata.get("category", ""),
                "document_type": metadata.get("document_type", ""),
                "expiry_date": metadata.get("expiry_date", ""),
                "tags": metadata.get("tags", []) or [],
                "search_aliases": metadata.get("search_aliases", []) or [],
                "entities": metadata.get("entities", {}) or {},
            })
        self.db.uploaded_files.update_one(
            {"user_id": user_id, "filename": filename},
            {"$set": doc},
            upsert=True,
        )
        logger.info(f"Stored file metadata for {filename} (user {user_id})")

    def _update_file_metadata(self, user_id: int, filename: str, metadata: Dict[str, Any]) -> None:
        """Attach AI-generated metadata to an already-stored file."""
        self.db.uploaded_files.update_one(
            {"user_id": user_id, "filename": filename},
            {"$set": {
                "ai_title": metadata.get("title", ""),
                "ai_description": metadata.get("description", ""),
                "category": metadata.get("category", ""),
                "document_type": metadata.get("document_type", ""),
                "expiry_date": metadata.get("expiry_date", ""),
                "tags": metadata.get("tags", []) or [],
                "search_aliases": metadata.get("search_aliases", []) or [],
                "entities": metadata.get("entities", {}) or {},
            }},
        )
        logger.info(f"Updated AI metadata for {filename} (user {user_id})")

    def _index_chunks(
        self,
        user_id: int,
        source: str,
        chunks: list,
        embeddings: list,
    ) -> None:
        """Index document chunks with their embeddings into MongoDB."""
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            documents.append({
                "user_id": user_id,
                "source": source,
                "chunk_index": i,
                "content": chunk,
                "embedding": embedding,
            })

        if documents:
            self.db.document_chunks.insert_many(documents)
            logger.info(f"Indexed {len(documents)} chunks for user {user_id} from {source}")

    async def _embeddings_for_chunks(self, chunks: list) -> list:
        """Avoid loading the large local ML model when cloud memory is limited."""
        if not Config.ENABLE_LOCAL_EMBEDDINGS:
            return [None] * len(chunks)
        return await asyncio.wait_for(
            asyncio.to_thread(self.embedding_service.embed_texts, chunks),
            timeout=Config.EMBEDDING_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _connection_terms(metadata: Dict[str, Any]) -> set[str]:
        """Return privacy-scoped, useful concepts for connecting documents."""
        values: List[str] = []
        values.extend(metadata.get("tags", []) or [])
        values.extend(metadata.get("search_aliases", []) or [])
        values.extend([metadata.get("category", ""), metadata.get("document_type", "")])
        for group in (metadata.get("entities", {}) or {}).values():
            if isinstance(group, list):
                values.extend(str(item) for item in group)
        return {
            re.sub(r"[^a-z0-9]+", "", value.lower())
            for value in values
            if isinstance(value, str) and len(re.sub(r"[^a-z0-9]+", "", value.lower())) >= 3
        }

    def _link_document_to_existing(self, user_id: int, filename: str, metadata: Dict[str, Any]) -> int:
        """Create/update meaningful links from one document to the user's other files."""
        own_terms = self._connection_terms(metadata)
        if not own_terms:
            return 0
        links_created = 0
        for other in self.db.uploaded_files.find({"user_id": user_id, "filename": {"$ne": filename}}):
            other_terms = self._connection_terms(other)
            shared = sorted(own_terms & other_terms)
            # A shared named entity is strong; otherwise require two concepts to
            # avoid connecting everything merely because it shares a category.
            other_entities = self._connection_terms({"entities": other.get("entities", {})})
            own_entities = self._connection_terms({"entities": metadata.get("entities", {})})
            if len(shared) < 2 and not (own_entities & other_entities):
                continue
            pair = sorted([filename, other["filename"]])
            self.db.document_links.update_one(
                {"user_id": user_id, "left": pair[0], "right": pair[1]},
                {"$set": {"shared_terms": shared[:12]}},
                upsert=True,
            )
            links_created += 1
        logger.info(f"Connected {filename} to {links_created} related documents for user {user_id}")
        return links_created

    def rebuild_document_links(self, user_id: int) -> int:
        """Rebuild the user's private document graph from existing metadata."""
        self.db.document_links.delete_many({"user_id": user_id})
        count = 0
        for record in self.db.uploaded_files.find({"user_id": user_id}):
            count += self._link_document_to_existing(user_id, record["filename"], record)
        return count

    def related_document_context(self, user_id: int, sources: List[str], limit: int = 3) -> List[Dict[str, str]]:
        """Fetch short excerpts from files connected to retrieved sources."""
        related: List[str] = []
        for source in sources:
            for link in self.db.document_links.find({"user_id": user_id, "$or": [{"left": source}, {"right": source}]}):
                other = link["right"] if link["left"] == source else link["left"]
                if other not in sources and other not in related:
                    related.append(other)
        items = []
        for filename in related[:limit]:
            text = self.get_document_text(user_id, filename, max_chars=700)
            if text:
                items.append({"source": filename, "content": text})
        return items

    @staticmethod
    def _file_to_dict(r: dict) -> Dict[str, Any]:
        """Convert a stored-file DB record into a plain dict."""
        return {
            "_id": str(r.get("_id", "")),
            "filename": r["filename"],
            "file_id": r["file_id"],
            "file_type": r.get("file_type", "document"),
            "description": r.get("description", ""),
            "ai_title": r.get("ai_title", ""),
            "category": r.get("category", ""),
            "expiry_date": r.get("expiry_date", ""),
        }

    def find_document(self, user_id: int, query: str) -> Optional[Dict[str, Any]]:
        """
        Find the BEST-matching stored file for a natural-language request.

        Matching priority:
          1. Filename contains the query text
          2. Description / AI title / tags / search_aliases keyword overlap
             (aliases cover abbreviations, Hinglish, informal names, typos)
          3. Indexed content search - highest text-score chunk whose source
             is a stored file
        """
        matches = self.find_documents(user_id, query, limit=1)
        return matches[0] if matches else None

    def find_documents(self, user_id: int, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find ALL stored files matching a natural-language request, best first.

        Uses matching tiers (filename, description / AI title / tags /
        search_aliases, indexed content) but returns every match so duplicate
        uploads of the same document are all surfaced.
        """
        query_lower = (query or "").lower().strip()
        if not query_lower:
            return []

        words = [w for w in re.findall(r"[a-z0-9]+", query_lower) if len(w) > 1]
        files = list(self.db.uploaded_files.find({"user_id": user_id}))
        if not files:
            return []

        scored: List[tuple] = []
        seen = set()

        # 1) Filename matches (very high score)
        for f in files:
            if query_lower in f["filename"].lower():
                scored.append((1000.0, self._file_to_dict(f)))
                seen.add(f["filename"])

        # 2) Description / AI title / tags / search_aliases scoring
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", s.lower())

        query_norm = _norm(query_lower)
        for f in files:
            if f["filename"] in seen:
                continue
            hay_parts = [
                f.get("description") or "",
                f.get("ai_title") or "",
                f.get("ai_description") or "",
            ]
            hay_parts.extend(f.get("tags") or [])
            hay_parts.extend(f.get("search_aliases") or [])
            hay = " ".join(hay_parts).lower()
            if not hay.strip():
                continue
            score = float(sum(1 for w in set(words) if w in hay))
            if query_lower in hay:
                score += len(set(words)) + 1
            if query_norm and query_norm in _norm(hay):
                score += len(set(words)) + 2
            if score > 0:
                scored.append((score, self._file_to_dict(f)))
                seen.add(f["filename"])

        # 3) Indexed content sources (lowest tier)
        try:
            chunks = list(
                self.db.document_chunks.find(
                    {"user_id": user_id, "$text": {"$search": query_lower}},
                    {"source": 1, "score": {"$meta": "textScore"}},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(10)
            )
        except Exception:
            chunks = []

        sources_by_name = {f["filename"]: f for f in files}
        for chunk in chunks:
            src = chunk.get("source")
            if src and src not in seen and src in sources_by_name:
                scored.append((1.0, self._file_to_dict(sources_by_name[src])))
                seen.add(src)

        scored.sort(key=lambda t: -t[0])
        return [d for _, d in scored[:limit]]

    def get_file_by_source(self, user_id: int, source: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a stored file record by its source filename.

        Args:
            user_id: The Telegram user ID.
            source: The filename the chunks were indexed under.

        Returns:
            A dict with '_id', 'filename', 'file_id', 'file_type' if found, None otherwise.
        """
        r = self.db.uploaded_files.find_one({"user_id": user_id, "filename": source})
        if not r:
            return None
        d = self._file_to_dict(r)
        d["_id"] = str(r["_id"])
        return d

    def list_user_files(
        self, user_id: int, skip: int = 0, limit: Optional[int] = None, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all stored files for a user.

        Args:
            user_id: The Telegram user ID.

        Returns:
            A list of dicts with '_id', 'filename', 'file_id', 'file_type',
            'ai_title' and 'description'.
        """
        query: Dict[str, Any] = {"user_id": user_id}
        if category:
            query["category"] = category
        results = self.db.uploaded_files.find(query).sort("_id", -1).skip(skip)
        if limit is not None:
            results = results.limit(limit)
        return [
            {
                "_id": str(r["_id"]),
                "filename": r["filename"],
                "file_id": r["file_id"],
                "file_type": r.get("file_type", "document"),
                "ai_title": r.get("ai_title", ""),
                "description": r.get("description", ""),
                "category": r.get("category", ""),
                "expiry_date": r.get("expiry_date", ""),
            }
            for r in results
        ]

    def count_user_files(self, user_id: int, category: Optional[str] = None) -> int:
        """Return the number of files a user has stored."""
        query: Dict[str, Any] = {"user_id": user_id}
        if category:
            query["category"] = category
        return self.db.uploaded_files.count_documents(query)

    def list_categories(self, user_id: int) -> List[str]:
        """Return the user's non-empty document categories."""
        return sorted(
            category for category in self.db.uploaded_files.distinct("category", {"user_id": user_id})
            if isinstance(category, str) and category.strip()
        )

    def get_document_text(self, user_id: int, filename: str, max_chars: int = 12000) -> str:
        """Return indexed text for a file, ordered by original chunk position."""
        chunks = self.db.document_chunks.find(
            {"user_id": user_id, "source": filename}, {"content": 1}
        ).sort("chunk_index", 1)
        parts: List[str] = []
        total = 0
        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue
            remaining = max_chars - total
            if remaining <= 0:
                break
            parts.append(content[:remaining])
            total += len(parts[-1])
        return "\n".join(parts)

    def append_user_note(self, user_id: int, object_id: str, note: str) -> Optional[Dict[str, Any]]:
        """Attach a user-supplied detail to one file and index it for search."""
        file_info = self.get_file_by_id(user_id, object_id)
        note = (note or "").strip()
        if not file_info or not note:
            return None
        entry = {"text": note, "created_at": datetime.now(timezone.utc)}
        self.db.uploaded_files.update_one(
            {"_id": ObjectId(object_id), "user_id": user_id},
            {"$push": {"user_notes": entry}},
        )
        searchable_note = f"User-added detail for {file_info['filename']}: {note}"
        chunks = self.chunker.chunk(searchable_note)
        embeddings = self.embedding_service.embed_texts(chunks) if Config.ENABLE_LOCAL_EMBEDDINGS else [None] * len(chunks)
        self._index_chunks(user_id, file_info["filename"], chunks, embeddings)
        logger.info(f"Added a searchable user note to {file_info['filename']} for user {user_id}")
        return file_info

    def delete_file_by_id(self, user_id: int, object_id: str) -> Optional[Dict[str, Any]]:
        """Delete a user's stored metadata and indexed chunks."""
        file_info = self.get_file_by_id(user_id, object_id)
        if not file_info:
            return None
        filename = file_info["filename"]
        self.db.document_chunks.delete_many({"user_id": user_id, "source": filename})
        self.db.uploaded_files.delete_one({"_id": ObjectId(object_id), "user_id": user_id})
        logger.info(f"Deleted file {filename} for user {user_id}")
        return file_info

    def get_file_by_id(self, user_id: int, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single stored file record by its MongoDB id.

        Args:
            user_id: The Telegram user ID.
            object_id: The MongoDB ObjectId string.

        Returns:
            A dict with 'filename', 'file_id', 'file_type', 'ai_title' if found, None otherwise.
        """
        try:
            r = self.db.uploaded_files.find_one({
                "_id": ObjectId(object_id),
                "user_id": user_id,
            })
        except Exception:
            return None
        if not r:
            return None
        return {
            "_id": str(r["_id"]),
            "filename": r["filename"],
            "file_id": r["file_id"],
            "file_type": r.get("file_type", "document"),
            "ai_title": r.get("ai_title", ""),
        }
