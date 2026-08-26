"""
Image processor - extracts text from images using OCR.

Primary engine: OCR.space REST API (cloud, no local install required).
Fallback engine: Tesseract, used automatically when no OCR.space key is
configured but a local Tesseract install is found.

OCR is optional overall; captions/descriptions remain usable as searchable
text even when neither engine is available.
"""

import io
import logging
import os
import re
import shutil
import time
from typing import List, Optional

import httpx

from config import Config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Service for extracting text from images using OCR."""

    # Common Tesseract install locations on Windows (fallback engine)
    _TESSERACT_PATHS = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    )

    # Map common 2-letter codes to 3-letter language codes accepted by both
    # Tesseract and OCR.space (e.g. 'en' -> 'eng', 'hi' -> 'hin').
    _LANG_MAP = {
        "en": "eng", "hi": "hin", "mr": "mar", "ta": "tam", "te": "tel",
        "kn": "kan", "gu": "guj", "bn": "ben", "pa": "pan", "ur": "urd",
        "sa": "san", "ne": "nep", "si": "sin",
        # Already-3-letter codes and OCR.space specials pass through untouched.
        "eng": "eng", "hin": "hin", "mar": "mar", "auto": "auto",
    }

    def __init__(self) -> None:
        self.api_key = (getattr(Config, "OCR_SPACE_API_KEY", "") or "").strip()
        self.api_endpoint = getattr(Config, "OCR_SPACE_ENDPOINT", "") or "https://api.ocr.space/parse/image"
        self.api_engine = int(getattr(Config, "OCR_SPACE_ENGINE", 2) or 2)
        self.timeout_seconds = int(getattr(Config, "OCR_TIMEOUT_SECONDS", 30) or 30)
        self.ocr_languages = self._normalize_languages(Config.OCR_LANGUAGES)
        # Only probe for a local Tesseract binary when there is no cloud key.
        self._tesseract_available = False if self.api_key else self._check_tesseract()

    # ------------------------------------------------------------------
    # Availability / configuration helpers
    # ------------------------------------------------------------------

    def _normalize_languages(self, raw: str) -> str:
        """
        Convert configured languages (e.g. 'en,hi,mr') into engine format
        (e.g. 'eng,hin,mar'). Falls back to 'eng' when nothing valid remains.
        """
        raw = (raw or "en").lower()
        codes = []
        for part in re.split(r"[,+;\s]+", raw):
            part = part.strip()
            if not part:
                continue
            codes.append(self._LANG_MAP.get(part, part))
        codes = [c for c in codes if c]
        return "+".join(codes) if codes else "eng"

    def _locate_tesseract(self) -> bool:
        """Point pytesseract at the Tesseract binary (Windows-friendly)."""
        import pytesseract

        exe = shutil.which("tesseract")
        if exe:
            pytesseract.pytesseract.tesseract_cmd = exe
            return True
        for path in self._TESSERACT_PATHS:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
        return False

    def _check_tesseract(self) -> bool:
        """Check if local Tesseract OCR is available (fallback engine)."""
        try:
            import pytesseract

            if not self._locate_tesseract():
                logger.info(
                    "Tesseract binary not found locally; cloud OCR handles OCR instead."
                )
                return False
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available.")
            return True
        except Exception as e:
            logger.info(f"Tesseract OCR not available: {e}")
            return False

    def is_available(self) -> bool:
        """Check if any OCR engine is available."""
        return bool(self.api_key) or self._tesseract_available

    def _api_languages(self) -> str:
        """
        OCR.space accepts ONE language code or 'auto' per request - combined
        codes are rejected with E201 (verified against the live API).

        Single configured language -> send it directly.
        Multiple configured languages -> 'auto' (engine-side language
        detection), unless OCR_SPACE_LANGUAGE_MODE='first' forces the first one.
        """
        codes = [c for c in self.ocr_languages.split("+") if c]
        if len(codes) <= 1:
            return codes[0] if codes else "eng"
        mode = getattr(Config, "OCR_SPACE_LANGUAGE_MODE", "auto").lower()
        return codes[0] if mode == "first" else "auto"

    # ------------------------------------------------------------------
    # OCR.space cloud engine
    # ------------------------------------------------------------------

    def _call_ocr_space(self, file_bytes: bytes, filename: str, languages: str,
                        overlay: bool = False, engine: Optional[int] = None):
        """
        Send image bytes to OCR.space and return extracted text.

        Retries once on transient gateway errors (the public API sometimes
        answers 5xx). Returns None when the API reports an error (so callers
        can decide whether to retry); returns "" when OCR found no text.
        """
        headers = {"apikey": self.api_key}
        data = {
            "language": languages,
            "isOverlayRequired": "true" if overlay else "false",
            "OCREngine": str(engine or self.api_engine),
            # Let OCR.space downscale large photos/scans beyond our control.
            "scale": "true",
            # Detect page orientation from photos taken sideways.
            "detectOrientation": "true",
        }
        files = {"file": (filename, file_bytes)}
        payload = None
        for attempt in range(2):  # 1 immediate retry for transient 5xx/network blips
            try:
                response = httpx.post(
                    self.api_endpoint,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"transient gateway error {response.status_code}",
                        request=response.request, response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as e:
                logger.warning(f"OCR.space request attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    time.sleep(1.5)

        if payload is None:
            logger.error(f"OCR.space request failed permanently: {filename}")
            return None

        errored = bool(payload.get("IsErroredOnProcessing"))
        message = "; ".join(payload.get("ErrorMessages") or []) or payload.get("ErrorMessage")
        if errored or (message and "ParsedResults" not in payload):
            logger.error(f"OCR.space processing error ({languages}): {message or 'unknown error'}")
            return None

        results = payload.get("ParsedResults")
        if results is None:
            logger.error(f"OCR.space returned no ParsedResults ({languages}): {filename}")
            return None
        if overlay:
            words: List[dict] = []
            for result in results:
                for line in ((result.get("TextOverlay") or {}).get("Lines") or []):
                    for word in line.get("Words") or []:
                        text = (word.get("WordText") or "").strip()
                        if not text:
                            continue
                        words.append({
                            "text": text,
                            "left": int(word.get("Left") or 0),
                            "top": int(word.get("Top") or 0),
                            "width": int(word.get("Width") or 0),
                            "height": int(word.get("Height") or 0),
                        })
            return words
        return "\n".join(
            (result.get("ParsedText") or "").strip()
            for result in results
            if result.get("ParsedText")
        ).strip()

    def _ocr_space_extract(self, file_bytes: bytes, filename: str) -> str:
        """
        Cloud extraction with automatic resilience:
        configured engine first, then the alternate engine if that fails,
        then a final plain-English retry. Only triggered on errors/empty text.
        """
        languages = self._api_languages()
        engines = [self.api_engine]
        alternate = 1 if self.api_engine != 1 else 2
        engines.append(alternate)

        for engine in engines:
            text = self._call_ocr_space(file_bytes, filename, languages, engine=engine)
            if text is None:
                continue  # API error on this engine - try the next one
            if text:
                return text
            # Parsed cleanly but no text; give the other engine a chance too.

        if languages != "eng":
            logger.warning(f"OCR.space failed with languages '{languages}', retrying with 'eng'.")
            for engine in engines:
                text = self._call_ocr_space(file_bytes, filename, "eng", engine=engine)
                if text:
                    return text

        return ""

    # ------------------------------------------------------------------
    # Tesseract fallback engine
    # ------------------------------------------------------------------

    def _tesseract_extract(self, file_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image

            with Image.open(file_path) as image:
                try:
                    text = pytesseract.image_to_string(
                        image, lang=self.ocr_languages.replace(",", "+")
                    )
                except Exception:
                    # Requested language data may be missing - retry with English.
                    logger.warning(f"Tesseract OCR with '{self.ocr_languages}' failed, retrying with 'eng'.")
                    text = pytesseract.image_to_string(image, lang="eng")
            logger.info(f"Extracted {len(text)} characters from image: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image {file_path}: {e}")
            return ""

    # ------------------------------------------------------------------
    # Public extraction API (signatures used by DocumentService unchanged)
    # ------------------------------------------------------------------

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from an image file using OCR.

        Args:
            file_path: Path to the image file.

        Returns:
            Extracted text as a string (empty when no engine/text).
        """
        if not self.is_available():
            logger.warning("No OCR engine available, skipping OCR.")
            return ""

        if self.api_key:
            try:
                with open(file_path, "rb") as handle:
                    return self._ocr_space_extract(handle.read(), os.path.basename(file_path))
            except OSError as e:
                logger.error(f"Could not read image {file_path}: {e}")
                return ""

        return self._tesseract_extract(file_path)

    def extract_text_from_image(self, image) -> str:
        """Extract OCR text from an already-open PIL image (PDF page renders)."""
        if not self.is_available():
            return ""
        if self.api_key:
            buffer = io.BytesIO()
            try:
                image.save(buffer, format="PNG")
            except Exception as e:
                logger.error(f"Could not encode rendered image for OCR: {e}")
                return ""
            return self._ocr_space_extract(buffer.getvalue(), "page.png")

        try:
            import pytesseract

            return pytesseract.image_to_string(image, lang=self.ocr_languages.replace(",", "+"))
        except Exception as e:
            logger.error(f"Error extracting text from rendered image: {e}")
            return ""

    def extract_text_with_boxes(self, file_path: str) -> List[dict]:
        """
        Extract text along with bounding box information from an image.

        Uses OCR.space overlay data when a cloud key is configured;
        otherwise falls back to local Tesseract word boxes.

        Returns:
            A list of dicts with 'text', 'left', 'top', 'width', 'height' keys.
        """
        if not self.is_available():
            logger.warning("No OCR engine available, skipping OCR with boxes.")
            return []

        if self.api_key:
            try:
                with open(file_path, "rb") as handle:
                    file_bytes = handle.read()
            except OSError as e:
                logger.error(f"Could not read image {file_path}: {e}")
                return []
            words = None
            engines = [self.api_engine, 1 if self.api_engine != 1 else 2]
            for engine in engines:
                words = self._call_ocr_space(
                    file_bytes, os.path.basename(file_path),
                    self._api_languages(), overlay=True, engine=engine,
                )
                if words is not None:
                    break
            if not words or isinstance(words, str):
                return []
            logger.info(f"Extracted {len(words)} text blocks via OCR.space: {file_path}")
            return words

        results = []
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(file_path)
            data = pytesseract.image_to_data(
                image, lang=self.ocr_languages.replace(",", "+"),
                output_type=pytesseract.Output.DICT,
            )
            for i in range(len(data["text"])):
                if data["text"][i].strip():
                    results.append({
                        "text": data["text"][i],
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                    })
            logger.info(f"Extracted {len(results)} text blocks from image: {file_path}")
        except Exception as e:
            logger.error(f"Error extracting text with boxes from image {file_path}: {e}")
        return results