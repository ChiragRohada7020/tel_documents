"""
Image processor - extracts text from images using OCR (Tesseract).
OCR is optional; captions/descriptions can be used as searchable text.
"""

import logging
import os
import re
import shutil
from typing import List

from config import Config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Service for extracting text from images using OCR."""

    # Common Tesseract install locations on Windows
    _TESSERACT_PATHS = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    )

    # Map common 2-letter codes to Tesseract's 3-letter language codes
    _LANG_MAP = {
        "en": "eng", "hi": "hin", "mr": "mar", "ta": "tam", "te": "tel",
        "kn": "kan", "gu": "guj", "bn": "ben", "pa": "pan", "ur": "urd",
        "sa": "san", "ne": "nep", "si": "sin",
    }

    def __init__(self) -> None:
        self.ocr_languages = self._normalize_languages(Config.OCR_LANGUAGES)
        self._tesseract_available = self._check_tesseract()

    def _normalize_languages(self, raw: str) -> str:
        """
        Convert configured languages (e.g. 'en,hi,mr') into Tesseract format
        (e.g. 'eng+hin+mar'). Falls back to 'eng' when nothing valid remains.
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
        """Check if Tesseract OCR is available on the system."""
        try:
            import pytesseract

            if not self._locate_tesseract():
                logger.warning(
                    "Tesseract binary not found on PATH or default install locations."
                )
                return False
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available.")
            return True
        except Exception as e:
            logger.warning(f"Tesseract OCR not available: {e}")
            return False

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from an image file using Tesseract OCR.
        Returns empty string if Tesseract is not available.

        Args:
            file_path: Path to the image file.

        Returns:
            Extracted text as a string.
        """
        if not self._tesseract_available:
            logger.warning("Tesseract not available, skipping OCR.")
            return ""

        try:
            import pytesseract
            from PIL import Image

            with Image.open(file_path) as image:
                try:
                    text = pytesseract.image_to_string(image, lang=self.ocr_languages)
                except Exception:
                    # Requested language data may be missing - retry with English.
                    logger.warning(
                        f"OCR with languages '{self.ocr_languages}' failed, retrying with 'eng'."
                    )
                    text = pytesseract.image_to_string(image, lang="eng")
            logger.info(f"Extracted {len(text)} characters from image: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image {file_path}: {e}")
            return ""

    def extract_text_from_image(self, image) -> str:
        """Extract OCR text from an already-open PIL image."""
        if not self._tesseract_available:
            return ""
        try:
            import pytesseract
            return pytesseract.image_to_string(image, lang=self.ocr_languages)
        except Exception as e:
            logger.error(f"Error extracting text from rendered image: {e}")
            return ""

    def extract_text_with_boxes(self, file_path: str) -> List[dict]:
        """
        Extract text along with bounding box information from an image.

        Args:
            file_path: Path to the image file.

        Returns:
            A list of dicts with 'text', 'left', 'top', 'width', 'height' keys.
        """
        if not self._tesseract_available:
            logger.warning("Tesseract not available, skipping OCR with boxes.")
            return []

        results = []
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(file_path)
            data = pytesseract.image_to_data(
                image, lang=self.ocr_languages, output_type=pytesseract.Output.DICT
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

    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self._tesseract_available
