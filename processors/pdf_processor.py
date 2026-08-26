"""
PDF processor - extracts text from PDF files.
"""

import logging
from typing import List

import PyPDF2

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Service for extracting text content from PDF files."""

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text as a single string.
        """
        text = ""
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    logger.debug(f"Extracted text from page {page_num + 1}")
            logger.info(f"Extracted {len(text)} characters from PDF: {file_path}")
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return text

    def render_pages_for_ocr(self, file_path: str) -> List[object]:
        """Render PDF pages to PIL images when OCR is needed for scanned PDFs."""
        try:
            import fitz  # PyMuPDF
            from PIL import Image

            images = []
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    images.append(Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples))
            return images
        except ImportError:
            logger.warning("PyMuPDF is not installed; scanned-PDF OCR is unavailable.")
            return []
        except Exception as e:
            logger.error(f"Could not render PDF for OCR {file_path}: {e}")
            return []

    def extract_pages(self, file_path: str) -> List[str]:
        """
        Extract text from each page of a PDF as a list.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A list of strings, one per page.
        """
        pages = []
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
            logger.info(f"Extracted {len(pages)} pages from PDF: {file_path}")
        except Exception as e:
            logger.error(f"Error extracting pages from PDF {file_path}: {e}")
        return pages
