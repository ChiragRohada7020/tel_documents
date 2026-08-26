"""
Image composer - combines multiple images into a single multi-page PDF.
Used when a user sends a media-group (album) of photos.
"""

import logging
from typing import List

from PIL import Image
# Importing JpegImagePlugin registers the "JPEG" save handler that Pillow's
# PDF encoder relies on for RGB images (otherwise Image.SAVE["JPEG"] is
# missing and saving a multi-page PDF raises KeyError: 'JPEG').
from PIL import JpegImagePlugin  # noqa: F401

logger = logging.getLogger(__name__)


class ImageComposer:
    """Combine a list of image files into one PDF (one page per image)."""

    def combine_to_pdf(self, image_paths: List[str], output_path: str) -> str:
        """
        Combine multiple images into a single PDF.

        Args:
            image_paths: Ordered list of image file paths.
            output_path: Where to write the resulting PDF.

        Returns:
            The output PDF path.

        Raises:
            ValueError: if no valid images could be opened.
        """
        images: List[Image.Image] = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    # PDF output does not support alpha / palette modes.
                    if img.mode in ("RGBA", "LA", "P"):
                        img = img.convert("RGB")
                    # Copy so the underlying file handle can be released
                    # immediately (avoids lock errors on Windows temp dirs).
                    images.append(img.copy())
            except Exception as e:
                logger.warning(f"Could not open image {path} for PDF composition: {e}")

        if not images:
            raise ValueError("No valid images to combine into a PDF")

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            resolution=150.0,
        )
        logger.info(f"Combined {len(images)} images into PDF: {output_path}")
        return output_path
