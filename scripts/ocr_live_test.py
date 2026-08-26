"""One-off live verification of OCR.space integration (not part of CI)."""
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

from PIL import Image, ImageDraw

from config import Config
from processors.image_processor import ImageProcessor

print("OCR_SPACE_API_KEY configured:", bool(Config.OCR_SPACE_API_KEY))
proc = ImageProcessor()
print("engine:", "ocr.space" if proc.api_key else "tesseract",
      "| available:", proc.is_available(),
      "| languages:", proc.ocr_languages, "->", proc._api_languages())

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "sample.png")
    img = Image.new("RGB", (600, 200), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 40), "Hello Telegram Vault 12345", fill="black")
    d.text((30, 90), "Invoice Total: INR 999", fill="black")
    img.save(path)

    text = proc.extract_text(path)
    print("--- extract_text result ---")
    print(repr(text[:300]))

    boxes = proc.extract_text_with_boxes(path)
    print(f"--- boxes: {len(boxes)} words ---")
    if boxes:
        print("first word:", repr(boxes[0]["text"]), "at", boxes[0]["left"], boxes[0]["top"])

    text2 = proc.extract_text_from_image(img)
    print("extract_text_from_image (PDF-render path) chars:", len(text2))

blob = ((text or "") + " " + "".join(b["text"] for b in boxes) + " " + (text2 or "")).upper()
ok = "HELLO" in blob and ("INVOICE" in blob or "999" in blob)
print("LIVE TEST", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)