"""Live check of AI metadata quality against the real Groq API.

Run:  python scripts/metadata_live_test.py
Verifies that uploads get specific, human-readable titles and that every
key detail (bill numbers, amounts, dates) lands in tags/entities.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.groq_service import GroqService  # noqa: E402

SAMPLE_OCR_TEXT = """OCR text from image:
MSEB ELECTRICITY BILL
Consumer Name: Chirag Rohada
Bill No: 2026-08/77341
Billing Month: August 2026
Units Consumed: 312 kWh
Amount Due: INR 2450
Due Date: 12/09/2026
Status: PAYMENT PENDING"""


async def main() -> int:
    service = GroqService()
    metadata = await service.generate_metadata("", SAMPLE_OCR_TEXT)
    if not metadata:
        print("FAIL: generate_metadata returned None")
        return 1

    title = (metadata.get("title") or "").strip()
    print(f"TITLE     : {title}")
    print(f"DOC TYPE  : {metadata.get('document_type')}")
    print(f"CATEGORY  : {metadata.get('category')}")
    print(f"TAGS      : {', '.join((metadata.get('tags') or [])[:12])}")
    entities = metadata.get("entities") or {}
    print(f"PEOPLE    : {entities.get('people')}")
    print(f"ORGANIZ.  : {entities.get('organizations')}")
    print(f"NUMBERS   : {entities.get('important_numbers')}")

    failures = []
    if not title:
        failures.append("empty title")
    elif title.lower() in {"image", "photo", "document"}:
        failures.append(f"generic title: {title!r}")

    combined = str(metadata).lower()
    for needle in ["77341", "2450"]:
        if needle not in combined:
            failures.append(f"detail lost: {needle!r} not captured anywhere")

    # Close the underlying HTTP client so the process exits promptly.
    try:
        await service.client.close()
    except Exception:
        pass

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f" - {f}")
        return 1
    print("\nPASS: specific title + key details captured.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))