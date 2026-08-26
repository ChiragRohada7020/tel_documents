"""Probe: inspect the RAW completion returned by Groq for a metadata request."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from services.groq_service import GroqService  # noqa: E402

SAMPLE_TEXT = """MSEB ELECTRICITY BILL
Consumer Name: Chirag Rohada
Bill No: 2026-08/77341
Billing Month: August 2026
Amount Due: INR 2450
Due Date: 12/09/2026"""


async def main() -> None:
    svc = GroqService()
    print(f"MODEL={Config.GROQ_MODEL} TIMEOUT={Config.METADATA_TIMEOUT_SECONDS}s")
    try:
        resp = await svc.client.chat.completions.create(
            model=svc.model,
            messages=[{"role": "user", "content": "Return ONLY compact JSON {\"title\":\"X\",\"description\":\"Y\"}"}],
            temperature=0.3,
            max_tokens=400,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        print(f"finish_reason={resp.choices[0].finish_reason}")
        print(f"content_len={len(content)}")
        print("RAW>>>", content[:600])
        # Some models (openai/gpt-oss) put text in reasoning_content when cut off.
        extra = getattr(msg, "reasoning_content", "") or ""
        if extra:
            print("REASONING>>>", str(extra)[:300])
    except Exception as e:
        print(f"PROBE ERROR: {type(e).__name__}: {e}")
    finally:
        try:
            await svc.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())