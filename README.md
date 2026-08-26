# Telegram AI Document Assistant

A private-chat Telegram bot that stores PDF, DOCX, and image uploads, indexes their text, and lets each user search, retrieve, paginate, or delete their own files. It also ships a **J.A.R.V.I.S.-style web dashboard** for monitoring your vault and chatting with the AI.

## Features

- **Document vault** — upload PDFs / DOCX / images; files stay on Telegram's servers (the bot only downloads them into a temp folder during processing, then deletes them). MongoDB keeps only file IDs + searchable metadata.
- **AI metadata** — every upload gets an AI-generated title, category, tags, search aliases (incl. Hinglish/typo variants), entities, and an expiry date.
- **Search** — keyword text search over indexed chunks (local embeddings are supported but disabled by default to keep memory low).
- **Chat** — ask questions about your documents; the bot retrieves relevant excerpts and answers via Groq.
- **Expiry reminders** — daily notifications for documents with an expiry date.
- **JARVIS dashboard** — a themed web UI at `/dashboard` (stats, document archive, semantic scan, chat console, upload bay).

## Setup (local)

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and supply your Telegram, Groq, and MongoDB credentials.
4. (Optional) Install Tesseract if you want OCR for images/scanned PDFs — set `ENABLE_OCR=true` in `.env`. OCR is **off by default** to keep the footprint small.
5. Start the server:
   ```bash
   python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
   ```
   Then open **http://localhost:8000/dashboard**.

   To run the Telegram bot directly in polling mode instead:
   ```bash
   python app.py
   ```

## JARVIS Dashboard

The dashboard is served from the same origin as the API (no CORS issues) at `/dashboard`, and `/` redirects there.

- **Connection settings** — click the ⚙ gear in the top-right to enter your `X-API-Key` and Telegram `user_id`. These are stored in your browser's `localStorage` only.
- **Vault statistics** — total files, documents vs images, expiring-soon count, category matrix, and live diagnostics (uptime, latency, search mode).
- **Communication console** — chat with the AI about your documents; the arc reactor pulses amber while thinking.
- **Document archive** — browse, quick-filter, or press `⌕ SCAN` for a semantic vault search. Click any card to view metadata, indexed text, and add a searchable note.
- **Upload bay** — drag & drop a PDF/DOCX/image, add a description, and watch it get indexed.

## API

The REST API is authenticated with the `X-API-Key` header. Interactive docs are at `/docs`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/documents?user_id=...` | List a user's documents |
| `GET` | `/api/v1/documents/{id}?user_id=...` | Get document details and indexed text |
| `POST` | `/api/v1/documents/upload` | Upload a multipart file (`user_id`, optional `description`, `file`) |
| `GET` | `/api/v1/search?user_id=...&q=...` | Search the knowledge vault |
| `POST` | `/api/v1/chat` | Talk to the document assistant |
| `POST` | `/api/v1/documents/{id}/notes` | Add searchable detail to a document |
| `DELETE` | `/api/v1/documents/{id}?user_id=...` | Delete document metadata and indexed knowledge |

## Configuration

Key settings in `.env` / `.env.example`:

| Variable | Default | Notes |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | — | Required |
| `GROQ_API_KEY` | — | Required |
| `MONGODB_URI` | `mongodb://localhost:27017` | |
| `POLLING_MODE` | `true` | `false` + `WEBHOOK_URL` for webhook mode |
| `ENABLE_OCR` | `false` | Tesseract OCR for images/scanned PDFs |
| `ENABLE_LOCAL_EMBEDDINGS` | `false` | Loads the local sentence-transformer model (needs RAM) |
| `API_KEY` | — | Required in production; used to auth the REST API |

## Notes

- Document operations are restricted to private chats unless `ALLOW_GROUP_DOCUMENT_ACCESS=true`.
- `ENABLE_LOCAL_EMBEDDINGS=false` keeps memory usage low (no PyTorch/torch loaded); search falls back to MongoDB text search. Set it to `true` on a machine with sufficient RAM to enable semantic vector search.
- Do not commit `.env` or any credentials.
