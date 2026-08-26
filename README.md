# Telegram AI Document Assistant

A private-chat Telegram bot that stores PDF, DOCX, and image uploads, indexes their text, and lets each user search, retrieve, paginate, or delete their own files.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and supply Telegram, Groq, and MongoDB credentials.
4. Install Tesseract for image/scanned-PDF OCR if needed.
5. Start the bot with `python app.py`.

Document operations are restricted to private chats unless `ALLOW_GROUP_DOCUMENT_ACCESS=true` is explicitly set. Files stay on Telegram's servers; the bot downloads them only into a temporary processing folder, which is removed immediately after indexing. MongoDB retains only file IDs and searchable metadata.

## Deploy on Render

This repository includes `render.yaml` for a free Render web service and Telegram webhooks. OCR is disabled to fit the free instance memory limit; captions, document text extraction, notes, and AI metadata remain searchable. Set `TELEGRAM_BOT_TOKEN`, `MONGODB_URI`, and `GROQ_API_KEY` in Render's environment-variable screen. After the first deploy, set `WEBHOOK_URL` to `https://<your-service>.onrender.com/<your Telegram bot token>` and redeploy. Do not commit `.env` or any credentials.

## Website API

The Render service also exposes an authenticated REST API. Send `X-API-Key: <API_KEY>` with every `/api/v1/...` request. Set a long random `API_KEY` in Render; never place it in browser-side code. Interactive OpenAPI documentation is available at `/docs`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/documents?user_id=...` | List a user's documents |
| `GET` | `/api/v1/documents/{id}?user_id=...` | Get document details and indexed text |
| `POST` | `/api/v1/documents/upload` | Upload a multipart file (`user_id`, optional `description`, `file`) |
| `GET` | `/api/v1/search?user_id=...&q=...` | Search the knowledge vault |
| `POST` | `/api/v1/chat` | Talk to the document assistant |
| `POST` | `/api/v1/documents/{id}/notes` | Add searchable detail to a document |
| `DELETE` | `/api/v1/documents/{id}?user_id=...` | Delete document metadata and indexed knowledge |
