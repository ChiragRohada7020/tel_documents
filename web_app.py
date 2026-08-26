"""Authenticated REST API and Telegram webhook for the document vault."""

import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from telegram import Update
from telegram.ext import Application

from bot.handlers import setup_handlers
from config import Config
from database.indexes import create_indexes
from database.mongo import close_db, init_db
from services.conversation_service import ConversationService
from services.document_service import DocumentService
from services.groq_service import GroqService
from services.search_service import SearchService


@lru_cache(maxsize=1)
def documents() -> DocumentService:
    return DocumentService()


@lru_cache(maxsize=1)
def searcher() -> SearchService:
    return SearchService()


@lru_cache(maxsize=1)
def groq() -> GroqService:
    return GroqService()


@lru_cache(maxsize=1)
def conversations() -> ConversationService:
    return ConversationService()


telegram_app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Config.validate()
    init_db(Config.MONGODB_URI, Config.MONGODB_DATABASE)
    create_indexes()
    setup_handlers(telegram_app)
    await telegram_app.initialize()
    await telegram_app.start()
    if Config.WEBHOOK_URL:
        await telegram_app.bot.set_webhook(url=Config.WEBHOOK_URL)
    try:
        yield
    finally:
        await telegram_app.stop()
        await telegram_app.shutdown()
        close_db()


app = FastAPI(title="Telegram AI Document Vault API", version="1.0.0", lifespan=lifespan)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not Config.API_KEY or x_api_key != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ChatRequest(BaseModel):
    user_id: int
    message: str = Field(min_length=1, max_length=4000)


class NoteRequest(BaseModel):
    user_id: int
    detail: str = Field(min_length=1, max_length=4000)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def root():
    return {"status": "ok"}


@app.post("/{token}", include_in_schema=False)
async def telegram_webhook(token: str, payload: dict):
    if token != Config.TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=404, detail="Not found")
    update = Update.de_json(payload, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}


@app.get("/api/v1/documents", dependencies=[Depends(require_api_key)])
async def list_documents(user_id: int, page: int = 0, limit: int = 20, category: str = ""):
    limit = max(1, min(limit, 50))
    page = max(0, page)
    service = documents()
    return {
        "items": service.list_user_files(user_id, skip=page * limit, limit=limit, category=category or None),
        "total": service.count_user_files(user_id, category=category or None),
    }


@app.get("/api/v1/documents/{document_id}", dependencies=[Depends(require_api_key)])
async def get_document(document_id: str, user_id: int):
    item = documents().get_file_by_id(user_id, document_id)
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")
    item["text"] = documents().get_document_text(user_id, item["filename"])
    return item


@app.get("/api/v1/search", dependencies=[Depends(require_api_key)])
async def search_documents(user_id: int, q: str, limit: int = 5):
    return {"items": await asyncio.to_thread(searcher().search, q, user_id, max(1, min(limit, 10)))}


@app.post("/api/v1/chat", dependencies=[Depends(require_api_key)])
async def chat_with_vault(request: ChatRequest):
    history = conversations().get_history(request.user_id)
    results = await asyncio.to_thread(searcher().search, request.message, request.user_id)
    sources = [str(item.get("source", "")) for item in results if item.get("source")]
    related = documents().related_document_context(request.user_id, sources)
    parts = [f"[From {item.get('source', 'document')}]\n{item.get('content', '')}" for item in results[:3]]
    parts.extend(f"[Related: {item['source']}]\n{item['content']}" for item in related)
    answer = await groq().chat(request.message, history, "\n\n".join(parts))
    conversations().add_message(request.user_id, "user", request.message)
    conversations().add_message(request.user_id, "assistant", answer)
    return {"answer": answer, "sources": sources}


@app.post("/api/v1/documents/{document_id}/notes", dependencies=[Depends(require_api_key)])
async def add_note(document_id: str, request: NoteRequest):
    item = await asyncio.to_thread(documents().append_user_note, request.user_id, document_id, request.detail)
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True, "document": item}


@app.delete("/api/v1/documents/{document_id}", dependencies=[Depends(require_api_key)])
async def delete_document(document_id: str, user_id: int):
    item = documents().delete_file_by_id(user_id, document_id)
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True}


@app.post("/api/v1/documents/upload", dependencies=[Depends(require_api_key)])
async def upload_document(user_id: int, description: str = "", file: UploadFile = File(...)):
    """Upload through the API; the bot stores the resulting file on Telegram."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds the configured size limit")
    from io import BytesIO
    stream = BytesIO(content)
    stream.name = file.filename or "document"
    message = await telegram_app.bot.send_document(chat_id=user_id, document=stream, caption=description)
    document = message.document
    if not document:
        raise HTTPException(status_code=422, detail="Telegram did not accept the uploaded file")
    if (document.mime_type or "").startswith("image/"):
        result = await documents().process_image(document, user_id, None, caption=description, original_name=document.file_name or "image.jpg")
    else:
        result = await documents().process_document(document, user_id, None, caption=description)
    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error", "Document processing failed"))
    return result
