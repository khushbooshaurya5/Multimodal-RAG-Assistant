"""FastAPI backend exposing ingestion and multimodal query endpoints.

Run with::

    uvicorn src.api.server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.audio.loader import AudioDecodeError
from src.audio.transcriber import AudioDisabledError
from src.config import get_settings
from src.models.errors import ModelUnavailableError
from src.rag.service import AssistantService
from src.retrieval.filters import RetrievalFilters
from src.schemas import ContentType, ImageAnalysis, RAGResponse, Transcript
from src.utils.logging import configure_logging, get_logger
from src.vision.loader import ImageDecodeError, load_image_bytes

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.service = AssistantService(settings)
    yield


app = FastAPI(
    title="Multimodal RAG Assistant API",
    version="0.1.0",
    description="Text / image / audio retrieval-augmented generation (Qwen-VL + Whisper + FAISS).",
    lifespan=lifespan,
)


def get_service(request: Request) -> AssistantService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "Service not initialised")
    return service


Service = Annotated[AssistantService, Depends(get_service)]


class HealthResponse(BaseModel):
    status: str
    backends: dict[str, Any]
    index: dict[str, Any]


class IngestItem(BaseModel):
    source: str
    doc_id: str | None
    content_type: ContentType | None
    num_chunks: int
    skipped: bool
    error: str | None
    timings_ms: dict[str, float]


@app.get("/health", response_model=HealthResponse)
def health(service: Service) -> HealthResponse:
    return HealthResponse(
        status="ok", backends=service.status().as_dict(), index=service.index_stats()
    )


@app.get("/sources")
def sources(service: Service) -> dict[str, int]:
    return service.store.sources()


@app.delete("/sources/{source}")
def delete_source(source: str, service: Service) -> dict[str, int]:
    return {"removed": service.remove_source(source)}


@app.delete("/index")
def clear_index(service: Service) -> dict[str, str]:
    service.clear_index()
    return {"status": "cleared"}


@app.post("/ingest", response_model=list[IngestItem])
async def ingest(service: Service, files: list[UploadFile] = File(...)) -> list[IngestItem]:
    results: list[IngestItem] = []
    for upload in files:
        data = await upload.read()
        result = service.ingest_upload(upload.filename or "upload", data)
        results.append(
            IngestItem(
                source=result.source,
                doc_id=result.doc_id,
                content_type=result.content_type,
                num_chunks=result.num_chunks,
                skipped=result.skipped,
                error=result.error,
                timings_ms=result.timings_ms,
            )
        )
    return results


@app.post("/transcribe", response_model=Transcript)
async def transcribe(service: Service, audio: UploadFile = File(...)) -> Transcript:
    try:
        return service.transcribe(await audio.read(), audio.filename or "audio.wav")
    except AudioDisabledError as exc:
        raise HTTPException(503, str(exc)) from exc
    except AudioDecodeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/analyze-image", response_model=ImageAnalysis)
async def analyze_image(service: Service, image: UploadFile = File(...)) -> ImageAnalysis:
    try:
        return service.image_processor.analyze_bytes(await image.read(), image.filename or "image")
    except ImageDecodeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/query", response_model=RAGResponse)
async def query(
    service: Service,
    question: str | None = Form(None),
    top_k: int | None = Form(None),
    similarity_threshold: float | None = Form(None),
    content_types: str | None = Form(None, description="JSON list or comma-separated modalities"),
    sources: str | None = Form(None, description="JSON list or comma-separated filenames"),
    index_image: bool = Form(False),
    include_prompt: bool = Form(False),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
) -> RAGResponse:
    pil_image = None
    if image is not None:
        try:
            pil_image = load_image_bytes(await image.read())
        except ImageDecodeError as exc:
            raise HTTPException(400, str(exc)) from exc
    clip = None
    if audio is not None:
        try:
            clip = service.load_audio(await audio.read(), audio.filename or "audio.wav")
        except AudioDecodeError as exc:
            raise HTTPException(400, str(exc)) from exc

    filters = RetrievalFilters(
        content_types=[ContentType(c) for c in _parse_list(content_types)] or None,
        sources=_parse_list(sources) or None,
    )
    try:
        return service.query(
            question,
            image=pil_image,
            image_source=(image.filename if image else "uploaded_image") or "uploaded_image",
            audio=clip,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters,
            index_image=index_image,
            include_prompt=include_prompt,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except AudioDisabledError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc


def _parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    value = value.strip()
    if value.startswith("["):
        try:
            return [str(v) for v in json.loads(value)]
        except json.JSONDecodeError:
            pass
    return [v.strip() for v in value.split(",") if v.strip()]
