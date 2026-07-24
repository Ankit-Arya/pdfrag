import logging
from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.models import (
    AnswerResponse,
    CollectionInfo,
    CollectionResponse,
    HealthResponse,
    QuestionRequest,
)
from app.rag.llm import LlmConfigurationError
from app.rag.pdf import (
    PdfProcessingError,
    ocr_available,
    safe_filename,
    table_extraction_available,
)
from app.rag.service import rag_service
from app.rag.store import CollectionNotFoundError, collection_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        ocr_mode=settings.ocr_mode,
        ocr_available=ocr_available(),
        table_extraction=settings.extract_tables,
        table_extraction_available=table_extraction_available(),
        query_rewrite=settings.query_rewrite_enabled,
    )


@router.post(
    "/collections",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_collection(
    files: Annotated[list[UploadFile], File(description="One or more PDF files")],
) -> CollectionResponse:
    settings = get_settings()
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")
    if len(files) > settings.max_files_per_collection:
        raise HTTPException(
            status_code=413,
            detail=f"At most {settings.max_files_per_collection} PDFs may be uploaded at once.",
        )

    uploaded: list[tuple[str, bytes]] = []
    used_names: set[str] = set()
    total_upload_bytes = 0
    for file in files:
        filename = _unique_filename(safe_filename(file.filename), used_names)
        used_names.add(filename)
        if not filename.lower().endswith(".pdf"):
            await file.close()
            raise HTTPException(status_code=415, detail=f"{filename}: only PDF files are accepted.")

        data = await file.read(settings.max_file_size_bytes + 1)
        await file.close()
        if len(data) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{filename}: exceeds the {settings.max_file_size_mb} MB limit.",
            )
        total_upload_bytes += len(data)
        if total_upload_bytes > settings.max_total_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"The combined upload exceeds the {settings.max_total_upload_mb} MB limit."
                ),
            )
        uploaded.append((filename, data))

    try:
        collection = await run_in_threadpool(rag_service.build_collection, uploaded)
    except (PdfProcessingError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return CollectionResponse(
        collection_id=collection.collection_id,
        files=collection.files,
        total_pages=collection.total_pages,
        total_chunks=len(collection.chunks),
        expires_in_minutes=settings.collection_ttl_minutes,
        warnings=collection.warnings,
    )


@router.get("/collections/{collection_id}", response_model=CollectionInfo)
def get_collection(collection_id: str) -> CollectionInfo:
    settings = get_settings()
    try:
        collection = collection_store.get(collection_id)
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Collection not found or expired.") from exc
    return CollectionInfo(
        collection_id=collection.collection_id,
        files=collection.files,
        total_pages=collection.total_pages,
        total_chunks=len(collection.chunks),
        expires_in_minutes=settings.collection_ttl_minutes,
        created_at=collection.created_at_iso,
        last_accessed_at=collection.last_accessed_at_iso,
        warnings=collection.warnings,
    )


@router.delete("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: str) -> None:
    if not collection_store.delete(collection_id):
        raise HTTPException(status_code=404, detail="Collection not found or expired.")


@router.post("/chat", response_model=AnswerResponse)
async def chat(payload: QuestionRequest, request: Request) -> AnswerResponse:
    try:
        response = await run_in_threadpool(
            rag_service.ask,
            payload.collection_id,
            payload.question,
            payload.top_k,
            payload.rewrite_question,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Collection not found or expired.") from exc
    except LlmConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="The language model request timed out.") from exc
    except Exception:
        logger.exception("Q&A request failed")
        raise HTTPException(status_code=502, detail="The Q&A dependency request failed.") from None

    response.request_id = getattr(request.state, "request_id", None)
    return response


def _unique_filename(filename: str, used_names: set[str]) -> str:
    if filename not in used_names:
        return filename
    path = PurePath(filename)
    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = f"{stem} ({counter}){suffix}"
        if candidate not in used_names:
            return candidate
        counter += 1
