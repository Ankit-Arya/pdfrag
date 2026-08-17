import logging
import time
import uuid
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.api import router
from app.auth.router import bootstrap_admin, router as auth_router
from app.config import get_settings
from app.db import SessionLocal, initialize_database
from app.document_processing import recover_interrupted_processing
from app.rag.embeddings import embedding_service

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    with SessionLocal() as db:
        bootstrap_admin(db)

    recover_interrupted_processing()

    logger.info("Loading embedding model: %s", settings.embedding_model)
    embedding_ready = embedding_service.warmup()
    if embedding_ready:
        logger.info("Embedding backend ready: %s", embedding_service.backend)
        if embedding_service.using_fallback:
            logger.warning(
                "The local hashing embedding fallback is active. Ingestion and Q&A "
                "are available, but reprocess documents after switching to a "
                "transformer model."
            )
    elif settings.require_embedding_at_startup:
        raise RuntimeError(
            embedding_service.last_error or "Embedding model warmup failed"
        )
    else:
        logger.error(
            "Backend started in degraded mode because the embedding model is not "
            "available. Health remains reachable; document processing and chat "
            "will return HTTP 503 until the model can be loaded."
        )

    yield


class CustomORJSONResponse(ORJSONResponse):
    def render(self, content: object) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS)


app = FastAPI(
    title=settings.app_name,
    version="2.0.2",
    default_response_class=CustomORJSONResponse,
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
    request.state.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-Ms"] = (
        f"{(time.perf_counter() - started) * 1000:.1f}"
    )
    return response


app.include_router(auth_router)
app.include_router(router)
