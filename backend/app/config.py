from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False
    )

    app_name: str = "Grounded PDF Q&A"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://pdfrag:pdfrag@postgres:5432/pdfrag"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 30
    bootstrap_admin_email: str = "admin@example.com"
    bootstrap_admin_password: str = "ChangeMe123!"

    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_base_url: str = ""
    llm_timeout_seconds: float = Field(default=45, ge=5, le=300)
    max_output_tokens: int = Field(default=1400, ge=100, le=8000)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_local_files_only: bool = False
    embedding_download_enabled: bool = False
    embedding_fallback_mode: Literal["hashing", "disabled"] = "hashing"
    require_embedding_at_startup: bool = False
    allow_insecure_hf_download: bool = False

    chunk_size_chars: int = Field(default=1200, ge=400, le=4000)
    chunk_overlap_chars: int = Field(default=200, ge=0, le=1000)
    top_k: int = Field(default=8, ge=1, le=20)
    min_similarity: float = Field(default=0.12, ge=0, le=1)
    max_context_chars: int = Field(default=18000, ge=2000, le=100000)
    max_chunks_per_page: int = 3

    query_rewrite_enabled: bool = True
    query_rewrite_max_variants: int = 4
    fuzzy_keyword_enabled: bool = True
    fuzzy_match_cutoff: float = 0.78
    max_query_terms: int = 32

    ocr_mode: Literal["never", "auto", "always"] = "auto"
    ocr_dpi: int = 220
    ocr_languages: str = "eng"
    ocr_min_native_chars: int = 80
    extract_tables: bool = True
    table_min_rows: int = 2

    max_file_size_mb: int = 25
    max_total_upload_mb: int = 200
    max_files_per_collection: int = 20
    max_total_pages: int = 2000
    max_extracted_chars: int = 20_000_000
    max_chunks_per_collection: int = 25_000
    collection_ttl_minutes: int = 120
    max_collections: int = 100
    cors_origins: str = (
        "http://localhost:5173,http://localhost:8080,http://localhost:8081"
    )

    @model_validator(mode="after")
    def validate_related(self) -> "Settings":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError(
                "CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS"
            )
        return self

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_total_upload_bytes(self) -> int:
        return self.max_total_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
