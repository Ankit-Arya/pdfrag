from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "Grounded PDF Q&A"
    log_level: str = "INFO"

    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_base_url: str = ""
    llm_timeout_seconds: float = Field(default=45.0, ge=5.0, le=300.0)
    max_output_tokens: int = Field(default=1400, ge=100, le=8000)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_chars: int = Field(default=1200, ge=400, le=4000)
    chunk_overlap_chars: int = Field(default=200, ge=0, le=1000)
    top_k: int = Field(default=8, ge=1, le=20)
    min_similarity: float = Field(default=0.12, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=18000, ge=2000, le=100000)
    max_chunks_per_page: int = Field(default=3, ge=1, le=20)

    query_rewrite_enabled: bool = True
    query_rewrite_max_variants: int = Field(default=4, ge=2, le=8)
    fuzzy_keyword_enabled: bool = True
    fuzzy_match_cutoff: float = Field(default=0.78, ge=0.5, le=1.0)
    max_query_terms: int = Field(default=32, ge=4, le=128)

    ocr_mode: Literal["never", "auto", "always"] = "auto"
    ocr_dpi: int = Field(default=220, ge=120, le=400)
    ocr_languages: str = "eng"
    ocr_min_native_chars: int = Field(default=80, ge=0, le=2000)
    extract_tables: bool = True
    table_min_rows: int = Field(default=2, ge=1, le=20)

    max_file_size_mb: int = Field(default=25, ge=1, le=250)
    max_total_upload_mb: int = Field(default=200, ge=1, le=2000)
    max_files_per_collection: int = Field(default=20, ge=1, le=100)
    max_total_pages: int = Field(default=2000, ge=1, le=20000)
    max_extracted_chars: int = Field(default=20_000_000, ge=10_000, le=200_000_000)
    max_chunks_per_collection: int = Field(default=25_000, ge=100, le=250_000)
    collection_ttl_minutes: int = Field(default=120, ge=5, le=1440)
    max_collections: int = Field(default=100, ge=1, le=1000)

    cors_origins: str = "http://localhost:5173,http://localhost:8080"

    @model_validator(mode="after")
    def validate_related_settings(self) -> "Settings":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS")
        if self.max_total_upload_mb < self.max_file_size_mb:
            raise ValueError("MAX_TOTAL_UPLOAD_MB must be at least MAX_FILE_SIZE_MB")
        return self

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_total_upload_bytes(self) -> int:
        return self.max_total_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
