from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "Grounded PDF Q&A"
    log_level: str = "INFO"

    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_base_url: str = ""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_chars: int = Field(default=1200, ge=400, le=4000)
    chunk_overlap_chars: int = Field(default=200, ge=0, le=1000)
    top_k: int = Field(default=6, ge=1, le=20)
    min_similarity: float = Field(default=0.20, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=18000, ge=2000, le=100000)

    max_file_size_mb: int = Field(default=25, ge=1, le=250)
    max_files_per_collection: int = Field(default=20, ge=1, le=100)
    max_total_pages: int = Field(default=2000, ge=1, le=20000)
    collection_ttl_minutes: int = Field(default=120, ge=5, le=1440)
    max_collections: int = Field(default=100, ge=1, le=1000)

    cors_origins: str = "http://localhost:5173,http://localhost:8080"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
