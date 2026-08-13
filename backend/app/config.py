from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

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
    # Use a stronger, pinned answer model for consistent operational-document QA.
    # QUERY_MODEL can remain smaller/cheaper because it only resolves intent and
    # produces retrieval variants; facts still have to come from document chunks.
    llm_model: str = "gpt-5.6-terra"
    query_model: str = "gpt-5.6-luna"
    summary_model: str = "gpt-5.6-luna"
    llm_base_url: str = ""
    llm_timeout_seconds: float = Field(default=60, ge=5, le=300)
    # Rate-limit-aware pacing. OpenAI can enforce TPM in rolling/quantized windows;
    # reserve tokens conservatively, serialize same-model calls in this worker,
    # and honor server reset headers before retrying a large evidence request.
    llm_max_retries: int = Field(default=6, ge=0, le=12)
    llm_retry_base_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    llm_retry_max_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    llm_rate_limit_max_wait_seconds: float = Field(default=75.0, ge=0.0, le=300.0)
    llm_rate_limit_total_wait_seconds: float = Field(default=90.0, ge=0.0, le=600.0)
    llm_rate_limit_safety_seconds: float = Field(default=0.35, ge=0.0, le=5.0)
    llm_rate_limit_safety_tokens: int = Field(default=1500, ge=0, le=20000)
    llm_chars_per_token_estimate: float = Field(default=3.5, ge=2.0, le=8.0)
    llm_proactive_rate_limit_enabled: bool = True
    llm_serialize_model_requests: bool = True
    summary_cache_entries: int = Field(default=384, ge=0, le=4096)
    llm_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "medium"
    query_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"
    summary_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"
    max_output_tokens: int = Field(default=2500, ge=100, le=20000)
    summary_max_output_tokens: int = Field(default=2500, ge=400, le=12000)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_local_files_only: bool = False
    embedding_download_enabled: bool = False
    embedding_fallback_mode: Literal["hashing", "disabled"] = "hashing"
    require_embedding_at_startup: bool = False
    allow_insecure_hf_download: bool = False

    # Metro procedures depend on heading/subheading context. Smaller chunks with
    # larger overlap keep steps readable while the heading-aware splitter carries
    # section path, page range, rolling stock/procedure hints, and tags into every
    # chunk text.
    chunk_size_chars: int = Field(default=900, ge=400, le=4000)
    chunk_overlap_chars: int = Field(default=220, ge=0, le=1000)
    top_k: int = Field(default=12, ge=1, le=200)
    min_similarity: float = Field(default=0.05, ge=0, le=1)
    # This is the direct-prompt ceiling. Larger evidence sets are summarized in
    # batches first, while preserving original source labels for final citations.
    max_context_chars: int = Field(default=300000, ge=10000, le=800000)
    summary_batch_chars: int = Field(default=120000, ge=10000, le=300000)
    max_chunks_per_page: int = 6

    query_rewrite_enabled: bool = True
    query_rewrite_max_variants: int = Field(default=6, ge=1, le=12)
    fuzzy_keyword_enabled: bool = True
    fuzzy_match_cutoff: float = 0.78
    max_query_terms: int = 48

    # A second lexical path uses PostgreSQL's built-in English stemming in addition
    # to the exact/simple FTS path. This catches word-family changes such as
    # obstruct/obstruction/obstructing without weakening acronym/code matching.
    stemmed_search_enabled: bool = True
    stemmed_search_max_chunks: int = Field(default=3000, ge=100, le=20000)

    # Route a question to a clearly matching dedicated SOP/instruction before
    # broad synthesis. Matching uses filename + opening subject text, fuzzy token
    # similarity and literal coverage. The primary document stays dominant while
    # a small number of genuinely relevant supplementary chunks may still be used.
    primary_document_routing_enabled: bool = True
    primary_document_match_threshold: float = Field(default=0.58, ge=0.35, le=0.95)
    primary_document_max_documents: int = Field(default=3, ge=1, le=8)
    primary_document_chunks_per_document: int = Field(default=600, ge=20, le=2500)
    primary_document_supplement_limit: int = Field(default=96, ge=0, le=500)

    # Scenario-body routing is a second stage after normal retrieval. It finds an
    # exact procedure even when the line/scenario wording occurs deep in the PDF
    # rather than in the filename/opening title. Small local windows also inherit
    # line/applicability headings across continuation chunks without re-embedding.
    scenario_document_routing_enabled: bool = True
    scenario_document_max_documents: int = Field(default=3, ge=1, le=8)
    scenario_document_window_chunks: int = Field(default=6, ge=2, le=16)
    applicability_inherit_chunk_window: int = Field(default=8, ge=1, le=24)
    local_anchor_context_window: int = Field(default=2, ge=0, le=5)
    preselection_neighbor_seed_limit: int = Field(default=180, ge=20, le=800)
    preselection_neighbor_window: int = Field(default=2, ge=0, le=4)
    line_alias_scan_chunks: int = Field(default=60, ge=5, le=300)

    # Hybrid retrieval still provides semantic candidates, but answer correctness
    # no longer depends on a tiny top-K. A corpus-wide lexical scan examines every
    # ready chunk and then returns matching rows up to a high safety ceiling.
    retrieval_chunks_per_document: int = Field(default=6, ge=1, le=24)
    max_retrieval_candidates: int = Field(default=3000, ge=50, le=20000)
    corpus_scan_max_chunks: int = Field(default=10000, ge=100, le=50000)
    answer_evidence_chunk_limit: int = Field(default=1200, ge=20, le=3000)
    reference_evidence_chunk_limit: int = Field(default=5000, ge=50, le=10000)
    neighbor_seed_limit: int = Field(default=300, ge=10, le=1000)
    neighbor_window: int = Field(default=1, ge=0, le=3)
    evidence_top_k: int = Field(default=48, ge=4, le=500)

    # When retrieval finds useful related evidence but normal synthesis cannot form
    # a definitive answer, review a bounded strongest-evidence set and produce a
    # clearly qualified best-supported answer rather than hiding the evidence.
    best_supported_answer_enabled: bool = True
    best_supported_source_limit: int = Field(default=64, ge=8, le=200)
    best_supported_candidate_review_limit: int = Field(default=48, ge=8, le=200)

    # Index/catalog rows can identify the exact procedure that should be searched
    # next. Follow those PDF-derived references before answering rather than
    # treating an index row as the final evidence.
    reference_hop_enabled: bool = True
    reference_hop_max_documents: int = Field(default=6, ge=1, le=20)
    reference_hop_chunks_per_document: int = Field(default=400, ge=20, le=2000)

    # Conversation history is intent context only. It is never treated as factual
    # evidence and is never cited in a grounded answer.
    chat_context_messages: int = Field(default=10, ge=0, le=30)
    chat_context_chars: int = Field(default=9000, ge=0, le=30000)
    chat_context_per_message_chars: int = Field(default=1600, ge=200, le=6000)

    # Short internal tokens (SC, PSD, OCC, UTO, etc.) are looked up in ready PDF
    # chunks before the planner is allowed to expand them.
    abbreviation_scan_terms: int = Field(default=8, ge=0, le=20)
    abbreviation_scan_chunks_per_term: int = Field(default=12, ge=1, le=50)

    ocr_mode: Literal["never", "auto", "always"] = "auto"
    ocr_dpi: int = Field(default=300, ge=150, le=450)
    ocr_languages: str = "eng"
    ocr_min_native_chars: int = 80
    ocr_min_text_quality: float = Field(default=0.62, ge=0, le=1)
    ocr_min_native_consensus: float = Field(default=0.42, ge=0, le=1)
    ocr_rotated_text_threshold: float = Field(default=0.08, ge=0, le=1)
    ocr_corruption_threshold: float = Field(default=0.12, ge=0, le=1)
    ocr_verify_all_pages: bool = False
    ocr_verify_dpi: int = Field(default=180, ge=120, le=300)
    ocr_verify_min_novel_terms: int = Field(default=10, ge=3, le=100)
    ocr_verify_novelty_threshold: float = Field(default=0.20, ge=0.05, le=0.8)
    ocr_primary_psm: int = Field(default=3, ge=1, le=13)
    ocr_fallback_psm: int = Field(default=6, ge=1, le=13)
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
    cors_origins: str = "http://localhost:5173,http://localhost:8080,http://localhost:8081"

    @model_validator(mode="after")
    def validate_related(self) -> "Settings":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS")
        if self.summary_batch_chars > self.max_context_chars:
            raise ValueError("SUMMARY_BATCH_CHARS must not exceed MAX_CONTEXT_CHARS")
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
