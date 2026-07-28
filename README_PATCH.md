# PostgreSQL + JWT persistent knowledge patch — v4

This ZIP is an overlay for `Ankit-Arya/pdfrag`. Copy the folder contents over the
repository root and allow existing files to be replaced.

## Included application changes

- PostgreSQL-backed users and roles.
- JWT access tokens plus rotating, revocable refresh-token sessions.
- Admin-only PDF upload, processing, listing, reprocessing, and deletion.
- Original PDF bytes, chunks, embeddings, and metadata stored in PostgreSQL.
- pgvector similarity combined with PostgreSQL full-text search.
- Persistent per-user chat sessions and messages.
- Logged-in users query the administrator's prepared knowledge base without a
  temporary `collection_id`.

## v4 embedding fix

Your Docker build and all containers are now healthy, but the network returns
HTTP 403 for Hugging Face model files. v4 removes that external dependency from
the default runtime path:

1. Docker model preloading is disabled by default, avoiding blocked network calls.
2. The backend first tries a mounted or cached SentenceTransformer model.
3. If no transformer model is available, it automatically uses a deterministic,
   normalized 384-dimensional local hashing embedding backend.
4. Document ingestion, pgvector storage, hybrid retrieval, and Q&A remain enabled.
5. `/api/health` reports `embedding_backend`, `embedding_fallback`, and the reason
   the transformer model was unavailable.
6. No TLS verification bypass is required for the fallback.

The local hashing backend is lexical rather than fully semantic. PostgreSQL
full-text retrieval remains active, so it is a reliable offline baseline. For
best semantic quality, mount a full SentenceTransformer model later and reprocess
all documents.

## Recommended `.env` values for the current network

```env
PRELOAD_EMBEDDING_MODEL=0
EMBEDDING_DOWNLOAD_ENABLED=0
EMBEDDING_FALLBACK_MODE=hashing
ALLOW_INSECURE_HF_DOWNLOAD=0
```

Remove the previous development TLS bypass. A `403 Forbidden` is an authorization
or network-policy response, so disabling TLS verification does not solve it.

## Clean rebuild

```powershell
docker compose down --remove-orphans
docker compose build --no-cache backend
docker compose up -d
docker compose logs -f backend
```

Verify:

```powershell
curl.exe http://localhost:8081/api/health
```

Expected functional offline response:

```json
{
  "status": "ok",
  "embedding_ready": true,
  "embedding_backend": "local-hashing",
  "embedding_fallback": true
}
```

Open Swagger at `http://localhost:8081/api/docs`.

## Typical API flow

1. `POST /api/auth/login` using the bootstrap administrator credentials.
2. Copy `access_token` into Swagger's **Authorize** dialog.
3. `POST /api/admin/documents?process=true` and upload a PDF.
4. A logged-in user calls `POST /api/chat` with `{ "question": "..." }`.
5. Continue a conversation using the returned `chat_session_id`.

## Switching to a transformer model later

Either place a complete model under `./models`, for example:

```env
EMBEDDING_MODEL=/models/all-MiniLM-L6-v2
EMBEDDING_LOCAL_FILES_ONLY=1
EMBEDDING_FALLBACK_MODE=disabled
```

or provide a trusted organization CA/Hugging Face access and enable downloads:

```env
EMBEDDING_DOWNLOAD_ENABLED=1
EMBEDDING_FALLBACK_MODE=hashing
```

After changing embedding backends, use the admin reprocess endpoint for every
existing document. Vectors created by different embedding backends must not be
mixed.

## Notes

- Keep `EMBEDDING_DIMENSIONS=384` unless both the database schema and selected
  model are intentionally migrated together.
- Change all sample secrets before exposing the service.
- The current frontend is not replaced. It still needs login/token handling and
  removal of its temporary `collection_id` workflow.
