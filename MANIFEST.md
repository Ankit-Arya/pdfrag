# Deliverables

- `notebooks/pdf_rag_end_to_end.ipynb` — end-to-end multi-PDF RAG notebook.
- `backend/` — FastAPI API, PDF processing, embeddings, FAISS store, strict prompts, guardrails, tests, and Dockerfile.
- `frontend/` — Vue 3 + TypeScript application and production static-server Dockerfile.
- `nginx/nginx.conf` — edge reverse proxy, upload/time-out controls, request IDs, compression, and security headers.
- `docker-compose.yml` — three services: frontend, backend, and nginx.
- `.env.example` — all runtime configuration.
- `README.md` — setup, architecture, API, limitations, and production guidance.
- `.github/workflows/ci.yml` — backend lint/tests and frontend production build.
