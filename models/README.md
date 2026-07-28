# Optional local embedding model

You may store a complete Sentence Transformers model in this directory and set:

```env
EMBEDDING_MODEL=/models/all-MiniLM-L6-v2
EMBEDDING_LOCAL_FILES_ONLY=1
```

The backend mounts this directory read-only at `/models`.
