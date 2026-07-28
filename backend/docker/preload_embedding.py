"""Optional embedding model preload used during the Docker build."""

from __future__ import annotations

import os
import sys


def _enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _allow_insecure_http() -> None:
    if not _enabled("ALLOW_INSECURE_HF_DOWNLOAD"):
        return

    import httpx
    import huggingface_hub

    setter = getattr(huggingface_hub, "set_client_factory", None)
    if setter is None:
        raise RuntimeError(
            "Installed huggingface_hub does not support the HTTP client factory "
            "required by ALLOW_INSECURE_HF_DOWNLOAD."
        )

    print(
        "WARNING: TLS verification is disabled only for the Hugging Face model "
        "preload. Do not use this setting in production.",
        file=sys.stderr,
    )
    setter(
        lambda: httpx.Client(
            verify=False,
            follow_redirects=True,
            timeout=httpx.Timeout(120.0),
        )
    )


def main() -> int:
    required = _enabled("REQUIRE_EMBEDDING_PRELOAD")
    if not _enabled("PRELOAD_EMBEDDING_MODEL") and not required:
        print(
            "Skipping Hugging Face model preload. Runtime will use a cached/local "
            "model when available or the built-in local hashing fallback."
        )
        return 0

    from sentence_transformers import SentenceTransformer

    model_name = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    try:
        _allow_insecure_http()
        SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        message = f"Embedding model preload failed: {type(exc).__name__}: {exc}"
        if required:
            print(message, file=sys.stderr)
            return 1
        print(f"WARNING: {message}", file=sys.stderr)
        print(
            "The image will still build and use the local hashing fallback.",
            file=sys.stderr,
        )
        return 0

    print(f"Embedding model cached successfully: {model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
