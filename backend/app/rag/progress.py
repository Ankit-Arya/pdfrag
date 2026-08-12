from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypedDict


class ProgressPayload(TypedDict, total=False):
    stage: str
    label: str
    detail: str
    current: int
    total: int
    timestamp: float


ProgressCallback = Callable[[ProgressPayload], None]
_LOCAL = threading.local()


@contextmanager
def progress_context(callback: ProgressCallback | None) -> Iterator[None]:
    """Attach a progress reporter to the current RAG worker thread only.

    The callback carries operational status (retrieval, routing, summarization,
    validation) and never model chain-of-thought or hidden reasoning.
    """
    previous = getattr(_LOCAL, "callback", None)
    _LOCAL.callback = callback
    try:
        yield
    finally:
        _LOCAL.callback = previous


def emit_progress(
    stage: str,
    label: str,
    detail: str = "",
    *,
    current: int | None = None,
    total: int | None = None,
) -> None:
    callback: ProgressCallback | None = getattr(_LOCAL, "callback", None)
    if callback is None:
        return
    payload: ProgressPayload = {
        "stage": stage,
        "label": label,
        "timestamp": time.time(),
    }
    if detail:
        payload["detail"] = detail
    if current is not None:
        payload["current"] = max(0, int(current))
    if total is not None:
        payload["total"] = max(0, int(total))
    try:
        callback(payload)
    except Exception:
        # Progress must never be able to break retrieval or answer generation.
        return
