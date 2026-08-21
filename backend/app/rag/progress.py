# IMS_LIVE_ACTIVITY_V1
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, TypedDict


class ProgressPayload(TypedDict, total=False):
    stage: str
    label: str
    detail: str
    current: int
    total: int
    timestamp: float
    actor: str
    phase: str
    status: str
    operation_id: str
    sequence: int
    duration_ms: int
    total_elapsed_ms: int
    prompt_summary: str
    reasoning_summary: str
    document: str
    page: int
    heading: str
    metrics: dict[str, object]


ProgressCallback = Callable[[ProgressPayload], None]
_LOCAL = threading.local()


@dataclass(slots=True)
class _ProgressState:
    callback: ProgressCallback | None
    started_monotonic: float = field(default_factory=time.monotonic)
    sequence: int = 0
    operation_started: dict[str, float] = field(default_factory=dict)


def _safe_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split())
    return text[:limit]


def _infer_actor(stage: str) -> str:
    folded = stage.casefold()
    if any(token in folded for token in ("verify", "review", "citation", "ground", "evidence_check")):
        return "verification"
    if any(token in folded for token in ("search", "route", "heading", "section", "retrieval", "corpus", "reference")):
        return "search"
    if any(token in folded for token in ("interpret", "answer", "rerank", "rewrite", "reason")):
        return "ai"
    return "backend"


def _infer_phase(stage: str, actor: str) -> str:
    folded = stage.casefold()
    if "interpret" in folded:
        return "interpret"
    if "route" in folded:
        return "route"
    if any(token in folded for token in ("search", "retrieval", "heading", "section", "corpus", "reference")):
        return "search"
    if "rerank" in folded:
        return "rerank"
    if any(token in folded for token in ("review", "evidence_check")):
        return "review"
    if "answer" in folded:
        return "answer"
    if any(token in folded for token in ("verify", "citation", "ground")):
        return "verify"
    return actor


@contextmanager
def progress_context(callback: ProgressCallback | None) -> Iterator[None]:
    """Attach a progress reporter to the current RAG worker thread only.

    The callback exposes operational activity, safe AI-task summaries, counts and
    source-navigation metadata. It intentionally does not expose hidden chain-of-thought,
    raw system/developer prompts, credentials, SQL text or private scratch work.
    """
    previous = getattr(_LOCAL, "state", None)
    _LOCAL.state = _ProgressState(callback=callback)
    try:
        yield
    finally:
        _LOCAL.state = previous


def emit_progress(
    stage: str,
    label: str,
    detail: str = "",
    *,
    current: int | None = None,
    total: int | None = None,
    actor: str | None = None,
    phase: str | None = None,
    status: str | None = None,
    operation_id: str | None = None,
    prompt_summary: str | None = None,
    reasoning_summary: str | None = None,
    document: str | None = None,
    page: int | None = None,
    heading: str | None = None,
    metrics: dict[str, object] | None = None,
) -> None:
    state: _ProgressState | None = getattr(_LOCAL, "state", None)
    if state is None or state.callback is None:
        return

    now_monotonic = time.monotonic()
    actor_value = _safe_text(actor or _infer_actor(stage), 24).casefold() or "backend"
    phase_value = _safe_text(phase or _infer_phase(stage, actor_value), 40).casefold() or actor_value
    status_value = _safe_text(status or "running", 24).casefold() or "running"
    operation_value = _safe_text(operation_id or stage, 120) or stage

    if status_value in {"running", "started"}:
        state.operation_started.setdefault(operation_value, now_monotonic)
    started = state.operation_started.get(operation_value)
    duration_ms: int | None = None
    if status_value in {"complete", "completed", "warning", "error", "failed"}:
        if started is not None:
            duration_ms = max(0, int((now_monotonic - started) * 1000))
            state.operation_started.pop(operation_value, None)
        else:
            duration_ms = 0

    state.sequence += 1
    payload: ProgressPayload = {
        "stage": _safe_text(stage, 120),
        "label": _safe_text(label, 240),
        "timestamp": time.time(),
        "actor": actor_value,
        "phase": phase_value,
        "status": status_value,
        "operation_id": operation_value,
        "sequence": state.sequence,
        "total_elapsed_ms": max(0, int((now_monotonic - state.started_monotonic) * 1000)),
    }
    if detail:
        payload["detail"] = _safe_text(detail, 1200)
    if current is not None:
        payload["current"] = max(0, int(current))
    if total is not None:
        payload["total"] = max(0, int(total))
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if prompt_summary:
        payload["prompt_summary"] = _safe_text(prompt_summary, 900)
    if reasoning_summary:
        payload["reasoning_summary"] = _safe_text(reasoning_summary, 1200)
    if document:
        payload["document"] = _safe_text(document, 320)
    if page is not None:
        payload["page"] = max(1, int(page))
    if heading:
        payload["heading"] = _safe_text(heading, 500)
    if metrics:
        safe_metrics: dict[str, object] = {}
        for key, value in list(metrics.items())[:24]:
            safe_key = _safe_text(key, 60)
            if not safe_key:
                continue
            if isinstance(value, (int, float, bool)) or value is None:
                safe_metrics[safe_key] = value
            else:
                safe_metrics[safe_key] = _safe_text(value, 240)
        if safe_metrics:
            payload["metrics"] = safe_metrics

    try:
        state.callback(payload)
    except Exception:
        # Progress is observability only and must never break retrieval/answer generation.
        return
