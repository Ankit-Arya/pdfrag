from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class V5Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block: int = 0
    line: int = 0
    font_size: float = 0.0
    font_name: str = ""
    source: str = "native"
    confidence: float = 1.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(slots=True)
class V5Element:
    element_id: str
    page_number: int
    order_index: int
    element_type: str
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    extraction_source: str = "native"
    heading_level: int | None = None
    parent_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class V5TableRow:
    row_index: int
    page_number: int
    cells: list[str]
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    source: str = "geometry"


@dataclass(slots=True)
class V5Table:
    table_id: str
    table_key: str
    title: str
    page_start: int
    page_end: int
    columns: list[str]
    rows: list[V5TableRow]
    bbox_by_page: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "geometry"
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class V5Page:
    page_number: int
    width: float
    height: float
    language: str = ""
    native_chars: int = 0
    ocr_used: bool = False
    quality_score: float = 1.0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class V5LayoutDocument:
    filename: str
    total_pages: int
    pages: list[V5Page]
    elements: list[V5Element]
    tables: list[V5Table]
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class V5Chunk:
    chunk_id: str
    chunk_index: int
    page_number: int
    page_end: int
    content_type: str
    text: str
    parent_key: str
    section_path: list[str] = field(default_factory=list)
    heading: str = ""
    table_id: str | None = None
    table_row_index: int | None = None
    extraction_confidence: float = 1.0
    authority_status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
