from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable

from app.config import get_settings
from app.rag.v5.types import (
    V5Element,
    V5LayoutDocument,
    V5Page,
    V5Table,
    V5TableRow,
    V5Word,
)

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore[assignment]

try:
    import pytesseract
    from PIL import Image
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

try:
    from app.rag.table_quality import is_plausible_table as _repo_table_quality
except Exception:  # pragma: no cover
    _repo_table_quality = None

_HEADING_NUMBER_RE = re.compile(r"^(?:\d{1,3}(?:\.\d{1,3}){0,6}|[A-Z]|[IVXLCM]{1,8})[.)-]?\s+\S+")
_SERIAL_RE = re.compile(r"^\(?\d{1,3}[.)]?\)?$")
_VALUE_RE = re.compile(
    r"(?:₹|rs\.?\s*)?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|km/?h|kmph|m/?s|mm|cm|km|m|kg|g|bar|kpa|psi|kv|v|a|ma|hz|sec(?:ond)?s?|min(?:ute)?s?|hours?|days?|lakh|crore|rupees?))?",
    re.IGNORECASE,
)
_TABLE_HEADER_CUES = {
    "amount", "compensation", "nature", "injury", "description", "condition", "speed",
    "limit", "value", "remarks", "remark", "action", "responsibility", "responsible",
    "role", "frequency", "interval", "item", "particulars", "parameter", "requirement",
    "sl no", "s no", "serial", "no.", "category", "type", "status", "unit",
}
_LIST_PREFIX_RE = re.compile(r"^(?:[-*•]|\(?[a-zA-Z0-9]{1,3}[.)])\s+")


@dataclass(slots=True)
class _Line:
    index: int
    words: list[V5Word]
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool
    source: str

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass(slots=True)
class _Segment:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()


def _hash_id(*parts: object) -> str:
    seed = "|".join(str(part) for part in parts)
    return str(uuid.uuid5(uuid.UUID("7b6f02d1-ae62-4b52-b312-c4fbd3c18f05"), seed))


def _bbox_union(boxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    values = list(boxes)
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def _bbox_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ix0, iy0 = max(left[0], right[0]), max(left[1], right[1])
    ix1, iy1 = min(left[2], right[2]), min(left[3], right[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    return inter / area


def _native_words(page: Any) -> list[V5Word]:
    output: list[V5Word] = []
    try:
        raw_words = page.get_text("words", sort=True)
    except Exception:
        raw_words = []
    for raw in raw_words:
        if len(raw) < 8:
            continue
        x0, y0, x1, y1, text_value, block, line, _word = raw[:8]
        text_value = _clean(text_value)
        if not text_value:
            continue
        output.append(
            V5Word(
                text=text_value,
                x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                block=int(block), line=int(line), source="native", confidence=1.0,
            )
        )
    return output


def _native_lines(page: Any, words: list[V5Word]) -> list[_Line]:
    word_groups: dict[tuple[int, int], list[V5Word]] = defaultdict(list)
    for word in words:
        word_groups[(word.block, word.line)].append(word)

    span_meta: dict[tuple[int, int], tuple[float, bool]] = {}
    try:
        data = page.get_text("dict", sort=True)
        for block_index, block in enumerate(data.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for line_index, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                if not spans:
                    continue
                sizes = [float(span.get("size") or 0.0) for span in spans if span.get("text")]
                names = [str(span.get("font") or "") for span in spans]
                flags = [int(span.get("flags") or 0) for span in spans]
                size = max(sizes, default=0.0)
                bold = any("bold" in name.casefold() for name in names) or any(flag & 16 for flag in flags)
                span_meta[(block_index, line_index)] = (size, bold)
    except Exception:
        pass

    lines: list[_Line] = []
    for idx, (key, group) in enumerate(sorted(word_groups.items(), key=lambda item: (min(w.y0 for w in item[1]), min(w.x0 for w in item[1])))):
        group.sort(key=lambda word: word.x0)
        text_value = _clean(" ".join(word.text for word in group))
        if not text_value:
            continue
        size, bold = span_meta.get(key, (median([max(1.0, w.y1 - w.y0) for w in group]), False))
        lines.append(
            _Line(
                index=idx,
                words=group,
                text=text_value,
                bbox=_bbox_union(word.bbox for word in group),
                font_size=size,
                bold=bold,
                source="native",
            )
        )
    return lines


def _ocr_words(page: Any, languages: str, dpi: int) -> list[V5Word]:
    if pytesseract is None or Image is None or fitz is None:
        return []
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    data = pytesseract.image_to_data(
        image,
        lang=languages,
        config="--psm 3",
        output_type=pytesseract.Output.DICT,
    )
    sx = float(page.rect.width) / max(1, pix.width)
    sy = float(page.rect.height) / max(1, pix.height)
    output: list[V5Word] = []
    count = len(data.get("text", []))
    for index in range(count):
        text_value = _clean(data["text"][index])
        if not text_value:
            continue
        try:
            conf = float(data.get("conf", ["-1"] * count)[index])
        except Exception:
            conf = -1.0
        if conf < 20:
            continue
        left = float(data["left"][index]) * sx
        top = float(data["top"][index]) * sy
        width = float(data["width"][index]) * sx
        height = float(data["height"][index]) * sy
        output.append(
            V5Word(
                text=text_value,
                x0=left,
                y0=top,
                x1=left + width,
                y1=top + height,
                block=int(data.get("block_num", [0] * count)[index] or 0),
                line=int(data.get("line_num", [0] * count)[index] or 0),
                source="ocr",
                confidence=max(0.0, min(1.0, conf / 100.0)),
            )
        )
    return output


def _cluster_lines(words: list[V5Word], tolerance: float = 4.0) -> list[_Line]:
    if not words:
        return []
    ordered = sorted(words, key=lambda word: ((word.y0 + word.y1) / 2.0, word.x0))
    groups: list[list[V5Word]] = []
    centers: list[float] = []
    for word in ordered:
        center = (word.y0 + word.y1) / 2.0
        target = None
        best = tolerance + 1.0
        for index, existing in enumerate(centers[-6:]):
            absolute = len(centers) - len(centers[-6:]) + index
            distance = abs(center - existing)
            if distance <= tolerance and distance < best:
                target = absolute
                best = distance
        if target is None:
            groups.append([word])
            centers.append(center)
        else:
            groups[target].append(word)
            centers[target] = sum((item.y0 + item.y1) / 2.0 for item in groups[target]) / len(groups[target])
    lines: list[_Line] = []
    for idx, group in enumerate(groups):
        group.sort(key=lambda word: word.x0)
        text_value = _clean(" ".join(word.text for word in group))
        if not text_value:
            continue
        sizes = [max(1.0, word.y1 - word.y0) for word in group]
        lines.append(
            _Line(
                index=idx,
                words=group,
                text=text_value,
                bbox=_bbox_union(word.bbox for word in group),
                font_size=median(sizes),
                bold=False,
                source=group[0].source,
            )
        )
    return sorted(lines, key=lambda line: (line.y0, line.bbox[0]))


def _image_coverage(page: Any) -> float:
    page_area = max(1.0, float(page.rect.width * page.rect.height))
    total = 0.0
    try:
        for image in page.get_images(full=True):
            xref = int(image[0])
            for rect in page.get_image_rects(xref):
                total += max(0.0, float(rect.width * rect.height))
    except Exception:
        return 0.0
    return min(1.0, total / page_area)


def _should_ocr_layout(page: Any, native_words: list[V5Word]) -> bool:
    settings = get_settings()
    if settings.ocr_mode == "never":
        return False
    if settings.ocr_mode == "always":
        return True
    native_text = " ".join(word.text for word in native_words)
    alnum = sum(char.isalnum() for char in native_text)
    if alnum < max(50, settings.ocr_min_native_chars):
        return True
    if _image_coverage(page) >= _float_env("RAG_V5_OCR_IMAGE_COVERAGE", 0.45) and alnum < 450:
        return True
    return False


def _line_segments(line: _Line, page_width: float) -> list[_Segment]:
    if not line.words:
        return []
    gap_threshold = max(18.0, page_width * 0.032)
    words = sorted(line.words, key=lambda word: word.x0)
    groups: list[list[V5Word]] = [[words[0]]]
    for word in words[1:]:
        previous = groups[-1][-1]
        gap = word.x0 - previous.x1
        if gap > gap_threshold:
            groups.append([word])
        else:
            groups[-1].append(word)
    return [
        _Segment(
            text=_clean(" ".join(word.text for word in group)),
            x0=min(word.x0 for word in group),
            x1=max(word.x1 for word in group),
            y0=min(word.y0 for word in group),
            y1=max(word.y1 for word in group),
        )
        for group in groups
        if _clean(" ".join(word.text for word in group))
    ]


def _cluster_positions(values: list[tuple[float, int]], tolerance: float) -> list[tuple[float, set[int], int]]:
    clusters: list[dict[str, Any]] = []
    for value, line_index in sorted(values):
        target = None
        best = tolerance + 1.0
        for index, cluster in enumerate(clusters):
            distance = abs(value - float(cluster["center"]))
            if distance <= tolerance and distance < best:
                best = distance
                target = index
        if target is None:
            clusters.append({"values": [value], "lines": {line_index}, "center": value})
        else:
            cluster = clusters[target]
            cluster["values"].append(value)
            cluster["lines"].add(line_index)
            cluster["center"] = median(cluster["values"])
    return [
        (float(cluster["center"]), set(cluster["lines"]), len(cluster["values"]))
        for cluster in clusters
    ]


def _looks_value(text_value: str) -> bool:
    text_value = _clean(text_value)
    if not text_value:
        return False
    return bool(_VALUE_RE.fullmatch(text_value) or (_VALUE_RE.search(text_value) and len(text_value.split()) <= 4))


def _looks_serial(text_value: str) -> bool:
    return bool(_SERIAL_RE.fullmatch(_clean(text_value)))


def _plausible_rows(rows: list[list[str]]) -> bool:
    nonempty_rows = [row for row in rows if sum(bool(_clean(cell)) for cell in row) >= 2]
    if len(nonempty_rows) < 3:
        return False
    width = max(len(row) for row in nonempty_rows)
    if width < 2:
        return False

    # A numbered procedure/list often has an aligned marker column plus one long prose
    # column. Geometry alone makes that look tabular, but converting it into a table hides
    # rule headings and damages section continuity. Treat it as prose unless a second
    # non-serial value/data column is actually present.
    list_like = 0
    for row in nonempty_rows:
        cells = [_clean(cell) for cell in row if _clean(cell)]
        serials = [cell for cell in cells if _looks_serial(cell)]
        other = [cell for cell in cells if not _looks_serial(cell)]
        if serials and len(other) == 1 and len(other[0]) >= 24:
            list_like += 1
    if list_like / len(nonempty_rows) >= 0.55:
        nonserial_values = sum(
            any(_looks_value(cell) and not _looks_serial(cell) for cell in row)
            for row in nonempty_rows
        )
        if nonserial_values < 2:
            return False

    numeric_rows = sum(
        any(_looks_value(cell) and not _looks_serial(cell) for cell in row)
        for row in nonempty_rows
    )
    avg_nonempty = sum(sum(bool(_clean(cell)) for cell in row) for row in nonempty_rows) / len(nonempty_rows)
    if numeric_rows >= 2:
        return True
    # Text-only responsibility/matrix tables need stronger repeated column structure.
    return len(nonempty_rows) >= 4 and avg_nonempty >= 2.0 and list_like < len(nonempty_rows) * 0.55


def _header_name(parts: list[str], fallback: str) -> str:
    value = _clean(" ".join(part for part in parts if _clean(part)))
    if not value:
        return fallback
    return value[:120]


def _geometry_tables(lines: list[_Line], page_number: int, page_width: float, page_height: float) -> list[V5Table]:
    if len(lines) < 3:
        return []
    segments_by_line = {line.index: _line_segments(line, page_width) for line in lines}
    position_values: list[tuple[float, int]] = []
    for line in lines:
        segments = segments_by_line[line.index]
        if len(segments) < 2:
            continue
        for segment in segments:
            position_values.append((segment.x0, line.index))
    tolerance = max(12.0, page_width * 0.025)
    clusters = _cluster_positions(position_values, tolerance)
    stable = [cluster for cluster in clusters if len(cluster[1]) >= 3]
    stable.sort(key=lambda item: item[0])
    if len(stable) < 2:
        return []

    # Remove near-duplicate x bands produced by indentation inside one semantic column.
    column_x: list[float] = []
    for center, line_ids, _count in stable:
        if column_x and center - column_x[-1] < max(28.0, page_width * 0.055):
            # Keep the band used by more lines.
            previous_center = column_x[-1]
            previous = min(stable, key=lambda item: abs(item[0] - previous_center))
            if len(line_ids) > len(previous[1]):
                column_x[-1] = center
            continue
        column_x.append(center)
    if len(column_x) < 2:
        return []
    column_x = column_x[:6]

    candidate_flags: list[bool] = []
    mapped_rows: list[list[str]] = []
    for line in lines:
        cells = [""] * len(column_x)
        used_cols: set[int] = set()
        for segment in segments_by_line[line.index]:
            nearest = min(range(len(column_x)), key=lambda index: abs(segment.x0 - column_x[index]))
            if abs(segment.x0 - column_x[nearest]) > max(tolerance * 1.8, 34.0):
                continue
            cells[nearest] = _clean(" ".join(part for part in (cells[nearest], segment.text) if part))
            used_cols.add(nearest)
        candidate = len(used_cols) >= 2
        candidate_flags.append(candidate)
        mapped_rows.append(cells)

    groups: list[list[int]] = []
    current: list[int] = []
    misses = 0
    for index, is_candidate in enumerate(candidate_flags):
        if is_candidate:
            if current and misses:
                current.extend(range(index - misses, index))
            current.append(index)
            misses = 0
        elif current:
            misses += 1
            if misses > 1:
                actual = [idx for idx in current if candidate_flags[idx]]
                if len(actual) >= 3:
                    groups.append(list(current))
                current = []
                misses = 0
    if current:
        actual = [idx for idx in current if candidate_flags[idx]]
        if len(actual) >= 3:
            groups.append(current)

    output: list[V5Table] = []
    for group_number, group in enumerate(groups, 1):
        selected = [idx for idx in group if candidate_flags[idx]]
        if len(selected) < 3:
            continue
        rows = [mapped_rows[idx] for idx in selected]
        if not _plausible_rows(rows):
            continue
        # Avoid interpreting ordinary two-column page headers as tables.
        y0 = min(lines[idx].y0 for idx in selected)
        y1 = max(lines[idx].y1 for idx in selected)
        if y1 - y0 < max(35.0, page_height * 0.045):
            continue

        header_count = 0
        for row in rows[:5]:
            joined = _clean(" ".join(row)).casefold()
            has_cue = any(cue in joined for cue in _TABLE_HEADER_CUES)
            has_value = any(_looks_value(cell) for cell in row)
            has_serial = any(_looks_serial(cell) for cell in row[:1])
            if (has_cue and not has_value) or (not has_value and not has_serial and header_count < 2):
                header_count += 1
                continue
            break
        header_count = min(header_count, max(0, len(rows) - 2))
        header_rows = rows[:header_count]
        data_rows = rows[header_count:]

        columns: list[str] = []
        for col in range(len(column_x)):
            parts = [row[col] for row in header_rows if col < len(row) and row[col]]
            fallback = "No." if col == 0 and any(_looks_serial(row[0]) for row in data_rows if row) else f"Column {col + 1}"
            columns.append(_header_name(parts, fallback))

        # Merge continuation lines into the preceding logical row.
        logical_rows: list[list[str]] = []
        logical_boxes: list[list[tuple[float, float, float, float]]] = []
        selected_data_indices = selected[header_count:]
        for row, line_idx in zip(data_rows, selected_data_indices, strict=False):
            has_serial = bool(row and _looks_serial(row[0]))
            has_value = any(_looks_value(cell) for cell in row[1:])
            first_nonempty = next((idx for idx, cell in enumerate(row) if cell), -1)
            continuation = bool(
                logical_rows
                and not has_serial
                and not has_value
                and first_nonempty >= 1
            )
            if continuation:
                previous = logical_rows[-1]
                for col, cell in enumerate(row):
                    if cell:
                        previous[col] = _clean(" ".join(part for part in (previous[col], cell) if part))
                logical_boxes[-1].append(lines[line_idx].bbox)
            else:
                logical_rows.append(list(row))
                logical_boxes.append([lines[line_idx].bbox])

        if not _plausible_rows(logical_rows):
            continue
        # Infer title from nearby single-column lines immediately above the table.
        first_line = selected[0]
        title_candidates: list[str] = []
        for prior in range(max(0, first_line - 6), first_line):
            if candidate_flags[prior]:
                continue
            text_value = _clean(lines[prior].text)
            if 2 <= len(text_value) <= 180:
                title_candidates.append(text_value)
        title = " | ".join(title_candidates[-3:])[:320] or f"Table on page {page_number}"

        table_rows: list[V5TableRow] = []
        for row_index, (cells, boxes) in enumerate(zip(logical_rows, logical_boxes, strict=True), 1):
            if sum(bool(_clean(cell)) for cell in cells) < 2:
                continue
            table_rows.append(
                V5TableRow(
                    row_index=row_index,
                    page_number=page_number,
                    cells=[_clean(cell) for cell in cells],
                    bbox=_bbox_union(boxes),
                    confidence=0.82,
                    source="geometry-aligned",
                )
            )
        if len(table_rows) < 2:
            continue
        bbox = _bbox_union(lines[idx].bbox for idx in selected)
        table_id = _hash_id("table", page_number, group_number, bbox, title)
        output.append(
            V5Table(
                table_id=table_id,
                table_key=table_id,
                title=title,
                page_start=page_number,
                page_end=page_number,
                columns=columns,
                rows=table_rows,
                bbox_by_page={page_number: bbox},
                confidence=min(0.94, 0.72 + min(0.18, len(table_rows) * 0.015)),
                source="geometry-aligned",
            )
        )
    return output


def _plumber_tables(plumber_page: Any, page_number: int) -> list[V5Table]:
    if plumber_page is None:
        return []
    strategies = [
        {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 4, "join_tolerance": 4},
        {
            "vertical_strategy": "text", "horizontal_strategy": "text",
            "min_words_vertical": 2, "min_words_horizontal": 1,
            "intersection_tolerance": 6, "text_x_tolerance": 2, "text_y_tolerance": 3,
        },
    ]
    output: list[V5Table] = []
    for strategy_index, settings in enumerate(strategies):
        try:
            tables = plumber_page.find_tables(table_settings=settings)
        except Exception:
            continue
        for table_index, table in enumerate(tables):
            try:
                raw_rows = table.extract() or []
            except Exception:
                continue
            width = max((len(row or []) for row in raw_rows), default=0)
            rows = [[_clean(cell or "") for cell in (row or [])] + [""] * (width - len(row or [])) for row in raw_rows]
            rows = [row for row in rows if any(row)]
            if not _plausible_rows(rows):
                continue
            if _repo_table_quality is not None and not _repo_table_quality(rows, border_based=(strategy_index == 0)):
                continue
            header = rows[0]
            columns = [cell or f"Column {index + 1}" for index, cell in enumerate(header)]
            data_rows = rows[1:] if len(rows) > 2 else rows
            bbox_raw = getattr(table, "bbox", None) or (0.0, 0.0, float(plumber_page.width), float(plumber_page.height))
            bbox = tuple(float(value) for value in bbox_raw)
            table_id = _hash_id("plumber", page_number, strategy_index, table_index, bbox)
            vrows = [
                V5TableRow(
                    row_index=index,
                    page_number=page_number,
                    cells=row,
                    bbox=bbox,
                    confidence=0.94 if strategy_index == 0 else 0.78,
                    source="pdfplumber-lines" if strategy_index == 0 else "pdfplumber-text",
                )
                for index, row in enumerate(data_rows, 1)
                if sum(bool(cell) for cell in row) >= 2
            ]
            if len(vrows) < 2:
                continue
            output.append(
                V5Table(
                    table_id=table_id,
                    table_key=table_id,
                    title=f"Table on page {page_number}",
                    page_start=page_number,
                    page_end=page_number,
                    columns=columns,
                    rows=vrows,
                    bbox_by_page={page_number: bbox},
                    confidence=0.95 if strategy_index == 0 else 0.80,
                    source="pdfplumber-lines" if strategy_index == 0 else "pdfplumber-text",
                )
            )
    # Keep higher-confidence table when strategies duplicate the same region.
    deduped: list[V5Table] = []
    for candidate in sorted(output, key=lambda table: -table.confidence):
        bbox = candidate.bbox_by_page[page_number]
        if any(_bbox_overlap_ratio(bbox, existing.bbox_by_page[page_number]) >= 0.65 for existing in deduped):
            continue
        deduped.append(candidate)
    return deduped


def _merge_table_candidates(primary: list[V5Table], fallback: list[V5Table]) -> list[V5Table]:
    output = list(primary)
    for candidate in fallback:
        bbox = candidate.bbox_by_page.get(candidate.page_start, (0.0, 0.0, 0.0, 0.0))
        overlapping = [
            existing for existing in output
            if existing.page_start == candidate.page_start
            and _bbox_overlap_ratio(bbox, existing.bbox_by_page.get(existing.page_start, (0, 0, 0, 0))) >= 0.55
        ]
        if not overlapping:
            output.append(candidate)
            continue
        best = max(overlapping, key=lambda table: table.confidence)
        if candidate.confidence > best.confidence + 0.08:
            output.remove(best)
            output.append(candidate)
    return sorted(output, key=lambda table: (table.page_start, table.bbox_by_page[table.page_start][1]))


def _heading_level(line: _Line, body_font: float, size_levels: list[float]) -> int | None:
    text_value = _clean(line.text)
    if not text_value or len(text_value) < 4 or len(text_value) > 220:
        return None
    words = text_value.split()
    letters = [char for char in text_value if char.isalpha()]
    upper_ratio = (sum(char.isupper() for char in letters) / len(letters)) if letters else 0.0
    title_ratio = (sum(char.isupper() for char in (word[:1] for word in words) if char) / max(1, len(words)))
    numbered = bool(_HEADING_NUMBER_RE.match(text_value))
    if re.fullmatch(r"PART\s+[IVXLCM]+", text_value, flags=re.IGNORECASE):
        return 3
    if re.fullmatch(r"(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s+SCHEDULE", text_value, flags=re.IGNORECASE):
        return 2
    larger = line.font_size >= max(body_font * 1.13, body_font + 0.8)
    short = len(words) <= 18
    # Font size alone is not enough: corrupted PDFs often expose isolated fragments in a
    # larger font. Require either explicit numbering, strong uppercase/title casing, or bold.
    heading_like = short and (
        numbered
        or (len(letters) >= 4 and upper_ratio >= 0.78)
        or (line.bold and len(letters) >= 4)
        or (larger and len(text_value) >= 8 and len(words) >= 2 and (upper_ratio >= 0.28 or title_ratio >= 0.50))
    )
    if not heading_like:
        return None
    if numbered:
        number_match = re.match(r"^(\d{1,3}(?:\.\d{1,3})*)", text_value)
        if number_match:
            return min(6, 1 + number_match.group(1).count("."))
    if size_levels:
        nearest = min(range(len(size_levels)), key=lambda idx: abs(size_levels[idx] - line.font_size))
        return min(6, nearest + 1)
    return 2 if line.bold else 1


def _merge_rule_number_heading_lines(lines: list[_Line]) -> list[_Line]:
    """Join a standalone rule number with the heading printed beside it on the same baseline."""
    ordered = sorted(lines, key=lambda line: (line.y0, line.bbox[0], line.index))
    pairs: dict[int, tuple[_Line, _Line]] = {}
    consumed: set[int] = set()
    for number in ordered:
        if not re.fullmatch(r"\d{1,3}[.)]?", _clean(number.text)):
            continue
        number_center = (number.bbox[1] + number.bbox[3]) / 2.0
        candidates = []
        for neighbor in ordered:
            if neighbor is number or neighbor.bbox[0] <= number.bbox[2]:
                continue
            if abs(number_center - (neighbor.bbox[1] + neighbor.bbox[3]) / 2.0) > 3.5:
                continue
            neighbor_text = _clean(neighbor.text)
            if not (4 <= len(neighbor_text) <= 220 and len(neighbor_text.split()) <= 18):
                continue
            letters = [char for char in neighbor_text if char.isalpha()]
            words = [word for word in neighbor_text.split() if any(char.isalpha() for char in word)]
            upper_ratio = (sum(char.isupper() for char in letters) / len(letters)) if letters else 0.0
            title_ratio = (sum(word[:1].isupper() for word in words) / len(words)) if words else 0.0
            dash_heading = bool(re.search(r"[-–—]{1,3}\s*$", neighbor_text))
            headingish = (
                neighbor.bold
                or dash_heading
                or (len(letters) >= 4 and upper_ratio >= 0.72)
                or (len(words) >= 2 and title_ratio >= 0.60 and not _looks_value(neighbor_text))
            )
            if headingish:
                candidates.append(neighbor)
        if not candidates:
            continue
        neighbor = min(candidates, key=lambda item: (abs(number_center - (item.bbox[1] + item.bbox[3]) / 2.0), item.bbox[0]))
        if id(number) in consumed or id(neighbor) in consumed:
            continue
        pairs[id(number)] = (number, neighbor)
        pairs[id(neighbor)] = (number, neighbor)
        consumed.update({id(number), id(neighbor)})

    output: list[_Line] = []
    emitted: set[tuple[int, int]] = set()
    for line in ordered:
        pair = pairs.get(id(line))
        if pair is None:
            output.append(line)
            continue
        number, neighbor = pair
        key = (id(number), id(neighbor))
        if key in emitted:
            continue
        emitted.add(key)
        merged_words = sorted([*number.words, *neighbor.words], key=lambda word: word.x0)
        output.append(
            _Line(
                index=min(number.index, neighbor.index),
                words=merged_words,
                text=f"{_clean(number.text)} {_clean(neighbor.text)}",
                bbox=_bbox_union([number.bbox, neighbor.bbox]),
                font_size=max(number.font_size, neighbor.font_size),
                bold=number.bold or neighbor.bold,
                source=number.source if number.source == neighbor.source else "mixed-native",
            )
        )
    return sorted(output, key=lambda line: (line.y0, line.bbox[0], line.index))


def _lines_to_elements(
    lines: list[_Line],
    *,
    page_number: int,
    table_boxes: list[tuple[float, float, float, float]],
) -> list[V5Element]:
    lines = _merge_rule_number_heading_lines(lines)
    body_sizes = [line.font_size for line in lines if len(line.text) >= 30 and line.font_size > 0]
    body_font = median(body_sizes) if body_sizes else median([line.font_size for line in lines if line.font_size > 0] or [10.0])
    size_levels = sorted({round(line.font_size, 1) for line in lines if line.font_size >= body_font * 1.08}, reverse=True)[:6]
    output: list[V5Element] = []
    for order, line in enumerate(lines):
        if any(_bbox_overlap_ratio(line.bbox, box) >= 0.45 for box in table_boxes):
            continue
        text_value = _clean(line.text)
        if not text_value:
            continue
        level = _heading_level(line, body_font, size_levels)
        if level is not None:
            element_type = "heading"
        elif _LIST_PREFIX_RE.match(text_value):
            element_type = "list_item"
        else:
            element_type = "paragraph"
        output.append(
            V5Element(
                element_id=_hash_id("element", page_number, order, line.bbox, text_value),
                page_number=page_number,
                order_index=order,
                element_type=element_type,
                text=text_value,
                bbox=line.bbox,
                confidence=min((word.confidence for word in line.words), default=1.0),
                extraction_source=line.source,
                heading_level=level,
                metadata={"font_size": round(line.font_size, 2), "bold": line.bold},
            )
        )
    return output


def _figure_elements(page: Any, page_number: int, lines: list[_Line]) -> list[V5Element]:
    output: list[V5Element] = []
    try:
        images = page.get_images(full=True)
    except Exception:
        return output
    order_base = 100000
    for image_index, image in enumerate(images):
        xref = int(image[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect_index, rect in enumerate(rects):
            bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if area < max(2500.0, float(page.rect.width * page.rect.height) * 0.01):
                continue
            nearby = [
                line.text for line in lines
                if line.y0 >= bbox[3] - 4 and line.y0 <= bbox[3] + 60
            ][:3]
            caption = _clean(" ".join(nearby))
            text_value = f"Figure/image on page {page_number}"
            if caption:
                text_value += f". Nearby caption/text: {caption}"
            output.append(
                V5Element(
                    element_id=_hash_id("figure", page_number, xref, rect_index, bbox),
                    page_number=page_number,
                    order_index=order_base + image_index * 10 + rect_index,
                    element_type="figure",
                    text=text_value,
                    bbox=bbox,
                    confidence=0.70 if caption else 0.45,
                    extraction_source="embedded-image",
                    metadata={"xref": xref, "caption": caption, "vision_described": False},
                )
            )
    return output


def _normalize_margin_text(value: str) -> str:
    value = re.sub(r"\d+", "#", _clean(value).casefold())
    value = re.sub(r"\s+", " ", value)
    return value[:180]


def _remove_repeated_margins(elements: list[V5Element], pages: list[V5Page]) -> list[V5Element]:
    if len(pages) < 3:
        return elements
    page_heights = {page.page_number: page.height for page in pages}
    candidates: list[tuple[V5Element, str]] = []
    counts: Counter[str] = Counter()
    page_sets: dict[str, set[int]] = defaultdict(set)
    for element in elements:
        height = page_heights.get(element.page_number, 0.0)
        if height <= 0:
            continue
        if element.bbox[1] > height * 0.10 and element.bbox[3] < height * 0.90:
            continue
        key = _normalize_margin_text(element.text)
        if len(key) < 4:
            continue
        candidates.append((element, key))
        counts[key] += 1
        page_sets[key].add(element.page_number)
    threshold = max(3, math.ceil(len(pages) * 0.45))
    repeated = {key for key, page_numbers in page_sets.items() if len(page_numbers) >= threshold}
    if not repeated:
        return elements
    repeated_ids = {element.element_id for element, key in candidates if key in repeated}
    return [element for element in elements if element.element_id not in repeated_ids]


def _table_signature(table: V5Table) -> tuple[int, tuple[str, ...]]:
    normalized = tuple(re.sub(r"[^a-z0-9]+", " ", col.casefold()).strip()[:40] for col in table.columns)
    return (len(table.columns), normalized)


def _similar_columns(left: V5Table, right: V5Table) -> bool:
    if len(left.columns) != len(right.columns):
        return False
    a = [set(re.findall(r"[a-z0-9]+", col.casefold())) for col in left.columns]
    b = [set(re.findall(r"[a-z0-9]+", col.casefold())) for col in right.columns]
    overlap = 0
    comparable = 0
    for left_terms, right_terms in zip(a, b, strict=True):
        if not left_terms and not right_terms:
            continue
        comparable += 1
        if left_terms & right_terms:
            overlap += 1
    if comparable and overlap / comparable >= 0.5:
        return True
    # Continuation pages often omit the header entirely but preserve column count and geometry.
    return left.source == right.source == "geometry-aligned" and len(left.rows) >= 3 and len(right.rows) >= 3


def _merge_multipage_tables(tables: list[V5Table]) -> list[V5Table]:
    if not tables:
        return []
    ordered = sorted(tables, key=lambda table: (table.page_start, table.bbox_by_page[table.page_start][1]))
    output: list[V5Table] = []
    for table in ordered:
        if output:
            previous = output[-1]
            if table.page_start == previous.page_end + 1 and _similar_columns(previous, table):
                # Merge only when the continuation begins in the upper half of the page.
                bbox = table.bbox_by_page.get(table.page_start, (0, 9999, 0, 9999))
                if bbox[1] < 420:
                    offset = len(previous.rows)
                    previous.rows.extend(
                        V5TableRow(
                            row_index=offset + index,
                            page_number=row.page_number,
                            cells=row.cells,
                            bbox=row.bbox,
                            confidence=row.confidence,
                            source=row.source,
                        )
                        for index, row in enumerate(table.rows, 1)
                    )
                    previous.page_end = table.page_end
                    previous.bbox_by_page.update(table.bbox_by_page)
                    previous.confidence = min(previous.confidence, table.confidence)
                    previous.metadata["multipage"] = True
                    continue
        output.append(table)
    return output


def _column_name_score(value: str) -> float:
    value = _clean(value)
    if not value:
        return 0.0
    folded = value.casefold()
    if folded.startswith("column ") or re.fullmatch(r"\(?\d+\)?", value):
        return 0.0
    letters = sum(char.isalpha() for char in value)
    words = re.findall(r"[A-Za-z]+", value)
    if letters < 3 or not words:
        return 0.05
    if len(value) <= 3 and len(words) <= 1:
        return 0.10
    return min(1.0, 0.35 + min(0.45, letters / 40.0) + min(0.20, len(words) * 0.05))


def _schema_score(columns: list[str]) -> float:
    if not columns:
        return 0.0
    return sum(_column_name_score(value) for value in columns) / len(columns)


def _same_section_family(left: V5Table, right: V5Table) -> bool:
    if not left.section_path or not right.section_path:
        return True
    left_norm = normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    left_root = normalize(left.section_path[0])
    right_root = normalize(right.section_path[0])
    return bool(left_root and right_root and (left_root == right_root or left_root in right_root or right_root in left_root))


def _propagate_table_schemas(tables: list[V5Table]) -> None:
    """Carry reliable column names onto continuation fragments.

    A visual table often repeats values but omits/reorders its header on later pages. Geometry can
    still recover the rows, yet generic ``Column 1`` labels lose the row/value relationship that
    RAG needs. We inherit a schema only from a nearby table in the same section family and only when
    column widths are compatible; no domain-specific labels are invented.
    """
    ordered = sorted(tables, key=lambda table: (table.page_start, table.page_end, table.table_key))
    for index, table in enumerate(ordered):
        current_score = _schema_score(table.columns)
        if current_score >= 0.62:
            continue
        candidates = []
        for previous in ordered[max(0, index - 8):index]:
            if table.page_start - previous.page_end > 3:
                continue
            if not _same_section_family(previous, table):
                continue
            previous_score = _schema_score(previous.columns)
            if previous_score < 0.62:
                continue
            if len(previous.columns) == len(table.columns):
                inherited = list(previous.columns)
            elif len(table.columns) == 2 and len(previous.columns) == 3:
                # A common continuation pattern drops the serial-number column while keeping
                # description + value. Use the last two *existing source headers* only.
                inherited = list(previous.columns[-2:])
            else:
                continue
            candidates.append((previous_score, previous.page_end, inherited))
        if not candidates:
            continue
        _score, _page, inherited = max(candidates, key=lambda item: (item[0], item[1]))
        table.columns = inherited
        table.metadata["schema_inherited"] = True


def _assign_section_paths(elements: list[V5Element], tables: list[V5Table]) -> None:
    stack: list[tuple[int, str]] = []
    ordered = sorted(elements, key=lambda element: (element.page_number, element.bbox[1], element.bbox[0], element.order_index))
    snapshots: list[tuple[int, float, list[str]]] = []
    for element in ordered:
        if element.element_type == "heading" and element.heading_level:
            level = element.heading_level
            stack = [(old_level, text) for old_level, text in stack if old_level < level]
            stack.append((level, element.text))
            element.parent_key = " > ".join(text for _, text in stack)
            element.metadata["section_path"] = [text for _, text in stack]
        else:
            element.metadata["section_path"] = [text for _, text in stack]
            element.parent_key = " > ".join(text for _, text in stack) or "Unsectioned content"
        snapshots.append((element.page_number, element.bbox[1], [text for _, text in stack]))

    for table in tables:
        bbox = table.bbox_by_page.get(table.page_start, (0.0, 0.0, 0.0, 0.0))
        eligible = [
            path for page_number, y0, path in snapshots
            if page_number < table.page_start or (page_number == table.page_start and y0 <= bbox[1] + 1.0)
        ]
        table.section_path = list(eligible[-1]) if eligible else []
        if table.section_path and (
            table.title.startswith("Table on page")
            or re.match(r"^\d{1,3}[.)]?\s", table.title)
            or len(table.title) > 220
        ):
            table.title = " > ".join(table.section_path[-3:])


def extract_layout_document(data: bytes, filename: str) -> V5LayoutDocument:
    """Extract a structure-preserving representation from a PDF.

    The v4 pipeline selected one flattened page string and appended tables after it. v5 keeps
    bounding boxes, heading levels, table rows/columns, OCR provenance and page reading order so
    later chunking can preserve semantic units instead of recreating structure from plain text.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for RAG v5 layout extraction")
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"{filename}: invalid PDF signature")
    settings = get_settings()
    doc = fitz.open(stream=data, filetype="pdf")
    plumber_doc = None
    if pdfplumber is not None:
        try:
            plumber_doc = pdfplumber.open(io.BytesIO(data))
        except Exception:
            logger.exception("pdfplumber could not open %s for v5 extraction", filename)

    pages: list[V5Page] = []
    elements: list[V5Element] = []
    tables: list[V5Table] = []
    warnings: list[str] = []
    ocr_page_numbers: list[int] = []
    table_candidate_pages: set[int] = set()
    table_rejected_pages: set[int] = set()
    try:
        for page_index in range(len(doc)):
            page_number = page_index + 1
            page = doc.load_page(page_index)
            native_words = _native_words(page)
            ocr_used = _should_ocr_layout(page, native_words)
            words = native_words
            if ocr_used:
                try:
                    ocr_words = _ocr_words(page, settings.ocr_languages, settings.ocr_dpi)
                    if ocr_words:
                        words = ocr_words
                        ocr_page_numbers.append(page_number)
                    else:
                        ocr_used = False
                except Exception as exc:
                    ocr_used = False
                    warnings.append(f"{filename}: v5 layout OCR failed on page {page_number} ({type(exc).__name__}).")
                    logger.exception("v5 layout OCR failed for %s page %s", filename, page_number)

            if words and words[0].source == "native":
                lines = _native_lines(page, words)
            else:
                lines = _cluster_lines(words, tolerance=max(3.0, float(page.rect.height) * 0.004))

            plumber_page = plumber_doc.pages[page_index] if plumber_doc is not None and page_index < len(plumber_doc.pages) else None
            native_tables = _plumber_tables(plumber_page, page_number)
            # PDF generators often place each visual table cell in a separate text block.
            # Re-cluster words spatially for table detection instead of trusting block/line IDs.
            table_lines = _cluster_lines(words, tolerance=max(3.0, float(page.rect.height) * 0.005))
            geometry_tables = _geometry_tables(table_lines, page_number, float(page.rect.width), float(page.rect.height))
            table_candidates = _merge_table_candidates(native_tables, geometry_tables)
            if table_candidates:
                table_candidate_pages.add(page_number)
            min_conf = _float_env("RAG_V5_MIN_TABLE_CONFIDENCE", 0.62)
            page_tables = [table for table in table_candidates if table.confidence >= min_conf]
            if table_candidates and len(page_tables) < len(table_candidates):
                table_rejected_pages.add(page_number)
            tables.extend(page_tables)
            table_boxes = [table.bbox_by_page[page_number] for table in page_tables]

            page_elements = _lines_to_elements(lines, page_number=page_number, table_boxes=table_boxes)
            page_elements.extend(_figure_elements(page, page_number, lines))
            elements.extend(page_elements)

            native_chars = sum(len(word.text) for word in native_words)
            quality = 1.0
            page_warnings: list[str] = []
            if not words:
                quality = 0.0
                page_warnings.append("no readable words")
            elif ocr_used:
                quality = sum(word.confidence for word in words) / max(1, len(words))
            if geometry_tables and not native_tables:
                page_warnings.append("table recovered from aligned word geometry")
            if table_candidates and not page_tables:
                page_warnings.append("table-like structure detected but all candidates were below confidence threshold")
            elif table_candidates and len(page_tables) < len(table_candidates):
                page_warnings.append("some table-like structures were below confidence threshold")
            if _image_coverage(page) >= 0.20:
                page_warnings.append("page contains substantial image content")
            pages.append(
                V5Page(
                    page_number=page_number,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    native_chars=native_chars,
                    ocr_used=ocr_used,
                    quality_score=max(0.0, min(1.0, quality)),
                    warnings=page_warnings,
                )
            )
    finally:
        if plumber_doc is not None:
            plumber_doc.close()
        doc.close()

    elements = _remove_repeated_margins(elements, pages)
    tables = _merge_multipage_tables(tables)
    _assign_section_paths(elements, tables)
    _propagate_table_schemas(tables)

    suspected_table_pages = sorted({table.page_start for table in tables})
    figure_count = sum(element.element_type == "figure" for element in elements)
    heading_count = sum(element.element_type == "heading" for element in elements)
    table_rows = sum(len(table.rows) for table in tables)
    low_quality_pages = [page.page_number for page in pages if page.quality_score < 0.55]
    if figure_count:
        warnings.append(
            f"{filename}: {figure_count} figure/image region(s) preserved with nearby captions/text; "
            "visual semantics are not inferred unless a future approved vision extractor is enabled."
        )
    if low_quality_pages:
        warnings.append(f"{filename}: low extraction confidence on pages {low_quality_pages[:30]}.")
    return V5LayoutDocument(
        filename=filename,
        total_pages=len(pages),
        pages=pages,
        elements=sorted(elements, key=lambda element: (element.page_number, element.bbox[1], element.bbox[0], element.order_index)),
        tables=tables,
        warnings=list(dict.fromkeys(warnings)),
        metrics={
            "processing_version": "rag-v5.0.0",
            "pages": len(pages),
            "ocr_pages": ocr_page_numbers,
            "headings": heading_count,
            "tables": len(tables),
            "table_rows": table_rows,
            "figures": figure_count,
            "low_quality_pages": low_quality_pages,
            "table_pages": suspected_table_pages,
            "table_candidate_pages": sorted(table_candidate_pages),
            "table_rejected_pages": sorted(table_rejected_pages),
        },
    )
