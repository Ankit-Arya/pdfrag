from __future__ import annotations

import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import get_settings
from app.rag.types import PageText

logger = logging.getLogger(__name__)

try:  # Optional but recommended for layout-preserving text and tables.
    import pdfplumber
except ImportError:  # pragma: no cover - exercised only in minimal installs
    pdfplumber = None  # type: ignore[assignment]

try:  # Optional OCR renderer.
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:  # Optional OCR engine bindings.
    import pytesseract
    from PIL import Image
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment,misc]


class PdfProcessingError(ValueError):
    """Raised when an uploaded PDF cannot be safely processed."""


@dataclass(slots=True)
class PdfExtractionResult:
    blocks: list[PageText]
    total_pages: int
    ocr_pages: list[int] = field(default_factory=list)
    table_count: int = 0
    warnings: list[str] = field(default_factory=list)


def ocr_available() -> bool:
    if fitz is None or pytesseract is None or Image is None:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def table_extraction_available() -> bool:
    return pdfplumber is not None


def safe_filename(filename: str | None) -> str:
    name = PurePath(filename or "document.pdf").name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return name[:160] or "document.pdf"


def extract_pdf_pages(data: bytes, filename: str) -> PdfExtractionResult:
    settings = get_settings()
    if not data.startswith(b"%PDF-"):
        raise PdfProcessingError(f"{filename}: file does not have a valid PDF signature")

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except (PdfReadError, ValueError, OSError) as exc:
        raise PdfProcessingError(f"{filename}: unreadable or malformed PDF") from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:  # pypdf exposes provider-specific exceptions here
            raise PdfProcessingError(f"{filename}: encrypted PDFs are not supported") from exc
        if not unlocked:
            raise PdfProcessingError(f"{filename}: password-protected PDFs are not supported")

    total_pages = len(reader.pages)
    plumber_doc = None
    fitz_doc = None
    warnings: list[str] = []

    if pdfplumber is not None:
        try:
            plumber_doc = pdfplumber.open(io.BytesIO(data))
        except Exception as exc:
            warnings.append(f"{filename}: layout/table extraction unavailable ({type(exc).__name__}).")
            logger.warning("pdfplumber could not open %s", filename, exc_info=True)
    elif settings.extract_tables:
        warnings.append(f"{filename}: install pdfplumber to enable table extraction.")

    ocr_dependencies_ready = ocr_available()
    if settings.ocr_mode != "never" and ocr_dependencies_ready:
        try:
            fitz_doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            warnings.append(f"{filename}: OCR rendering unavailable ({type(exc).__name__}).")
            logger.warning("PyMuPDF could not open %s for OCR", filename, exc_info=True)
            fitz_doc = None
    elif settings.ocr_mode != "never":
        warnings.append(
            f"{filename}: OCR dependencies missing; install PyMuPDF, Pillow and pytesseract, "
            "plus the Tesseract system package."
        )

    page_records: list[dict[str, Any]] = []
    ocr_pages: list[int] = []
    table_count = 0

    try:
        for page_index in range(total_pages):
            page_number = page_index + 1
            native_text = _extract_native_text(reader, plumber_doc, page_index)
            native_quality = _text_quality(native_text)
            should_ocr = settings.ocr_mode == "always" or (
                settings.ocr_mode == "auto" and native_quality < settings.ocr_min_native_chars
            )

            ocr_text = ""
            if should_ocr and fitz_doc is not None:
                try:
                    ocr_text = _ocr_page(fitz_doc, page_index)
                    if ocr_text:
                        ocr_pages.append(page_number)
                except Exception as exc:
                    warnings.append(
                        f"{filename}: OCR failed on page {page_number} ({type(exc).__name__})."
                    )
                    logger.warning("OCR failed for %s page %s", filename, page_number, exc_info=True)

            best_text = _choose_text(native_text, ocr_text, force_ocr=settings.ocr_mode == "always")
            tables: list[str] = []
            if settings.extract_tables and plumber_doc is not None:
                try:
                    tables = _extract_tables(plumber_doc.pages[page_index], settings.table_min_rows)
                except Exception as exc:
                    warnings.append(
                        f"{filename}: table extraction failed on page {page_number} "
                        f"({type(exc).__name__})."
                    )
                    logger.warning(
                        "Table extraction failed for %s page %s", filename, page_number, exc_info=True
                    )

            table_count += len(tables)
            page_records.append(
                {
                    "page_number": page_number,
                    "text": best_text,
                    "tables": tables,
                }
            )
    finally:
        if plumber_doc is not None:
            plumber_doc.close()
        if fitz_doc is not None:
            fitz_doc.close()

    cleaned_pages = _remove_repeated_margin_lines(
        [str(record["text"]) for record in page_records]
    )

    blocks: list[PageText] = []
    for record, cleaned_text in zip(page_records, cleaned_pages, strict=True):
        page_number = int(record["page_number"])
        if cleaned_text:
            blocks.append(
                PageText(
                    filename=filename,
                    page_number=page_number,
                    text=cleaned_text,
                    content_type="text",
                )
            )
        for table_index, table_text in enumerate(record["tables"], start=1):
            blocks.append(
                PageText(
                    filename=filename,
                    page_number=page_number,
                    text=f"Table {table_index} on page {page_number}\n\n{table_text}",
                    content_type="table",
                    table_index=table_index,
                )
            )

    if not blocks:
        if settings.ocr_mode == "never":
            raise PdfProcessingError(
                f"{filename}: no extractable text found. Enable OCR_MODE=auto for scanned PDFs."
            )
        if not ocr_dependencies_ready:
            raise PdfProcessingError(
                f"{filename}: no extractable text found and OCR dependencies are unavailable."
            )
        raise PdfProcessingError(f"{filename}: no readable text or tables could be extracted")

    return PdfExtractionResult(
        blocks=blocks,
        total_pages=total_pages,
        ocr_pages=ocr_pages,
        table_count=table_count,
        warnings=_deduplicate(warnings),
    )


def _extract_native_text(reader: PdfReader, plumber_doc: Any, page_index: int) -> str:
    raw_text = ""
    if plumber_doc is not None:
        try:
            raw_text = plumber_doc.pages[page_index].extract_text(
                layout=True,
                x_tolerance=2,
                y_tolerance=3,
            ) or ""
        except Exception:
            logger.debug("Layout extraction failed; falling back to pypdf", exc_info=True)

    if not raw_text:
        page = reader.pages[page_index]
        try:
            raw_text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:  # Older pypdf versions.
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise PdfProcessingError(f"failed to extract text from page {page_index + 1}") from exc

    return _clean_text(raw_text)


def _ocr_page(document: Any, page_index: int) -> str:
    settings = get_settings()
    page = document.load_page(page_index)
    zoom = settings.ocr_dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    text = pytesseract.image_to_string(
        image,
        lang=settings.ocr_languages,
        config="--oem 3 --psm 6",
    )
    return _clean_text(text)


def _extract_tables(page: Any, min_rows: int) -> list[str]:
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 5,
    }
    tables = page.extract_tables(table_settings=settings) or []
    if not tables:
        # Text-based fallback catches borderless tables.
        text_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
            "intersection_tolerance": 5,
        }
        tables = page.extract_tables(table_settings=text_settings) or []

    rendered: list[str] = []
    seen: set[str] = set()
    for table in tables:
        normalized_rows = _normalize_table_rows(table)
        if len(normalized_rows) < min_rows or max((len(row) for row in normalized_rows), default=0) < 2:
            continue
        markdown = _table_to_markdown(normalized_rows)
        signature = re.sub(r"\s+", " ", markdown).strip().lower()
        if signature and signature not in seen:
            seen.add(signature)
            rendered.append(markdown)
    return rendered


def _normalize_table_rows(table: list[list[Any] | None]) -> list[list[str]]:
    rows: list[list[str]] = []
    width = max((len(row or []) for row in table), default=0)
    for raw_row in table:
        if raw_row is None:
            continue
        row = [_clean_cell(value) for value in raw_row]
        row.extend([""] * (width - len(row)))
        if any(cell for cell in row):
            rows.append(row)
    return rows


def _table_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    first = padded[0]
    meaningful_header = sum(bool(cell) for cell in first) >= max(1, width // 2)
    header = first if meaningful_header else [f"Column {index}" for index in range(1, width + 1)]
    body = padded[1:] if meaningful_header else padded

    lines = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |" for row in body
    )
    return "\n".join(lines)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _choose_text(native_text: str, ocr_text: str, force_ocr: bool) -> str:
    if force_ocr and ocr_text:
        return ocr_text
    if _text_quality(ocr_text) > _text_quality(native_text) * 1.15:
        return ocr_text
    return native_text or ocr_text


def _text_quality(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _remove_repeated_margin_lines(page_texts: list[str]) -> list[str]:
    if len(page_texts) < 3:
        return page_texts

    candidate_counts: Counter[str] = Counter()
    candidates_by_page: list[set[str]] = []
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidates = set(lines[:3] + lines[-3:])
        normalized = {_margin_signature(line) for line in candidates if _is_margin_candidate(line)}
        normalized.discard("")
        candidates_by_page.append(normalized)
        candidate_counts.update(normalized)

    threshold = max(3, int(len(page_texts) * 0.6 + 0.5))
    repeated = {signature for signature, count in candidate_counts.items() if count >= threshold}
    if not repeated:
        return page_texts

    cleaned_pages: list[str] = []
    for text in page_texts:
        kept = [
            line
            for line in text.splitlines()
            if _margin_signature(line.strip()) not in repeated or not _is_margin_candidate(line.strip())
        ]
        cleaned_pages.append(_clean_text("\n".join(kept)))
    return cleaned_pages


def _is_margin_candidate(line: str) -> bool:
    return 1 <= len(line) <= 120


def _margin_signature(line: str) -> str:
    normalized = re.sub(r"\d+", "#", line.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
