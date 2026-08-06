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
from app.rag.normalization import search_terms
from app.rag.table_quality import is_plausible_table
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
            warnings.append(
                f"{filename}: layout/table extraction unavailable ({type(exc).__name__})."
            )
            logger.warning("pdfplumber could not open %s", filename, exc_info=True)
    elif settings.extract_tables:
        warnings.append(f"{filename}: install pdfplumber to enable table extraction.")

    ocr_dependencies_ready = ocr_available()
    if fitz is not None:
        try:
            fitz_doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            warnings.append(
                f"{filename}: PyMuPDF text/OCR rendering unavailable ({type(exc).__name__})."
            )
            logger.warning("PyMuPDF could not open %s", filename, exc_info=True)
            fitz_doc = None
    if settings.ocr_mode != "never" and not ocr_dependencies_ready:
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
            rotation_ratio = _page_rotation_ratio(
                plumber_doc.pages[page_index] if plumber_doc is not None else None,
                fitz_doc.load_page(page_index) if fitz_doc is not None else None,
            )
            rotated_text_detected = rotation_ratio >= settings.ocr_rotated_text_threshold
            native_text, native_consensus = _extract_native_text(
                reader,
                plumber_doc,
                fitz_doc,
                page_index,
            )
            corruption_score = _native_corruption_score(native_text)
            corrupted_text_detected = corruption_score >= settings.ocr_corruption_threshold
            force_ocr = rotated_text_detected or corrupted_text_detected
            verify_native = (
                settings.ocr_mode == "auto" and settings.ocr_verify_all_pages and not force_ocr
            )
            should_ocr = settings.ocr_mode == "always" or (
                settings.ocr_mode == "auto"
                and (
                    force_ocr
                    or _needs_ocr(
                        native_text,
                        settings.ocr_min_native_chars,
                        settings.ocr_min_text_quality,
                        consensus=native_consensus,
                        min_consensus=settings.ocr_min_native_consensus,
                    )
                )
            )

            ocr_text = ""
            completeness_ocr_detected = False
            if should_ocr and fitz_doc is not None and ocr_dependencies_ready:
                try:
                    ocr_text = _ocr_page(
                        fitz_doc,
                        page_index,
                        try_rotations=force_ocr,
                    )
                except Exception as exc:
                    warnings.append(
                        f"{filename}: OCR failed on page {page_number} ({type(exc).__name__})."
                    )
                    logger.warning(
                        "OCR failed for %s page %s", filename, page_number, exc_info=True
                    )

            if verify_native and not should_ocr and fitz_doc is not None and ocr_dependencies_ready:
                try:
                    verification_text = _ocr_page(
                        fitz_doc,
                        page_index,
                        dpi=settings.ocr_verify_dpi,
                        use_fallback=False,
                    )
                    if _ocr_recovers_missing_content(
                        native_text,
                        verification_text,
                        min_novel_terms=settings.ocr_verify_min_novel_terms,
                        novelty_threshold=settings.ocr_verify_novelty_threshold,
                    ):
                        ocr_text = _ocr_page(fitz_doc, page_index)
                        completeness_ocr_detected = _ocr_recovers_missing_content(
                            native_text,
                            ocr_text,
                            min_novel_terms=settings.ocr_verify_min_novel_terms,
                            novelty_threshold=settings.ocr_verify_novelty_threshold,
                        )
                except Exception as exc:
                    warnings.append(
                        f"{filename}: OCR verification failed on page {page_number} "
                        f"({type(exc).__name__})."
                    )
                    logger.warning(
                        "OCR verification failed for %s page %s",
                        filename,
                        page_number,
                        exc_info=True,
                    )

            best_text = _choose_text(
                native_text,
                ocr_text,
                force_ocr=(settings.ocr_mode == "always" or force_ocr),
            )
            if completeness_ocr_detected and ocr_text:
                best_text = ocr_text
            if ocr_text and best_text == ocr_text:
                ocr_pages.append(page_number)
                if rotated_text_detected:
                    logger.info(
                        "Rotation-aware OCR selected for %s page %s",
                        filename,
                        page_number,
                    )
                    warnings.append(f"{filename}: rotation-aware OCR used on page {page_number}.")
                elif corrupted_text_detected:
                    logger.info(
                        "Corruption-aware OCR selected for %s page %s (score %.3f)",
                        filename,
                        page_number,
                        corruption_score,
                    )
                    warnings.append(f"{filename}: corruption-aware OCR used on page {page_number}.")
                elif completeness_ocr_detected:
                    logger.info(
                        "Completeness OCR selected for %s page %s",
                        filename,
                        page_number,
                    )
                    warnings.append(f"{filename}: completeness OCR used on page {page_number}.")
            tables: list[str] = []
            if settings.extract_tables and plumber_doc is not None and not corrupted_text_detected:
                try:
                    tables = _extract_tables(
                        plumber_doc.pages[page_index],
                        settings.table_min_rows,
                    )
                except Exception as exc:
                    warnings.append(
                        f"{filename}: table extraction failed on page {page_number} "
                        f"({type(exc).__name__})."
                    )
                    logger.warning(
                        "Table extraction failed for %s page %s",
                        filename,
                        page_number,
                        exc_info=True,
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

    cleaned_pages = _remove_repeated_margin_lines([str(record["text"]) for record in page_records])

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


def _extract_native_text(
    reader: PdfReader,
    plumber_doc: Any,
    fitz_doc: Any,
    page_index: int,
) -> tuple[str, float]:
    candidates: list[str] = []
    if plumber_doc is not None:
        try:
            layout_text = (
                plumber_doc.pages[page_index].extract_text(
                    layout=True,
                    x_tolerance=2,
                    y_tolerance=3,
                )
                or ""
            )
            candidates.append(_clean_text(layout_text))
        except Exception:
            logger.debug("pdfplumber layout extraction failed", exc_info=True)
        try:
            flow_text = (
                plumber_doc.pages[page_index].extract_text(
                    layout=False,
                    x_tolerance=2,
                    y_tolerance=3,
                )
                or ""
            )
            candidates.append(_clean_text(flow_text))
        except Exception:
            logger.debug("pdfplumber flow extraction failed", exc_info=True)

    page = reader.pages[page_index]
    try:
        pypdf_text = page.extract_text(extraction_mode="layout") or ""
        candidates.append(_clean_text(pypdf_text))
    except TypeError:  # Older pypdf versions.
        try:
            candidates.append(_clean_text(page.extract_text() or ""))
        except Exception:
            logger.debug("pypdf extraction failed", exc_info=True)
    except Exception:
        logger.debug("pypdf extraction failed", exc_info=True)

    if fitz_doc is not None:
        try:
            fitz_text = fitz_doc.load_page(page_index).get_text("text", sort=True) or ""
            candidates.append(_clean_text(fitz_text))
        except Exception:
            logger.debug("PyMuPDF native extraction failed", exc_info=True)

    chosen = _choose_best_native_text(candidates)
    return chosen, _candidate_consensus(chosen, candidates)


def _ocr_page(
    document: Any,
    page_index: int,
    *,
    try_rotations: bool = False,
    dpi: int | None = None,
    use_fallback: bool = True,
) -> str:
    settings = get_settings()
    page = document.load_page(page_index)
    zoom = (dpi or settings.ocr_dpi) / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    primary_candidates: list[tuple[str, Any]] = []
    for angle in _ocr_orientations(try_rotations):
        oriented = image if angle == 0 else image.rotate(angle, expand=True, fillcolor="white")
        primary_candidates.append(
            (
                _ocr_image(
                    oriented,
                    settings.ocr_languages,
                    settings.ocr_primary_psm,
                ),
                oriented,
            )
        )

    primary, best_image = max(
        primary_candidates,
        key=lambda item: _text_quality(item[0]),
    )
    candidates = [primary]
    if (
        use_fallback
        and settings.ocr_fallback_psm != settings.ocr_primary_psm
        and _text_quality(primary) < max(settings.ocr_min_text_quality, 0.68)
    ):
        candidates.append(
            _ocr_image(
                best_image,
                settings.ocr_languages,
                settings.ocr_fallback_psm,
            )
        )
    return _choose_best_native_text(candidates)


def _ocr_orientations(try_rotations: bool) -> tuple[int, ...]:
    return (0, 90, 180, 270) if try_rotations else (0,)


def _page_rotation_ratio(plumber_page: Any, fitz_page: Any) -> float:
    """Measure non-horizontal text using two independent layout engines."""
    ratios: list[float] = []

    if plumber_page is not None:
        try:
            total = 0
            rotated = 0
            for character in plumber_page.chars:
                weight = max(1, len(str(character.get("text", ""))))
                total += weight
                if character.get("upright") is False:
                    rotated += weight
            if total >= 12:
                ratios.append(rotated / total)
        except Exception:
            logger.debug("pdfplumber rotation inspection failed", exc_info=True)

    if fitz_page is not None:
        try:
            total = 0
            rotated = 0
            layout = fitz_page.get_text("dict") or {}
            for block in layout.get("blocks", []):
                for line in block.get("lines", []):
                    direction = line.get("dir", (1.0, 0.0))
                    line_length = sum(
                        len(str(span.get("text", ""))) for span in line.get("spans", [])
                    )
                    total += line_length
                    if len(direction) >= 2 and (
                        abs(float(direction[0]) - 1.0) > 0.15 or abs(float(direction[1])) > 0.15
                    ):
                        rotated += line_length
            if total >= 12:
                ratios.append(rotated / total)
        except Exception:
            logger.debug("PyMuPDF rotation inspection failed", exc_info=True)

    return max(ratios, default=0.0)


def _ocr_image(image: Any, languages: str, psm: int) -> str:
    text = pytesseract.image_to_string(
        image,
        lang=languages,
        config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1",
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
    border_based = bool(tables)
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
        border_based = False

    rendered: list[str] = []
    seen: set[str] = set()
    for table in tables:
        normalized_rows = _normalize_table_rows(table)
        if (
            len(normalized_rows) < min_rows
            or max((len(row) for row in normalized_rows), default=0) < 2
        ):
            continue
        if not is_plausible_table(normalized_rows, border_based=border_based):
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
    if not native_text:
        return ocr_text
    if _text_quality(ocr_text) >= _text_quality(native_text) + 0.04:
        return ocr_text
    return native_text or ocr_text


def _choose_best_native_text(candidates: list[str]) -> str:
    nonempty = [candidate for candidate in candidates if candidate.strip()]
    if not nonempty:
        return ""
    return max(
        nonempty,
        key=lambda value: (
            _text_quality(value) * 0.72 + _candidate_consensus(value, nonempty) * 0.28,
            min(_alnum_count(value), 5000),
        ),
    )


def _needs_ocr(
    text: str,
    min_chars: int,
    min_quality: float,
    *,
    consensus: float = 1.0,
    min_consensus: float = 0.0,
) -> bool:
    return (
        _alnum_count(text) < min_chars
        or _text_quality(text) < min_quality
        or consensus < min_consensus
    )


def _native_corruption_score(text: str) -> float:
    """Estimate whether a native PDF text layer contains glyph-order garbage.

    Some PDFs expose a non-empty text layer made from rotated form cells or
    incorrectly mapped glyphs. Multiple native extractors reproduce the same bad
    layer, so normal quality and consensus checks can both pass. These pages are
    characterised by several dense lines dominated by repeated one/two-character
    fragments, for example ``tt tT | TT TT | rT``.

    Requiring multiple suspicious lines keeps legitimate acronyms, numbered list
    markers and compact two-column tables from forcing OCR by themselves.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    total_tokens = 0
    suspicious_tokens = 0
    suspicious_lines = 0
    for line in lines:
        tokens = re.findall(r"[A-Za-z0-9]+", line)
        total_tokens += len(tokens)
        if len(tokens) < 6:
            continue

        folded = [token.casefold() for token in tokens]
        short_ratio = sum(len(token) <= 2 for token in tokens) / len(tokens)
        repeated_ratio = 1.0 - len(set(folded)) / len(folded)
        pipe_count = line.count("|")
        looks_fragmented = (
            short_ratio >= 0.68
            or (short_ratio >= 0.52 and repeated_ratio >= 0.34)
            or (pipe_count >= 4 and short_ratio >= 0.55)
        )
        if looks_fragmented:
            suspicious_lines += 1
            suspicious_tokens += len(tokens)

    if suspicious_lines < 3 or total_tokens == 0:
        return 0.0

    token_share = suspicious_tokens / total_tokens
    line_share = suspicious_lines / len(lines)
    return min(1.0, token_share * 0.7 + line_share * 0.3)


def _ocr_recovers_missing_content(
    native_text: str,
    ocr_text: str,
    *,
    min_novel_terms: int,
    novelty_threshold: float,
) -> bool:
    """Detect visible OCR content absent from a plausible native text layer."""
    if not native_text.strip() or not ocr_text.strip():
        return bool(ocr_text.strip() and not native_text.strip())
    if _text_quality(ocr_text) < max(0.55, _text_quality(native_text) - 0.12):
        return False

    native_terms = _meaningful_content_terms(native_text)
    ocr_terms = _meaningful_content_terms(ocr_text)
    if len(ocr_terms) < min_novel_terms:
        return False
    novel_terms = ocr_terms - native_terms
    return (
        len(novel_terms) >= min_novel_terms
        and len(novel_terms) / len(ocr_terms) >= novelty_threshold
    )


def _meaningful_content_terms(value: str) -> set[str]:
    return {term for term in search_terms(value) if len(term) >= 3 and not term.isdigit()}


def _candidate_consensus(chosen: str, candidates: list[str]) -> float:
    chosen_tokens = _quality_tokens(chosen)
    peers = [candidate for candidate in candidates if candidate.strip()]
    if not chosen_tokens or len(peers) <= 1:
        return 1.0

    similarities: list[float] = []
    skipped_chosen = False
    for candidate in peers:
        if candidate == chosen and not skipped_chosen:
            skipped_chosen = True
            continue
        candidate_tokens = _quality_tokens(candidate)
        if not candidate_tokens:
            continue
        similarities.append(
            len(chosen_tokens & candidate_tokens) / len(chosen_tokens | candidate_tokens)
        )
    return sum(similarities) / len(similarities) if similarities else 1.0


def _quality_tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 1}


def _alnum_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _text_quality(text: str) -> float:
    if not text.strip():
        return 0.0

    nonspace = [character for character in text if not character.isspace()]
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonspace or not tokens or not lines:
        return 0.0

    length_score = min(1.0, _alnum_count(text) / 220)
    readable_character_ratio = sum(
        character.isalnum() or character in ".,;:!?()[]{}'\"/%&+-–—" for character in nonspace
    ) / len(nonspace)
    word_ratio = sum(len(token) >= 2 for token in tokens) / len(tokens)
    short_lines = sum(
        len(re.findall(r"[A-Za-z0-9]+", line)) <= 1 and len(line) <= 3 for line in lines
    )
    line_score = 1.0 - short_lines / len(lines)
    control_penalty = sum(
        ord(character) < 32 and character not in "\n\t" for character in text
    ) / max(len(text), 1)

    score = (
        length_score * 0.22
        + readable_character_ratio * 0.28
        + word_ratio * 0.28
        + line_score * 0.22
        - min(0.35, control_penalty * 10)
    )
    return max(0.0, min(1.0, score))


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
            if _margin_signature(line.strip()) not in repeated
            or not _is_margin_candidate(line.strip())
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
