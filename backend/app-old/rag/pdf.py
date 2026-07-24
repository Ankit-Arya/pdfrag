import io
import re
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.rag.types import PageText


class PdfProcessingError(ValueError):
    """Raised when an uploaded PDF cannot be safely processed."""


def safe_filename(filename: str | None) -> str:
    name = PurePath(filename or "document.pdf").name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return name[:160] or "document.pdf"


def extract_pdf_pages(data: bytes, filename: str) -> list[PageText]:
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

    pages: list[PageText] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise PdfProcessingError(
                f"{filename}: failed to extract text from page {page_number}"
            ) from exc
        text = _clean_text(raw_text)
        if text:
            pages.append(PageText(filename=filename, page_number=page_number, text=text))

    if not pages:
        raise PdfProcessingError(
            f"{filename}: no extractable text found. Scanned PDFs require an OCR pipeline."
        )
    return pages


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
