from io import BytesIO

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

from app.rag.chunking import chunk_pages
from app.rag.pdf import extract_pdf_pages


def _digital_table_pdf() -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    table = Table([["Quarter", "Revenue"], ["Q1", "100"], ["Q2", "125"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    document.build([Paragraph("Financial Summary", styles["Heading1"]), table])
    return buffer.getvalue()


def test_extracts_digital_table_as_markdown() -> None:
    result = extract_pdf_pages(_digital_table_pdf(), "financials.pdf")
    table_blocks = [block for block in result.blocks if block.content_type == "table"]

    assert result.total_pages == 1
    assert result.table_count >= 1
    assert table_blocks
    assert "| Quarter | Revenue |" in table_blocks[0].text
    assert "| Q2 | 125 |" in table_blocks[0].text


def test_table_chunker_repeats_header_and_keeps_rows() -> None:
    result = extract_pdf_pages(_digital_table_pdf(), "financials.pdf")
    chunks = chunk_pages(result.blocks, chunk_size=75, overlap=10)
    table_chunks = [chunk for chunk in chunks if chunk.content_type == "table"]

    assert table_chunks
    assert all("| Quarter | Revenue |" in chunk.text for chunk in table_chunks)
    assert any("| Q2 | 125 |" in chunk.text for chunk in table_chunks)


def test_ocr_fallback_reads_image_only_pdf(monkeypatch) -> None:
    monkeypatch.setenv("OCR_MODE", "auto")
    from app.config import get_settings

    get_settings.cache_clear()
    image = Image.new("RGB", (1200, 500), "white")
    drawing = ImageDraw.Draw(image)
    drawing.text((60, 80), "Invoice Total: 4250 USD", fill="black")
    drawing.text((60, 150), "Payment Due: 30 September 2026", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PDF", resolution=150)

    result = extract_pdf_pages(buffer.getvalue(), "scan.pdf")
    extracted = "\n".join(block.text for block in result.blocks)

    assert result.ocr_pages == [1]
    assert "Invoice Total" in extracted
    assert "4250" in extracted.replace(",", "")
    get_settings.cache_clear()
