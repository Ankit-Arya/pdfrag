from app.rag.v5.chunking import build_v5_chunks
from app.rag.v5.types import V5Element, V5LayoutDocument, V5Page, V5Table, V5TableRow


def test_table_row_chunk_inherits_table_schema_and_section() -> None:
    table = V5Table(
        table_id="11111111-1111-4111-8111-111111111111",
        table_key="claims-second-schedule",
        title="Second Schedule - Part III",
        page_start=10,
        page_end=10,
        columns=["No.", "Nature of Injury", "Amount of Compensation (Rs.)"],
        rows=[
            V5TableRow(
                row_index=33,
                page_number=10,
                cells=["33", "Fracture of Major Bone-Femur, Tibia of one limb", "80,000"],
                bbox=(70, 78, 520, 92),
            )
        ],
        section_path=["The Second Schedule", "PART III"],
    )
    doc = V5LayoutDocument(
        filename="claims.pdf",
        total_pages=10,
        pages=[V5Page(page_number=10, width=602, height=842)],
        elements=[],
        tables=[table],
    )
    chunks = build_v5_chunks(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content_type == "table_row"
    assert "Nature of Injury: Fracture of Major Bone-Femur, Tibia of one limb" in chunk.text
    assert "Amount of Compensation (Rs.): 80,000" in chunk.text
    assert "The Second Schedule > PART III" in chunk.text


def test_prose_chunks_preserve_heading_parent() -> None:
    elements = [
        V5Element(
            element_id="22222222-2222-4222-8222-222222222222", page_number=1, order_index=0,
            element_type="heading", text="51 Train divided", bbox=(70, 80, 250, 95),
            heading_level=2, parent_key="CHAPTER > 51 Train divided",
        ),
        V5Element(
            element_id="33333333-3333-4333-8333-333333333333", page_number=1, order_index=1,
            element_type="paragraph", text="The Train Operator shall verify train integrity and inform the Traffic Controller.",
            bbox=(70, 100, 520, 125), parent_key="CHAPTER > 51 Train divided",
            metadata={"section_path": ["CHAPTER", "51 Train divided"]},
        ),
    ]
    doc = V5LayoutDocument(
        filename="mrgr.pdf", total_pages=1,
        pages=[V5Page(page_number=1, width=602, height=842)], elements=elements, tables=[]
    )
    chunks = build_v5_chunks(doc)
    assert chunks
    assert "Section path: CHAPTER > 51 Train divided" in chunks[0].text
    assert "verify train integrity" in chunks[0].text
