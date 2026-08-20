from app.rag.v5.layout import _Line, _geometry_tables, _merge_multipage_tables
from app.rag.v5.types import V5Word


def _line(index: int, y: float, cells: list[tuple[float, str]]) -> _Line:
    words = []
    for word_index, (x, text) in enumerate(cells):
        words.append(V5Word(text=text, x0=x, y0=y, x1=x + max(12, len(text) * 5), y1=y + 10, block=index, line=0))
    return _Line(
        index=index,
        words=words,
        text=" ".join(word.text for word in words),
        bbox=(min(word.x0 for word in words), y, max(word.x1 for word in words), y + 10),
        font_size=10,
        bold=False,
        source="native",
    )


def test_aligned_geometry_recovers_category_value_rows() -> None:
    lines = [
        _line(0, 50, [(210, "Nature of Injury"), (470, "Amount of Compensation")]),
        _line(1, 65, [(485, "(in Rs.)")]),
        _line(2, 90, [(70, "32"), (90, "Fracture of Pelvis not involving joint"), (490, "80,000")]),
        _line(3, 108, [(70, "33."), (90, "Fracture of Major Bone-Femur, Tibia of one limb"), (490, "80,000")]),
        _line(4, 126, [(70, "34."), (90, "Fracture of Major Bone-Humerus, Radius Ulna of one limb"), (490, "64,000")]),
        _line(5, 144, [(70, "35."), (90, "Another scheduled injury"), (490, "40,000")]),
    ]
    tables = _geometry_tables(lines, page_number=10, page_width=602, page_height=842)
    assert tables
    text = "\n".join(" | ".join(row.cells) for row in tables[0].rows)
    assert "Femur" in text
    assert "Tibia" in text
    assert "80,000" in text
    assert len(tables[0].columns) >= 2


def test_multipage_table_preserves_row_relationships() -> None:
    page1 = _geometry_tables(
        [
            _line(0, 50, [(210, "Nature of Injury"), (470, "Amount")]),
            _line(1, 80, [(70, "1."), (90, "First injury"), (490, "8,00,000")]),
            _line(2, 100, [(70, "2."), (90, "Second injury"), (490, "7,20,000")]),
            _line(3, 120, [(70, "3."), (90, "Third injury"), (490, "6,40,000")]),
        ], 8, 602, 842,
    )[0]
    page2 = _geometry_tables(
        [
            _line(0, 60, [(70, "4."), (90, "Fourth injury"), (490, "5,60,000")]),
            _line(1, 80, [(70, "5."), (90, "Fifth injury"), (490, "4,80,000")]),
            _line(2, 100, [(70, "6."), (90, "Sixth injury"), (490, "4,00,000")]),
        ], 9, 602, 842,
    )[0]
    merged = _merge_multipage_tables([page1, page2])
    assert len(merged) == 1
    assert merged[0].page_start == 8
    assert merged[0].page_end == 9
    assert len(merged[0].rows) >= 6


def test_numbered_rule_paragraphs_are_not_misclassified_as_table() -> None:
    lines = [
        _line(0, 50, [(60, "52."), (96, "Unusual occurrences --")]),
        _line(1, 75, [(94, "(1)"), (130, "Authorised persons shall know the location and use of fire fighting equipment at their place of work.")]),
        _line(2, 95, [(94, "(2)"), (130, "Authorised persons observing smoke or fire shall raise the alarm and inform the controller.")]),
        _line(3, 115, [(94, "(3)"), (130, "The Train Operator shall report smoke or fire between stations and follow the applicable procedure.")]),
    ]
    assert _geometry_tables(lines, page_number=109, page_width=602, page_height=842) == []


def test_standalone_rule_number_is_joined_with_same_line_heading() -> None:
    from app.rag.v5.layout import _lines_to_elements

    lines = [
        _line(0, 50, [(60, "52.")]),
        _line(1, 49, [(96, "Unusual occurrences --")]),
        _line(2, 75, [(94, "(1)"), (130, "Authorised persons shall know the applicable emergency instructions.")]),
    ]
    elements = _lines_to_elements(lines, page_number=109, table_boxes=[])
    assert any(element.element_type == "heading" and element.text.startswith("52. Unusual occurrences") for element in elements)
