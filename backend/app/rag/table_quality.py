from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def is_plausible_table(
    rows: list[list[str]],
    *,
    border_based: bool = False,
) -> bool:
    """Reject prose that a borderless-table detector split into fake columns."""
    normalized = _rectangular_rows(rows)
    if len(normalized) < 2:
        return False

    width = len(normalized[0])
    if width < 2:
        return False

    nonempty = [cell for row in normalized for cell in row if cell]
    if len(nonempty) < 3:
        return False

    boundary_count = 0
    broken_word_boundaries = 0
    long_prose_rows = 0
    numeric_cells = 0
    alpha_cells = 0

    for row in normalized:
        row_text = " ".join(cell for cell in row if cell)
        if len(row_text) >= 70:
            long_prose_rows += 1
        for cell in row:
            if not cell:
                continue
            if any(char.isdigit() for char in cell):
                numeric_cells += 1
            if any(char.isalpha() for char in cell):
                alpha_cells += 1
        for left, right in zip(row, row[1:], strict=False):
            if not left or not right:
                continue
            boundary_count += 1
            if left[-1:].islower() and right[:1].islower():
                broken_word_boundaries += 1

    fragmented_ratio = broken_word_boundaries / max(boundary_count, 1)
    numeric_ratio = numeric_cells / len(nonempty)
    alpha_ratio = alpha_cells / len(nonempty)
    prose_ratio = long_prose_rows / len(normalized)
    average_cell_length = sum(len(cell) for cell in nonempty) / len(nonempty)

    # Running text split at x-coordinates frequently cuts words between cells.
    # That is not a valid tabular relationship even if pdfplumber returns rows.
    fragmentation_limit = 0.34 if border_based else 0.22
    if width >= 3 and fragmented_ratio >= fragmentation_limit:
        return False

    # Borderless extraction of multi-column prose usually creates many narrow,
    # alphabetic columns and long sentence-like rows without structured values.
    if (
        not border_based
        and width >= 4
        and prose_ratio >= 0.4
        and numeric_ratio < 0.08
        and alpha_ratio >= 0.75
    ):
        return False
    return not (width >= 6 and average_cell_length < 18 and alpha_ratio >= 0.75)


def normalized_table_tokens(rows: list[list[str]]) -> set[str]:
    return {
        token.casefold()
        for row in rows
        for cell in row
        for token in _TOKEN_RE.findall(cell)
        if len(token) > 1
    }


def _rectangular_rows(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(row) for row in rows), default=0)
    return [
        [cell.strip() for cell in row] + [""] * (width - len(row))
        for row in rows
        if any(cell.strip() for cell in row)
    ]
