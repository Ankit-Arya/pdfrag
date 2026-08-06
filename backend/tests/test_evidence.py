from app.rag.evidence import build_evidence_answer
from app.rag.types import PromptSource, RetrievedChunk, TextChunk


def _source(chunk_id: str, page: int, section: str, body: str) -> PromptSource:
    text = (
        "[PDF CHUNK CONTEXT]\n"
        "File: 02. MRGR 2020.pdf\n"
        f"Pages: {page}\n"
        f"Section path: {section}\n"
        "Content type: text\n"
        "[/PDF CHUNK CONTEXT]\n\n"
        f"{body}"
    )
    result = RetrievedChunk(
        TextChunk(chunk_id, "02. MRGR 2020.pdf", page, text),
        0.9,
    )
    return PromptSource(result=result, excerpt=text)


def test_evidence_answer_includes_multiple_relevant_chunks_from_same_document() -> None:
    answer = build_evidence_answer(
        [
            _source(
                "definition",
                78,
                "PRELIMINARY",
                '“wake-up process” means process initiated by UTMS.',
            ),
            _source(
                "examination",
                128,
                "Examination of trains",
                "(1)\nUTMS shall perform wake-up test before passenger service.\n"
                "(2)\nWake-up test examination shall check cab signalling and brakes.",
            ),
        ]
    )

    assert "02. MRGR 2020.pdf — page 78" in answer
    assert "02. MRGR 2020.pdf — page 128" in answer
    assert "**Heading/subheading:** PRELIMINARY" in answer
    assert "**Heading/subheading:** Examination of trains" in answer
    assert "UTMS shall perform wake-up test before passenger service. [S2]" in answer
    assert "Wake-up test examination shall check cab signalling and brakes. [S2]" in answer
    assert "No further procedural steps" not in answer
