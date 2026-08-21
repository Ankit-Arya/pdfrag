from __future__ import annotations

import argparse

from app.db import SessionLocal
from app.rag.smart_understanding import interpret_user_message
from app.rag.v5.assistant_retrieval import assistant_terminology_hints, retrieve_assistant_v51


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect v5.1 assistant interpretation, document routing and evidence ranking.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    with SessionLocal() as db:
        hints = assistant_terminology_hints(db, args.question)
        interpretation = interpret_user_message(
            args.question,
            history=[],
            abbreviation_hints=hints,
            routing_hints=[],
        )
        bundle = retrieve_assistant_v51(
            db,
            interpretation,
            original_question=args.question,
        )

    print("RAG v5.1 assistant debug")
    print("------------------------")
    print(f"question: {args.question}")
    print(f"resolved: {interpretation.resolved_question}")
    print(f"intent: {interpretation.intent}")
    print(f"conversation_act: {interpretation.conversation_act}")
    print(f"concepts: {list(interpretation.concepts)}")
    print(f"evidence_needs: {list(interpretation.evidence_needs)}")
    print(f"terminology_hints: {hints}")
    print(f"search_queries: {bundle.search_queries}")
    print("routed_documents:")
    for index, filename in enumerate(bundle.routed_documents, 1):
        print(f"  {index:02d}. {filename}")
    print(f"candidate_count: {bundle.candidate_count}")
    print("top_evidence:")
    for index, item in enumerate(bundle.results[: max(1, min(80, args.limit))], 1):
        chunk = item.chunk
        pages = chunk.page_number if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
        section = " > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or "")
        excerpt = " ".join(chunk.text.split())[:260]
        print(f"  {index:02d}. score={item.score:.4f} file={chunk.filename} page={pages}")
        print(f"      section={section}")
        print(f"      method={item.method}")
        print(f"      text={excerpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
