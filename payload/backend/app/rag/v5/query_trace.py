from __future__ import annotations

import argparse

from app.db import SessionLocal
from app.rag.smart_understanding import interpret_user_message
from app.rag.v5.retrieval_completeness import complete_terminology_hints
from app.rag.v5.synthesis_retrieval import retrieve_assistant_v52, review_bundle_coverage, response_synthesis_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect v5.2.1 retrieval decisions for one question.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    with SessionLocal() as db:
        hints = complete_terminology_hints(db, args.question, limit=80)
        interpretation = interpret_user_message(
            args.question,
            history=[],
            abbreviation_hints=hints,
            routing_hints=[],
        )
        bundle = retrieve_assistant_v52(
            db,
            interpretation,
            original_question=args.question,
        )
        review = review_bundle_coverage(db, interpretation, bundle)
        metadata = response_synthesis_metadata(bundle, review)

    print("IMS RAG v5.2.1 retrieval trace")
    print("================================")
    print(f"question: {args.question}")
    print(f"resolved: {interpretation.resolved_question}")
    print(f"intent: {interpretation.intent}")
    print(f"answer_strategy: {metadata.get('answer_strategy')}")
    print(f"coverage: {metadata.get('evidence_coverage_status')}")
    print(f"search_round: {bundle.search_round}")
    print(f"terminology_hints: {hints}")
    print(f"summary: {metadata.get('retrieval_diagnostic_summary')}")
    print()
    print("document decisions:")
    rows = list(metadata.get("retrieval_diagnostics") or [])[: max(1, args.limit)]
    for index, row in enumerate(rows, 1):
        print(
            f"{index:03d}. {row.get('filename')} | decision={row.get('decision')} | "
            f"score={row.get('discovery_score')} | vector={row.get('vector_score')} | "
            f"lexical={row.get('keyword_score')} | dimensions={row.get('dimension_hits')} | "
            f"routed={row.get('routed')} | deep={row.get('deep_searched')} | "
            f"role={row.get('rerank_role')} | contributing={row.get('contributing')}"
        )
        print(f"     reason: {row.get('reason')}")
        if row.get("best_heading") or row.get("best_page"):
            print(f"     best: {row.get('best_heading') or '-'} p.{row.get('best_page') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
