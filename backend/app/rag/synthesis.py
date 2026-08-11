from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.rag.llm import llm_service
from app.rag.prompts import NO_ANSWER, build_user_prompt
from app.rag.types import PromptSource, QueryPlan, RetrievedChunk


_SUMMARY_SYSTEM_PROMPT = """You are an evidence compressor for official internal metro documents.
You are not answering from memory. Use only the supplied source chunks/digests.

Your job is to preserve EVERY fact that directly answers or materially qualifies the stated
question while removing unrelated text and duplicate wording. Preserve exact source labels
such as [S17]. Every factual bullet must carry one or more existing labels.

Never merge incompatible rolling stock, line, mode, system, equipment, procedure, location
or document contexts. Preserve procedural order, permissions, prerequisites, responsible
roles, warnings, prohibitions, exceptions, alternatives, verification/record requirements,
numbers, units, dates and limits. If a chunk only mentions a keyword without answering or
qualifying the question, omit it. Do not invent facts or source labels.
"""


@dataclass(slots=True)
class SynthesisBundle:
    raw_answer: str
    sources: list[PromptSource]
    used_hierarchy: bool = False
    digests: str = ""


def synthesize_answer(
    plan: QueryPlan,
    results: list[RetrievedChunk],
) -> SynthesisBundle:
    settings = get_settings()
    estimated_chars = sum(len(item.chunk.text) + len(item.chunk.filename) + 180 for item in results)

    if estimated_chars <= int(settings.max_context_chars * 0.85):
        prompt, included = build_user_prompt(
            plan.original_question,
            plan.contextual_question or plan.rewritten_question,
            results,
            settings.max_context_chars,
            question_intent=plan.intent,
            response_mode="concise",
        )
        typed_sources = [source for source in included if isinstance(source, PromptSource)]
        return SynthesisBundle(
            raw_answer=llm_service.answer(prompt),
            sources=typed_sources,
            used_hierarchy=False,
        )

    sources = _prompt_sources(results)
    digests = _summarize_all_sources(plan, sources)
    final_prompt = _build_digest_answer_prompt(plan, sources, digests)
    return SynthesisBundle(
        raw_answer=llm_service.answer(final_prompt),
        sources=sources,
        used_hierarchy=True,
        digests=digests,
    )


def repair_hierarchical_answer(
    plan: QueryPlan,
    previous_answer: str,
    sources: list[PromptSource],
    digests: str,
) -> str:
    source_map = _source_map(sources)
    prompt = f"""Rewrite the previous draft so it is a complete, grounded answer to the current
question. Use ONLY the evidence digests below. Keep all materially relevant facts, preserve
separate applicability contexts, and cite every factual bullet with valid labels [S1]...[S{len(sources)}].
Do not invent or renumber citations.

CURRENT QUESTION:
{plan.original_question}

CONTEXTUAL INTERPRETATION (intent only):
{plan.contextual_question or plan.rewritten_question}

PREVIOUS DRAFT:
{previous_answer}

SOURCE MAP:
{source_map}

EVIDENCE DIGESTS:
{digests}

If the digests contain no supported answer, reply exactly:
{NO_ANSWER}
"""
    return llm_service.answer(prompt)


def _prompt_sources(results: list[RetrievedChunk]) -> list[PromptSource]:
    ordered = sorted(
        results,
        key=lambda item: (
            item.chunk.filename.casefold(),
            item.chunk.page_number,
            item.chunk.chunk_index if item.chunk.chunk_index is not None else -1,
            item.chunk.chunk_id,
        ),
    )
    return [PromptSource(result=item, excerpt=item.chunk.text.strip()) for item in ordered]


def _summarize_all_sources(plan: QueryPlan, sources: list[PromptSource]) -> str:
    settings = get_settings()
    batches = _source_batches(sources, settings.summary_batch_chars)
    digests: list[str] = []
    for batch in batches:
        prompt = _build_batch_summary_prompt(plan, batch)
        digest = llm_service.summarize(
            _SUMMARY_SYSTEM_PROMPT,
            prompt,
            max_output_tokens=settings.summary_max_output_tokens,
        ).strip()
        if digest and digest != "NO_RELEVANT_EVIDENCE":
            digests.append(digest)

    if not digests:
        return "NO_RELEVANT_EVIDENCE"

    combined = "\n\n--- EVIDENCE BATCH ---\n\n".join(digests)
    # If the first compression pass is still too large, recursively compress the
    # digests themselves. Original [S#] labels are preserved through every level.
    target = int(settings.max_context_chars * 0.68)
    while len(combined) > target and len(digests) > 1:
        digest_batches = _text_batches(digests, settings.summary_batch_chars)
        next_digests: list[str] = []
        for batch_text in digest_batches:
            prompt = f"""Question:
{plan.original_question}

Contextual interpretation:
{plan.contextual_question or plan.rewritten_question}

Compress these already-grounded evidence digests again. Preserve every distinct fact that
answers or qualifies the question, every applicability distinction, and all existing [S#]
citations. Do not add facts or labels.

DIGESTS:
{batch_text}
"""
            digest = llm_service.summarize(
                _SUMMARY_SYSTEM_PROMPT,
                prompt,
                max_output_tokens=settings.summary_max_output_tokens,
            ).strip()
            if digest and digest != "NO_RELEVANT_EVIDENCE":
                next_digests.append(digest)
        if not next_digests or next_digests == digests:
            break
        digests = next_digests
        combined = "\n\n--- EVIDENCE BATCH ---\n\n".join(digests)

    return combined


def _source_batches(
    sources: list[PromptSource],
    max_chars: int,
) -> list[list[tuple[int, PromptSource]]]:
    batches: list[list[tuple[int, PromptSource]]] = []
    current: list[tuple[int, PromptSource]] = []
    used = 0
    for index, source in enumerate(sources, 1):
        cost = len(source.excerpt) + len(source.result.chunk.filename) + 220
        if current and used + cost > max_chars:
            batches.append(current)
            current = []
            used = 0
        current.append((index, source))
        used += cost
    if current:
        batches.append(current)
    return batches


def _text_batches(values: list[str], max_chars: int) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    used = 0
    for value in values:
        cost = len(value) + 80
        if current and used + cost > max_chars:
            result.append("\n\n---\n\n".join(current))
            current = []
            used = 0
        current.append(value)
        used += cost
    if current:
        result.append("\n\n---\n\n".join(current))
    return result


def _build_batch_summary_prompt(
    plan: QueryPlan,
    batch: list[tuple[int, PromptSource]],
) -> str:
    blocks: list[str] = []
    for index, source in batch:
        chunk = source.result.chunk
        blocks.append(
            f"[S{index}] File: {chunk.filename} | Page: {chunk.page_number} | "
            f"Type: {chunk.content_type} | Retrieval: {source.result.method}\n"
            f"{source.excerpt}"
        )
    return f"""QUESTION TO SUPPORT:
{plan.original_question}

CONTEXTUAL INTERPRETATION (intent only, not evidence):
{plan.contextual_question or plan.rewritten_question}

INTENT: {plan.intent}

Extract all directly relevant evidence from EVERY source chunk in this batch. A procedure may
span adjacent chunks; preserve its complete sequence when the chunks support it. Keep different
documents/lines/procedures separate. Output concise evidence bullets with original [S#] labels.
If nothing in this batch answers or materially qualifies the question, output exactly
NO_RELEVANT_EVIDENCE.

SOURCE CHUNKS:
{chr(10).join(blocks)}
"""


def _build_digest_answer_prompt(
    plan: QueryPlan,
    sources: list[PromptSource],
    digests: str,
) -> str:
    return f"""Produce the final answer to the ORIGINAL QUESTION using only the supplied evidence
digests. The digests were produced by exhaustively processing a larger set of relevant PDF
chunks and preserve their original source labels.

ORIGINAL QUESTION:
{plan.original_question}

CONTEXTUAL INTERPRETATION (conversation context is intent only, never factual evidence):
{plan.contextual_question or plan.rewritten_question}

QUESTION INTENT:
{plan.intent}

SOURCE MAP:
{_source_map(sources)}

SOURCE EXCERPTS (HIERARCHICAL EVIDENCE DIGESTS):
{digests}

Required answer behavior:
- Start with `## Information found in the documents`.
- Answer the exact question, not merely the search terms.
- Include every materially relevant fact preserved in the digests.
- Keep different documents, lines, rolling stock, systems, modes and procedures separate when
  their applicability differs; never blend incompatible procedures.
- For procedures, preserve prerequisites, permissions, chronological steps, warnings,
  exceptions, checks, records and responsible roles.
- Cite every factual bullet using the exact [S#] labels from the digests/source map.
- Do not use conversation history, the model's general knowledge, or uncited assumptions as facts.
- If no supported answer exists, reply exactly: {NO_ANSWER}
"""


def _source_map(sources: list[PromptSource]) -> str:
    lines: list[str] = []
    for index, source in enumerate(sources, 1):
        chunk = source.result.chunk
        lines.append(
            f"[S{index}] {chunk.filename} — page {chunk.page_number} — {chunk.content_type}"
        )
    return "\n".join(lines)
