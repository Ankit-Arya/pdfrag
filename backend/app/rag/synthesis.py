from __future__ import annotations

from dataclasses import dataclass
import re

from app.config import get_settings
from app.rag.llm import llm_service
from app.rag.prompts import NO_ANSWER
from app.rag.types import PromptSource, QueryPlan, RetrievedChunk


_ANSWER_SYSTEM_PROMPT = """You are a document-grounded assistant for official internal metro documents.
Use only the supplied PDF evidence. Conversation context may clarify intent but is never factual evidence.

Answer the user's exact question at the shortest length that is complete and safe:
- A simple fact, figure, date, name, speed, limit or yes/no lookup should normally be one direct sentence.
- A request for several facts should use a compact list.
- A procedure/safety question must give the actionable solution, not a vague summary. Preserve the operational order, responsible roles, prerequisites, notifications, restrictions, special modes, restoration criteria and important alternatives actually supported by the procedure.
- When the evidence identifies a dedicated primary SOP/instruction for the exact scenario, use that document as the backbone of the answer. Other PDFs may supplement it only when they add a directly applicable requirement; never replace the dedicated procedure with generic nearby safety guidance.
- Procedure formatting: start with a short applicability/scope sentence when material, then use a clear numbered sequence. If the source itself separates materially different obstruction/scenario types, use compact bold scenario labels beneath the main sequence. Do not dump raw evidence.
- Never dump or restate all source excerpts merely because they were retrieved; the UI exposes those separately.
- Preserve applicability: line, rolling stock, mode, equipment, procedure, location, conditions, warnings and exceptions.
- Never merge incompatible contexts. If the user's context is ambiguous and the evidence contains materially different answers, state the alternatives succinctly instead of inventing one.
- Cite each factual sentence/bullet with the supplied [S#] labels.
- Use no source label that is not supplied.
- Do not claim that the documents do not specify/define/provide something unless the supplied evidence genuinely fails to support the answer.
- If no supported answer exists, reply exactly with the configured no-answer sentence.
"""


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
    primary_document_ids: frozenset[str] = frozenset()
    primary_document_names: tuple[str, ...] = ()


def synthesize_answer(
    plan: QueryPlan,
    results: list[RetrievedChunk],
    *,
    primary_document_ids: set[str] | None = None,
    primary_document_names: list[str] | None = None,
) -> SynthesisBundle:
    settings = get_settings()
    estimated_chars = sum(len(item.chunk.text) + len(item.chunk.filename) + 180 for item in results)

    primary_ids = primary_document_ids or set()
    primary_names = primary_document_names or []
    sources = _prompt_sources(plan, results, primary_ids)
    if estimated_chars <= int(settings.max_context_chars * 0.85):
        prompt = _build_direct_answer_prompt(plan, sources, primary_names)
        return SynthesisBundle(
            raw_answer=_answer(prompt),
            sources=sources,
            used_hierarchy=False,
            primary_document_ids=frozenset(primary_ids),
            primary_document_names=tuple(primary_names),
        )

    digests = _summarize_all_sources(plan, sources, primary_ids)
    final_prompt = _build_digest_answer_prompt(plan, sources, digests, primary_names, primary_ids)
    return SynthesisBundle(
        raw_answer=_answer(final_prompt),
        sources=sources,
        used_hierarchy=True,
        digests=digests,
        primary_document_ids=frozenset(primary_ids),
        primary_document_names=tuple(primary_names),
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
    return _answer(prompt)


def repair_direct_answer(
    plan: QueryPlan,
    previous_answer: str,
    sources: list[PromptSource],
) -> str:
    prompt = f"""Repair the previous draft using ONLY the supplied source chunks. Keep the answer proportional to the question: a simple fact should remain one sentence; expand only when necessary. Preserve valid [S#] labels and remove unsupported claims.

ORIGINAL QUESTION:
{plan.original_question}

CONTEXTUAL INTERPRETATION (intent only):
{plan.contextual_question or plan.rewritten_question}

PREVIOUS DRAFT:
{previous_answer}

SOURCE CHUNKS:
{_source_blocks(sources)}

If no supported answer exists, reply exactly:
{NO_ANSWER}
"""
    return _answer(prompt)



def procedure_answer_needs_repair(
    answer: str,
    sources: list[PromptSource],
    primary_document_ids: set[str],
) -> bool:
    if not answer or not primary_document_ids:
        return False
    citation_numbers = {
        int(value)
        for value in re.findall(r"\[S(\d+)\]", answer)
        if value.isdigit()
    }
    cites_primary = any(
        1 <= number <= len(sources)
        and sources[number - 1].result.chunk.document_id in primary_document_ids
        for number in citation_numbers
    )
    has_numbered_steps = bool(re.search(r"(?m)^\s*\d+[.)]\s+", answer))
    return not cites_primary or not has_numbered_steps


def repair_procedure_answer(
    plan: QueryPlan,
    previous_answer: str,
    sources: list[PromptSource],
    primary_document_ids: set[str],
    *,
    max_primary_sources: int = 48,
) -> str:
    """Repair a vague/misformatted procedure from the dedicated SOP evidence."""
    if not primary_document_ids:
        return previous_answer
    ranked = [
        (index, source)
        for index, source in enumerate(sources, 1)
        if source.result.chunk.document_id in primary_document_ids
    ][:max_primary_sources]
    if not ranked:
        return previous_answer
    blocks: list[str] = []
    for index, source in ranked:
        chunk = source.result.chunk
        blocks.append(
            f"[S{index}] File: {chunk.filename} | Page: {chunk.page_number} | "
            f"Type: {chunk.content_type}\n{source.excerpt}"
        )
    prompt = f"""Rewrite the previous draft as the actionable procedure requested by the user.
Use ONLY the PRIMARY PROCEDURE evidence below. Do not replace it with generic rules from other
PDFs. Start with a short applicability/scope sentence only if supported, then give a numbered
operational sequence. Preserve responsible roles, notifications, restrictions, special modes,
assistance/escalation, restoration criteria, and materially different scenario branches. Keep
it concise but complete. Cite every factual step using the existing [S#] labels; do not renumber.

ORIGINAL QUESTION:
{plan.original_question}

PREVIOUS DRAFT:
{previous_answer}

PRIMARY PROCEDURE EVIDENCE:
{chr(10).join(blocks)}

If the primary evidence does not support an answer, reply exactly: {NO_ANSWER}
"""
    return _answer(prompt)


def rescue_fact_answer(
    plan: QueryPlan,
    sources: list[PromptSource],
    *,
    max_sources: int = 24,
) -> str:
    """Retry an unexpected no-answer using the strongest original source labels.

    The normal synthesis still reviews every selected chunk. This rescue is only
    invoked when that pass says no answer for a direct fact/definition despite
    having evidence. It keeps the same S-numbering, so citations remain valid
    against the complete reviewed-evidence list.
    """
    if plan.intent not in {"fact_lookup", "definition"} or not sources:
        return NO_ANSWER
    selected = list(enumerate(sources[:max_sources], 1))
    blocks: list[str] = []
    for index, source in selected:
        chunk = source.result.chunk
        blocks.append(
            f"[S{index}] File: {chunk.filename} | Page: {chunk.page_number} | "
            f"Type: {chunk.content_type}\n{source.excerpt}"
        )
    prompt = f"""The broad evidence pass unexpectedly returned no answer for a direct lookup.
Re-check only the strongest evidence below and answer the ORIGINAL QUESTION if it is supported.
Do not infer from filenames or conversation history. For a single fact, return one sentence.
Use only the existing [S#] labels.

ORIGINAL QUESTION:
{plan.original_question}

CONTEXTUAL INTERPRETATION (intent only):
{plan.contextual_question or plan.rewritten_question}

STRONGEST SOURCE EXCERPTS:
{chr(10).join(blocks)}

If these excerpts still do not support the answer, reply exactly: {NO_ANSWER}
"""
    return _answer(prompt)


_NEGATIVE_ANSWER_RE = re.compile(
    r"\b(?:documents?|pdfs?)\b.{0,80}\b(?:do not|does not|don't|doesn't|cannot|can't)\b.{0,80}\b(?:specify|define|provide|contain|state|mention|include)\b|"
    r"\b(?:not specified|not defined|not provided|not stated|not found|no information)\b",
    re.IGNORECASE | re.DOTALL,
)


def looks_like_negative_answer(value: str) -> bool:
    return bool(_NEGATIVE_ANSWER_RE.search(value or ""))


def _answer(prompt: str) -> str:
    settings = get_settings()
    return llm_service.generate(
        _ANSWER_SYSTEM_PROMPT,
        prompt,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
    )


def _prompt_sources(
    plan: QueryPlan,
    results: list[RetrievedChunk],
    primary_document_ids: set[str],
) -> list[PromptSource]:
    # Relevance selection already returns strongest-first evidence. Preserve that
    # order for simple facts/definitions so the direct answer is not buried deep
    # in a long document-sorted prompt. For procedures and broader synthesis,
    # document/page order still helps preserve sequence and applicability.
    if plan.intent in {"fact_lookup", "definition"}:
        ordered = list(results)
    elif primary_document_ids:
        ordered = sorted(
            results,
            key=lambda item: (
                0 if item.chunk.document_id in primary_document_ids else 1,
                item.chunk.filename.casefold(),
                item.chunk.page_number,
                item.chunk.chunk_index if item.chunk.chunk_index is not None else -1,
                item.chunk.chunk_id,
            ),
        )
    else:
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


def _summarize_all_sources(
    plan: QueryPlan,
    sources: list[PromptSource],
    primary_document_ids: set[str],
) -> str:
    settings = get_settings()
    batches = _source_batches(sources, settings.summary_batch_chars)
    digests: list[str] = []
    for batch in batches:
        prompt = _build_batch_summary_prompt(plan, batch, primary_document_ids)
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


def _build_direct_answer_prompt(
    plan: QueryPlan,
    sources: list[PromptSource],
    primary_document_names: list[str],
) -> str:
    return f"""Answer the ORIGINAL QUESTION using all supplied source chunks.

ORIGINAL QUESTION:
{plan.original_question}

CONTEXTUAL INTERPRETATION (conversation is intent only, never evidence):
{plan.contextual_question or plan.rewritten_question}

QUESTION INTENT:
{plan.intent}

PRIMARY PROCEDURE DOCUMENTS (routing priority only; facts still require source citations):
{chr(10).join(primary_document_names) if primary_document_names else "None"}

SOURCE CHUNKS:
{_source_blocks(sources)}

Answer only what the user asked. Use every source silently when deciding the answer, but do not reproduce unrelated or merely nearby evidence. For a single requested fact, return one direct sentence with citation. For a procedure, produce an actionable, properly structured solution: a short scope/applicability line if needed, then numbered operational steps in source order. If a dedicated primary procedure document is listed, make it the backbone and use other documents only for directly applicable supplementary requirements.
"""


def _source_blocks(sources: list[PromptSource]) -> str:
    blocks: list[str] = []
    for index, source in enumerate(sources, 1):
        chunk = source.result.chunk
        blocks.append(
            f"[S{index}] File: {chunk.filename} | Page: {chunk.page_number} | "
            f"Type: {chunk.content_type} | Retrieval: {source.result.method}\n"
            f"{source.excerpt}"
        )
    return "\n\n---\n\n".join(blocks)


def _build_batch_summary_prompt(
    plan: QueryPlan,
    batch: list[tuple[int, PromptSource]],
    primary_document_ids: set[str],
) -> str:
    blocks: list[str] = []
    for index, source in batch:
        chunk = source.result.chunk
        priority = " | PRIMARY PROCEDURE" if chunk.document_id in primary_document_ids else ""
        blocks.append(
            f"[S{index}] File: {chunk.filename} | Page: {chunk.page_number} | "
            f"Type: {chunk.content_type} | Retrieval: {source.result.method}{priority}\n"
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
    primary_document_names: list[str],
    primary_document_ids: set[str],
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

PRIMARY PROCEDURE DOCUMENTS:
{chr(10).join(primary_document_names) if primary_document_names else "None"}

SOURCE MAP:
{_source_map(sources)}

HIGH-PRIORITY DIRECT EVIDENCE:
{_priority_source_blocks(plan, sources, primary_document_ids=primary_document_ids)}

SOURCE EXCERPTS (HIERARCHICAL EVIDENCE DIGESTS):
{digests}

Required answer behavior:
- Answer the exact question, not merely the search terms.
- Match answer length to the question: one sentence for a single fact; compact bullets for several facts; longer structure only when required for a procedure, comparison or multi-part answer.
- Do not dump the evidence digests or list unrelated retrieved material; the UI exposes the complete reviewed evidence separately.
- Include every materially relevant fact preserved in the digests that is needed to answer or qualify the question.
- Keep different documents, lines, rolling stock, systems, modes and procedures separate when
  their applicability differs; never blend incompatible procedures.
- For procedures, preserve prerequisites, permissions, chronological steps, warnings,
  exceptions, checks, records and responsible roles. Format the answer as an actionable
  numbered procedure. When primary procedure documents are listed, derive the core sequence
  from them and use other PDFs only for directly applicable supplements.
- Cite every factual bullet using the exact [S#] labels from the digests/source map.
- Do not use conversation history, the model's general knowledge, or uncited assumptions as facts.
- If no supported answer exists, reply exactly: {NO_ANSWER}
"""


def _priority_source_blocks(
    plan: QueryPlan,
    sources: list[PromptSource],
    *,
    primary_document_ids: set[str] | None = None,
    max_sources: int = 20,
    max_chars: int = 36000,
) -> str:
    primary_ids = primary_document_ids or set()
    if plan.intent not in {"fact_lookup", "definition"} and not primary_ids:
        return "Not required for this question type."
    blocks: list[str] = []
    used = 0
    ranked = list(enumerate(sources, 1))
    if primary_ids:
        ranked.sort(key=lambda pair: (0 if pair[1].result.chunk.document_id in primary_ids else 1, pair[0]))
    for index, source in ranked[:max_sources]:
        chunk = source.result.chunk
        block = (
            f"[S{index}] File: {chunk.filename} | Page: {chunk.page_number} | "
            f"Type: {chunk.content_type}\n{source.excerpt}"
        )
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks) if blocks else "None"


def _source_map(sources: list[PromptSource]) -> str:
    lines: list[str] = []
    for index, source in enumerate(sources, 1):
        chunk = source.result.chunk
        lines.append(
            f"[S{index}] {chunk.filename} — page {chunk.page_number} — {chunk.content_type}"
        )
    return "\n".join(lines)
