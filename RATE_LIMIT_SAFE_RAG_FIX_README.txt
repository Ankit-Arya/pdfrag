pdfrag - rate-limit-safe hierarchical RAG fix
============================================

Observed failure
----------------
The backend received OpenAI HTTP 429 on gpt-5.6-luna TPM:
  Limit 200000, Used 177397, Requested 29136
The existing hierarchy submitted several large summary batches back-to-back. llm.py then retried
with short fixed/exponential delays that did not reliably clear the token window, so the exception
escaped and /api/chat returned 502.

What this patch changes
-----------------------
1. Rate-limit-header-aware pacing
   - OpenAI Responses calls use the SDK with_raw_response interface.
   - Reads x-ratelimit-limit-tokens, x-ratelimit-remaining-tokens and
     x-ratelimit-reset-tokens.
   - Estimates the next prompt + output reservation conservatively.
   - If the next call will not fit, waits for token capacity before sending it.

2. Safer 429 retry behavior
   - Retries are centralized in llm.py (SDK retries remain disabled).
   - Honors Retry-After / x-ratelimit-reset-tokens / "try again in ..." timing when available.
   - Uses exponential backoff with jitter and a total wait budget.
   - Same-model calls are serialized in this single-worker deployment so concurrent users cannot
     race two large requests into the same TPM bucket.
   - If the budget is genuinely exhausted after safe retries, API returns 503 + Retry-After rather
     than a misleading generic 502.

3. Lower token use without dropping PDF evidence
   - LLM prompts no longer repeat the full [PDF CHUNK CONTEXT] envelope for every chunk.
   - Actual PDF body text is unchanged.
   - File, page/page range, section, rolling-stock context, procedure context and important tags
     are retained once in a compact source header.
   - Evidence shown to users still uses the original complete PromptSource/excerpt.

4. Hierarchical summary cache
   - Successful evidence-batch summaries are cached in-process by model/effort/prompt hash.
   - If batch 8 rate-limits after batches 1-7 succeeded, retrying the user question reuses 1-7
     instead of paying for and rate-limiting on them again.
   - Cache is bounded and contains no new persistent storage.

5. Smaller final hierarchical prompt
   - The final source map contains only S# labels that survived into evidence digests.
   - The full reviewed-evidence list is still preserved for UI/audit transparency.

New optional environment controls
---------------------------------
LLM_MAX_RETRIES=6
LLM_RETRY_BASE_SECONDS=1.0
LLM_RETRY_MAX_SECONDS=30
LLM_RATE_LIMIT_MAX_WAIT_SECONDS=75
LLM_RATE_LIMIT_TOTAL_WAIT_SECONDS=90
LLM_RATE_LIMIT_SAFETY_SECONDS=0.35
LLM_RATE_LIMIT_SAFETY_TOKENS=1500
LLM_CHARS_PER_TOKEN_ESTIMATE=3.5
LLM_PROACTIVE_RATE_LIMIT_ENABLED=1
LLM_SERIALIZE_MODEL_REQUESTS=1
SUMMARY_CACHE_ENTRIES=384

The defaults are already wired through config.py and docker-compose.yml. You do not have to add
these to an existing .env unless you want to tune them.

Recommended model settings for critical operational PDFs
--------------------------------------------------------
Accuracy-first:
  LLM_MODEL=gpt-5.6-terra
  QUERY_MODEL=gpt-5.6-luna
  SUMMARY_MODEL=gpt-5.6-terra
  LLM_REASONING_EFFORT=high
  QUERY_REASONING_EFFORT=low
  SUMMARY_REASONING_EFFORT=medium

Cost-first while retaining the same safeguards:
  LLM_MODEL=gpt-5.6-terra
  QUERY_MODEL=gpt-5.6-luna
  SUMMARY_MODEL=gpt-5.6-luna
  LLM_REASONING_EFFORT=high
  QUERY_REASONING_EFFORT=low
  SUMMARY_REASONING_EFFORT=medium

The rate-limit fix is model-independent. Do not switch models merely to hide a TPM bug.

Install
-------
Replace the files from this ZIP and run:
  docker compose up -d --build

No PDF reprocessing/re-embedding is required.

Expected behavior
-----------------
For a large hierarchical answer you may now see an INFO line such as:
  Pacing gpt-5.6-luna request for 2.3s to stay within the token rate limit
That is intentional. A slightly slower successful grounded answer is preferable to re-sending
failed 29k-token requests and returning 502.

If rate limits are still too low for your total multi-user traffic, increase the OpenAI project/
organization usage tier/rate limit. This patch prevents avoidable bursts but cannot manufacture
capacity above the account's server-side limit.
