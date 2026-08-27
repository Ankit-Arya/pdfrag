from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from pathlib import Path

MARKER = "IMS_UI_V2_SYNTHESIS_V52"


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + ".bak-before-ims-ui-v2-synthesis-v52")
    if not target.exists():
        shutil.copy2(path, target)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_models(source: str) -> str:
    if "answer_strategy: str = \"direct_lookup\"" in source:
        return source
    anchor = '    answer_policy_version: str = ""\n'
    addition = '''    # Query/answer strategy metadata for the v5.2 inspector and persisted chats.\n    answer_strategy: str = "direct_lookup"\n    synthesis_dimensions: list[str] = Field(default_factory=list)\n    search_scope: str = "focused"\n    relevant_documents: list[str] = Field(default_factory=list)\n    contributing_documents: list[str] = Field(default_factory=list)\n    search_rounds: int = 1\n    evidence_coverage_status: str = "unknown"\n    conflicts: list[dict[str, Any]] = Field(default_factory=list)\n'''
    return replace_once(source, anchor, addition + anchor, "AnswerResponse synthesis metadata")


def patch_api(source: str) -> str:
    if '"answer_strategy": response.answer_strategy' in source:
        return source
    anchor = '            "search_queries": response.search_queries,\n'
    addition = '''            "answer_strategy": response.answer_strategy,\n            "synthesis_dimensions": response.synthesis_dimensions,\n            "search_scope": response.search_scope,\n            "relevant_documents": response.relevant_documents,\n            "contributing_documents": response.contributing_documents,\n            "search_rounds": response.search_rounds,\n            "evidence_coverage_status": response.evidence_coverage_status,\n            "conflicts": response.conflicts,\n'''
    return replace_once(source, anchor, anchor + addition, "persist synthesis metadata")


def patch_service(source: str) -> str:
    if "from app.rag.v5.synthesis_retrieval import" not in source:
        start = source.find("from app.rag.v5.assistant_retrieval import")
        if start < 0:
            raise RuntimeError(
                "Assistant v5.1 import not found in service.py. Apply IMS Assistant v5.1 before v5.2."
            )
        line_end = source.find("\n", start)
        if line_end < 0:
            raise RuntimeError("Malformed assistant_retrieval import in service.py")
        if source[start:line_end].rstrip().endswith("("):
            import_end = source.find("\n)\n", line_end)
            if import_end < 0:
                raise RuntimeError("Could not find end of multiline assistant_retrieval import")
            import_end += len("\n)\n")
        else:
            import_end = line_end + 1
        import_block = '''from app.rag.v5.synthesis_retrieval import (\n    SynthesisRetrievalBundle,\n    coverage_prompt_status,\n    response_synthesis_metadata,\n    review_bundle_coverage,\n    retrieve_assistant_v52,\n    select_results_for_answer,\n)\n'''
        source = source[:import_end] + import_block + source[import_end:]

    if "retrieve_assistant_v51(" in source:
        source = source.replace("retrieve_assistant_v51(", "retrieve_assistant_v52(")

    source = source.replace(
        "bundle: AssistantRetrievalBundle = retrieve_assistant_v52(",
        "bundle: SynthesisRetrievalBundle = retrieve_assistant_v52(",
    )

    old_review = "review_retrieved_evidence(interpretation, bundle.results)"
    if old_review in source:
        source = source.replace(old_review, "review_bundle_coverage(db, interpretation, bundle)")

    source_selection_variants = [
        (
            "prompt_sources = _assistant_sources(bundle.results, evidence_limit, interpretation)",
            "answer_results = select_results_for_answer(bundle.results, review)\n"
            "        prompt_sources = _assistant_sources(answer_results, evidence_limit, interpretation)",
        ),
        (
            "prompt_sources = _document_diverse_sources(bundle.results, evidence_limit)",
            "answer_results = select_results_for_answer(bundle.results, review)\n"
            "        prompt_sources = _document_diverse_sources(answer_results, evidence_limit)",
        ),
    ]
    if "answer_results = select_results_for_answer(bundle.results, review)" not in source:
        for old_prompt_sources, new_prompt_sources in source_selection_variants:
            if old_prompt_sources in source:
                source = source.replace(old_prompt_sources, new_prompt_sources, 1)
                break
        else:
            raise RuntimeError("Could not find assistant evidence-selection anchor in service.py")

    old_evidence_limit = 'evidence_limit = min(top_k or _int_env("RAG_V5_FINAL_EVIDENCE", 32, 12, 80), 80)'
    if "RAG_V52_SYNTHESIS_EVIDENCE" not in source and old_evidence_limit in source:
        source = source.replace(
            old_evidence_limit,
            'base_evidence_limit = min(top_k or _int_env("RAG_V5_FINAL_EVIDENCE", 32, 12, 80), 80)\n'
            '        evidence_limit = (\n'
            '            min(80, max(base_evidence_limit, _int_env("RAG_V52_SYNTHESIS_EVIDENCE", 48, 24, 80)))\n'
            '            if bundle.answer_strategy == "multi_document_synthesis"\n'
            '            else base_evidence_limit\n'
            '        )',
            1,
        )

    if "coverage_prompt_status(coverage_status, bundle, review)" not in source:
        draft_anchor = "        draft = llm_service.generate(\n"
        if draft_anchor not in source:
            raise RuntimeError("Could not find answer-generation anchor in service.py")
        source = source.replace(
            draft_anchor,
            "        coverage_status = coverage_prompt_status(coverage_status, bundle, review)\n"
            + draft_anchor,
            1,
        )

    synthesis_rules = '''\nMULTI-DOCUMENT SYNTHESIS:\n- When COVERAGE STATUS says answer strategy is multi_document_synthesis, use every supplied document that materially contributes to the requested answer.\n- Organize the answer by the user's requested responsibilities, conditions, procedure, applicability, exceptions or comparison -- not by document name.\n- Combine complementary evidence across documents when it supports different parts of the answer.\n- Do not force incidental or merely keyword-matching documents into the answer.\n- If documents differ because of scope, explain the scope difference. If an amendment/current provision establishes precedence, use it and explain the material change when useful.\n- If a genuine conflict remains unresolved by scope or authority evidence, clearly state the competing provisions instead of silently choosing one.\n- Cite the specific source(s) supporting every important synthesized point; multi-source citations are encouraged when several documents jointly support a statement.\n'''
    if "MULTI-DOCUMENT SYNTHESIS:" not in source:
        anchor = "For procedure questions, provide an actionable numbered sequence in operational order."
        if anchor not in source:
            raise RuntimeError("Could not find v5 answer-system synthesis insertion anchor")
        source = source.replace(anchor, synthesis_rules + "\n" + anchor, 1)

    # Add v5.2 metadata to every AnswerResponse path that uses the assistant policy.
    if "**response_synthesis_metadata(bundle, review)," not in source:
        for old_policy in ("rag-v5.1-assistant", "rag-v5.0.0"):
            inline = f'search_queries=bundle.search_queries, answer_policy_version="{old_policy}",'
            if inline in source:
                source = source.replace(
                    inline,
                    'search_queries=bundle.search_queries,\n'
                    '                **response_synthesis_metadata(bundle, review),\n'
                    '                answer_policy_version="rag-v5.2-synthesis",',
                    1,
                )
        policy_pattern = re.compile(
            r'(?P<indent>^[ \t]*)answer_policy_version="rag-v5\.(?:0\.0|1-assistant)",',
            re.MULTILINE,
        )
        source = policy_pattern.sub(
            lambda m: (
                f'{m.group("indent")}**response_synthesis_metadata(bundle, review),\n'
                f'{m.group("indent")}answer_policy_version="rag-v5.2-synthesis",'
            ),
            source,
        )
        if "**response_synthesis_metadata(bundle, review)," not in source:
            raise RuntimeError("Could not find v5 AnswerResponse policy fields for synthesis metadata")
    else:
        source = re.sub(
            r'answer_policy_version="rag-v5\.(?:0\.0|1-assistant)"',
            'answer_policy_version="rag-v5.2-synthesis"',
            source,
        )

    return source


def patch_api_ts(source: str) -> str:
    if "export interface ConflictSummary" not in source:
        anchor = "export interface AnswerResponse {"
        addition = """

export interface ConflictSummary {
  type: 'complementary' | 'scope_difference' | 'authority_difference' | 'unresolved' | string
  documents: string[]
  summary: string
  resolution: string
}
"""
        source = replace_once(source, anchor, addition + "\n" + anchor, "frontend conflict type")

    if "export interface HealthResponse" not in source:
        anchor = "export interface AnswerResponse {"
        addition = """

export interface HealthResponse {
  status: string
  embedding_model: string
  embedding_ready: boolean
  embedding_backend?: string | null
  embedding_fallback?: boolean
  embedding_error?: string | null
  llm_model: string
  query_model?: string
  summary_model?: string
  answer_policy_version?: string
  ocr_mode: string
  ocr_available: boolean
  table_extraction: boolean
  table_extraction_available: boolean
  query_rewrite: boolean
}
"""
        source = replace_once(source, anchor, addition + "\n" + anchor, "frontend health type")

    if "answer_strategy?:" not in source:
        anchor = "  search_queries: string[]\n"
        addition = '''  answer_strategy?: 'direct_lookup' | 'multi_document_synthesis' | string\n  synthesis_dimensions?: string[]\n  search_scope?: 'focused' | 'broad_relevant_corpus' | string\n  relevant_documents?: string[]\n  contributing_documents?: string[]\n  search_rounds?: number\n  evidence_coverage_status?: 'complete' | 'incomplete' | 'unknown' | string\n  conflicts?: ConflictSummary[]\n'''
        source = replace_once(source, anchor, anchor + addition, "AnswerResponse frontend synthesis fields")

    if "export async function getHealth(" not in source:
        anchor = "export async function getKnowledgeStatus(): Promise<KnowledgeStatus> {"
        function = '''\n\nexport async function getHealth(): Promise<HealthResponse> {\n  return apiRequest<HealthResponse>('/api/health')\n}\n'''
        source = replace_once(source, anchor, function + "\n" + anchor, "frontend health API")
    return source


def _canonical_view_type(source: str) -> str:
    pattern = re.compile(r"type ViewName = [^\n]+")
    match = pattern.search(source)
    if not match:
        raise RuntimeError("ViewName type not found")
    return source[: match.start()] + "type ViewName = 'chat' | 'search' | 'documents' | 'admin' | 'account'" + source[match.end():]


def patch_app(source: str) -> str:
    source = _canonical_view_type(source)

    imports = {
        "AnswerInspector": "import AnswerInspector from './components/AnswerInspector.vue'\n",
        "CommandPalette": "import CommandPalette from './components/CommandPalette.vue'\n",
        "DocumentsPanel": "import DocumentsPanel from './components/DocumentsPanel.vue'\n",
    }
    for name, line in imports.items():
        if line.strip() in source:
            continue
        anchor = "import ChatPanel from './components/ChatPanel.vue'\n"
        source = replace_once(source, anchor, anchor + line, f"{name} import")

    if "const inspectedResponse = ref<AnswerResponse | null>(null)" not in source:
        anchor = "const view = ref<ViewName>('chat')\n"
        addition = (
            "const inspectedResponse = ref<AnswerResponse | null>(null)\n"
            "const commandOpen = ref(false)\n"
        )
        source = replace_once(source, anchor, anchor + addition, "App inspector state")

    if "answer_strategy:" not in source:
        anchor = '''    search_queries: Array.isArray(metadata.search_queries)\n      ? (metadata.search_queries as string[])\n      : [],\n'''
        addition = '''    answer_strategy:\n      typeof metadata.answer_strategy === 'string' ? metadata.answer_strategy : 'direct_lookup',\n    synthesis_dimensions: Array.isArray(metadata.synthesis_dimensions)\n      ? (metadata.synthesis_dimensions as string[])\n      : [],\n    search_scope:\n      typeof metadata.search_scope === 'string' ? metadata.search_scope : 'focused',\n    relevant_documents: Array.isArray(metadata.relevant_documents)\n      ? (metadata.relevant_documents as string[])\n      : [],\n    contributing_documents: Array.isArray(metadata.contributing_documents)\n      ? (metadata.contributing_documents as string[])\n      : [],\n    search_rounds:\n      typeof metadata.search_rounds === 'number' ? metadata.search_rounds : 1,\n    evidence_coverage_status:\n      typeof metadata.evidence_coverage_status === 'string'\n        ? metadata.evidence_coverage_status\n        : 'unknown',\n    conflicts: Array.isArray(metadata.conflicts)\n      ? (metadata.conflicts as NonNullable<AnswerResponse['conflicts']>)\n      : [],\n'''
        source = replace_once(source, anchor, anchor + addition, "stored synthesis metadata mapping")

    if "inspectedResponse.value = null" not in source:
        # New chat is the cleanest place to reset the inspector.
        anchor = "function newChat(): void {\n  cancel()\n"
        source = replace_once(
            source,
            anchor,
            anchor + "  inspectedResponse.value = null\n",
            "new-chat inspector reset",
        )

    if "messages.value].reverse().find" not in source:
        open_start = source.find("async function openChat")
        if open_start < 0:
            raise RuntimeError("openChat function not found")
        open_tail = source[open_start:]
        pair_match = re.search(
            r"(?m)^(?P<indent>[ \t]*)question\.value = ''\s*\n(?P=indent)view\.value = 'chat'",
            open_tail,
        )
        if not pair_match:
            raise RuntimeError("openChat inspector anchor not found")
        anchor_pos = open_start + pair_match.start()
        indent = pair_match.group("indent")
        addition = (
            f"{indent}inspectedResponse.value =\n"
            f"{indent}  [...messages.value].reverse().find(\n"
            f"{indent}    (message) => message.role === 'assistant' && message.response,\n"
            f"{indent}  )?.response ?? null\n"
        )
        source = source[:anchor_pos] + addition + source[anchor_pos:]

    if "inspectedResponse.value = response" not in source:
        ask_start = source.find("async function ask(")
        if ask_start < 0:
            raise RuntimeError("ask function not found")
        ask_tail = source[ask_start:]
        chats_match = re.search(r"(?m)^(?P<indent>[ \t]*)chats\.value = await listChats\(\)\s*$", ask_tail)
        if not chats_match:
            raise RuntimeError("ask response inspector anchor not found")
        anchor_pos = ask_start + chats_match.start()
        indent = chats_match.group("indent")
        source = source[:anchor_pos] + f"{indent}inspectedResponse.value = response\n" + source[anchor_pos:]

    if "function inspectResponse(response: AnswerResponse)" not in source:
        anchor = "\nonMounted(() => {\n"
        handlers = '''\nfunction inspectResponse(response: AnswerResponse): void {\n  inspectedResponse.value = response\n}\n\nfunction askFromCommand(value: string): void {\n  question.value = value\n  view.value = 'chat'\n  commandOpen.value = false\n}\n\nfunction navigateFromCommand(nextView: 'chat' | 'search' | 'documents'): void {\n  view.value = nextView\n  commandOpen.value = false\n}\n'''
        source = replace_once(source, anchor, handlers + anchor, "App inspector/command handlers")

    if '@command="commandOpen = true"' not in source:
        upload_start = source.find("    <UploadPanel")
        upload_end = source.find("    />", upload_start)
        anchor = '      @logout="signOut"\n'
        anchor_pos = source.find(anchor, upload_start, upload_end if upload_end >= 0 else None)
        if upload_start < 0 or anchor_pos < 0:
            raise RuntimeError("UploadPanel logout anchor not found for command handler")
        source = source[:anchor_pos] + '      @command="commandOpen = true"\n' + source[anchor_pos:]

    if '@inspect="inspectResponse"' not in source:
        anchor = '      @cancel="cancel"\n'
        source = replace_once(source, anchor, anchor + '      @inspect="inspectResponse"\n', "ChatPanel inspector handler")

    if '<DocumentsPanel' not in source:
        anchor = '''    <AdminPanel\n      v-else-if="view === 'admin' && user.role === 'admin'"\n'''
        addition = '''    <DocumentsPanel\n      v-else-if="view === 'documents'"\n      :knowledge="knowledge"\n    />\n\n'''
        source = replace_once(source, anchor, addition + anchor, "Documents workspace")

    if '<AnswerInspector' not in source:
        anchor = '    <div v-if="error" class="error-toast" role="alert">'
        addition = '''    <AnswerInspector\n      v-if="view === 'chat'"\n      :response="inspectedResponse"\n      :knowledge="knowledge"\n    />\n\n    <CommandPalette\n      v-model:open="commandOpen"\n      @ask="askFromCommand"\n      @navigate="navigateFromCommand"\n    />\n\n'''
        source = replace_once(source, anchor, addition + anchor, "Answer inspector and command palette")

    return source


def patch_sidebar(source: str) -> str:
    source = _canonical_view_type(source)

    if "command: []" not in source:
        anchor = "  logout: []\n"
        source = replace_once(source, anchor, anchor + "  command: []\n", "sidebar command emit")

    if "sidebar-search-button" not in source:
        anchor = '''    <button class="new-chat-button" @click="emit('newChat')">\n      <span aria-hidden="true">＋</span>\n      New chat\n    </button>\n'''
        addition = '''\n    <button class="sidebar-search-button" type="button" @click="emit('command')">\n      <span aria-hidden="true">⌕</span>\n      <strong>Search all</strong>\n      <kbd>Ctrl K</kbd>\n    </button>\n'''
        source = replace_once(source, anchor, anchor + addition, "sidebar command launcher")

    if "view === 'documents'" not in source:
        admin_anchor = '''      <button\n        v-if="user.role === 'admin'"\n'''
        addition = '''      <button :class="{ active: view === 'documents' }" @click="emit('navigate', 'documents')">\n        <span aria-hidden="true">▱</span>\n        Documents\n      </button>\n'''
        source = replace_once(source, admin_anchor, addition + admin_anchor, "Documents nav item")
    return source


def patch_chat_panel(source: str) -> str:
    if "inspect: [response: AnswerResponse]" not in source:
        anchor = "  cancel: []\n"
        source = replace_once(source, anchor, anchor + "  inspect: [response: AnswerResponse]\n", "ChatPanel inspect emit")

    if "View details" not in source:
        anchor = '''            <button\n              type="button"\n              class="copy-answer-button"\n              :aria-label="copiedMessageId === message.id ? 'Answer copied' : 'Copy answer'"\n              @click="copyAnswer(message)"\n            >\n              {{ copiedMessageId === message.id ? 'Copied' : 'Copy answer' }}\n            </button>\n'''
        addition = '''            <button\n              v-if="message.response"\n              type="button"\n              class="answer-details-button"\n              @click="emit('inspect', message.response)"\n            >\n              View details\n            </button>\n'''
        source = replace_once(source, anchor, anchor + addition, "Chat answer details action")

    if "class=\"key-references\"" not in source:
        # Handle the optional-chaining form used by current ChatPanel.
        anchor = '''          <details\n            v-if="message.response?.evidence?.length"\n'''
        if anchor not in source:
            anchor = '''          <details\n            v-if="message.response?.evidence.length"\n'''
        if anchor not in source:
            raise RuntimeError("ChatPanel evidence anchor not found for Key References")
        addition = '''          <section v-if="message.response?.sources?.length" class="key-references">\n            <div class="key-reference-heading">\n              <strong>Key references</strong>\n              <span>{{ message.response.sources.length }} cited</span>\n            </div>\n            <div class="key-reference-list">\n              <div\n                v-for="source in message.response.sources.slice(0, 4)"\n                :key="`key-ref-${message.id}-${source.id}`"\n                class="key-reference-row"\n              >\n                <span class="key-reference-pdf">PDF</span>\n                <div>\n                  <strong :title="source.filename">{{ source.filename }}</strong>\n                  <small>\n                    p. {{ source.pages || source.page }}\n                    <template v-if="source.section"> · {{ source.section }}</template>\n                  </small>\n                </div>\n                <span class="key-reference-score">{{ source.score.toFixed(2) }}</span>\n              </div>\n            </div>\n          </section>\n\n'''
        source = replace_once(source, anchor, addition + anchor, "Key References panel")

    if "IMS_UI_V2_CHAT_STYLES" not in source:
        style_close = source.rfind("</style>")
        if style_close < 0:
            raise RuntimeError("ChatPanel </style> not found")
        css = r'''
/* IMS_UI_V2_CHAT_STYLES */
.answer-details-button {
  min-height: 27px;
  padding: 0 8px;
  border: 1px solid #cddbd5;
  border-radius: 8px;
  color: #315f4f;
  background: #f3f8f5;
  font-size: 9px;
  font-weight: 800;
}

.key-references {
  margin-top: 12px;
  padding: 11px 12px;
  border: 1px solid #dce6e2;
  border-radius: 12px;
  background: rgba(249, 252, 250, .9);
}

.key-reference-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 7px;
}
.key-reference-heading strong { color: #30483e; font-size: 9px; }
.key-reference-heading span { color: #81908a; font-size: 8px; }
.key-reference-list { display: grid; gap: 5px; }
.key-reference-row {
  display: grid;
  grid-template-columns: 27px minmax(0, 1fr) auto;
  align-items: start;
  gap: 7px;
  padding: 6px 0;
  border-bottom: 1px solid #ebf0ed;
}
.key-reference-row:last-child { border-bottom: 0; }
.key-reference-pdf {
  width: 25px;
  height: 25px;
  display: grid;
  place-items: center;
  border-radius: 7px;
  background: #f8eded;
  color: #a45252;
  font-size: 6px;
  font-weight: 900;
}
.key-reference-row div { min-width: 0; }
.key-reference-row div strong,
.key-reference-row div small { display: block; }
.key-reference-row div strong {
  overflow: hidden;
  color: #43564e;
  font-size: 8.5px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.key-reference-row div small { margin-top: 2px; color: #89958f; font-size: 7.5px; }
.key-reference-score { color: #337159; font-size: 8px; font-weight: 800; }
'''
        source = source[:style_close] + css + "\n" + source[style_close:]
    return source



def patch_compose(source: str) -> str:
    if "RAG_V52_MAX_RELEVANT_DOCUMENTS" in source:
        return source
    anchor = "      RAG_V51_MAX_SECTION_CHUNKS: ${RAG_V51_MAX_SECTION_CHUNKS:-12}\n"
    addition = anchor + '''      # v5.2 synthesis policy: broad relevant-document discovery without forcing irrelevant PDFs.
      RAG_V52_MAX_QUERY_VARIANTS: ${RAG_V52_MAX_QUERY_VARIANTS:-12}
      RAG_V52_DISCOVERY_PER_QUERY: ${RAG_V52_DISCOVERY_PER_QUERY:-110}
      RAG_V52_MAX_RELEVANT_DOCUMENTS: ${RAG_V52_MAX_RELEVANT_DOCUMENTS:-24}
      RAG_V52_DOCUMENT_RELATIVE_THRESHOLD: ${RAG_V52_DOCUMENT_RELATIVE_THRESHOLD:-0.18}
      RAG_V52_RERANK_CANDIDATES: ${RAG_V52_RERANK_CANDIDATES:-72}
      RAG_V52_FINAL_CANDIDATES: ${RAG_V52_FINAL_CANDIDATES:-96}
      RAG_V52_SYNTHESIS_EVIDENCE: ${RAG_V52_SYNTHESIS_EVIDENCE:-48}
      RAG_V52_COVERAGE_CANDIDATES: ${RAG_V52_COVERAGE_CANDIDATES:-48}
'''
    return replace_once(source, anchor, addition, "docker compose v5.2 settings")


def patch_env_merge(source: str) -> str:
    if '"RAG_V52_MAX_RELEVANT_DOCUMENTS"' in source:
        return source
    anchor = '    "RAG_V51_MAX_SECTION_CHUNKS" = "12"\n'
    addition = anchor + '''    "RAG_V52_MAX_QUERY_VARIANTS" = "12"
    "RAG_V52_DISCOVERY_PER_QUERY" = "110"
    "RAG_V52_MAX_RELEVANT_DOCUMENTS" = "24"
    "RAG_V52_DOCUMENT_RELATIVE_THRESHOLD" = "0.18"
    "RAG_V52_RERANK_CANDIDATES" = "72"
    "RAG_V52_FINAL_CANDIDATES" = "96"
    "RAG_V52_SYNTHESIS_EVIDENCE" = "48"
    "RAG_V52_COVERAGE_CANDIDATES" = "48"
'''
    return replace_once(source, anchor, addition, "merge-v5-env v5.2 settings")

def patch_global_style(source: str) -> str:
    if "IMS_UI_V2_GLOBAL_LAYOUT" in source:
        return source
    return source + r'''

/* IMS_UI_V2_GLOBAL_LAYOUT */
@media (min-width: 1181px) {
  .app-layout {
    grid-template-columns: 270px minmax(0, 1fr) 340px;
  }

  .app-layout > .chat-shell {
    grid-column: 2;
  }

  .app-layout > .answer-inspector {
    grid-column: 3;
  }

  .app-layout > .content-shell:not(.chat-shell) {
    grid-column: 2 / 4;
  }
}

.sidebar-search-button {
  width: 100%;
  min-height: 36px;
  margin-top: 8px;
  padding: 0 10px;
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr) auto;
  align-items: center;
  gap: 7px;
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 10px;
  color: #a9beb7;
  background: rgba(255,255,255,.035);
  text-align: left;
}
.sidebar-search-button:hover { color: #eff9f5; background: rgba(255,255,255,.075); }
.sidebar-search-button > span { color: #88b5a6; font-size: 15px; }
.sidebar-search-button strong { font-size: 10px; }
.sidebar-search-button kbd {
  padding: 2px 5px;
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 5px;
  color: #78948a;
  background: rgba(0,0,0,.08);
  font-size: 7px;
  font-family: inherit;
}

@media (max-width: 1180px) {
  .app-layout {
    grid-template-columns: 270px minmax(0, 1fr);
  }
  .app-layout > .content-shell {
    grid-column: 2 !important;
  }
  .app-layout > .answer-inspector {
    display: none;
  }
  .answer-details-button {
    display: none;
  }
}
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install IMS UI v2 information architecture plus v5.2 multi-document synthesis retrieval."
    )
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    payload = Path(__file__).resolve().parent / "payload"

    paths = {
        "models": repo / "backend/app/models.py",
        "api": repo / "backend/app/api.py",
        "service": repo / "backend/app/rag/v5/service.py",
        "assistant": repo / "backend/app/rag/v5/assistant_retrieval.py",
        "api_ts": repo / "frontend/src/services/api.ts",
        "app": repo / "frontend/src/App.vue",
        "sidebar": repo / "frontend/src/components/UploadPanel.vue",
        "chat": repo / "frontend/src/components/ChatPanel.vue",
        "styles": repo / "frontend/src/style.css",
        "chunk_search": repo / "frontend/src/components/ChunkSearchPanel.vue",
        "compose": repo / "docker-compose.v5.yml",
        "env_merge": repo / "merge-v5-env.ps1",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Required repository file(s) not found:\n" + "\n".join(missing))

    assistant_source = paths["assistant"].read_text(encoding="utf-8", errors="ignore")
    service_source = paths["service"].read_text(encoding="utf-8", errors="ignore")
    api_ts_source = paths["api_ts"].read_text(encoding="utf-8", errors="ignore")
    if "retrieve_assistant_v51" not in assistant_source or "retrieve_assistant_v51" not in service_source:
        raise SystemExit(
            "IMS Assistant v5.1 is not active in the real backend files. Apply the v5.1 assistant patch first."
        )
    if "searchChunks(" not in api_ts_source:
        raise SystemExit(
            "The separate Search chunks workspace/API is not active. Apply the chunk-search patch before this UI v2 patch."
        )

    added = {
        repo / "backend/app/rag/v5/synthesis_retrieval.py": payload / "backend/app/rag/v5/synthesis_retrieval.py",
        repo / "backend/tests/test_v52_synthesis_policy.py": payload / "backend/tests/test_v52_synthesis_policy.py",
        repo / "frontend/src/components/AnswerInspector.vue": payload / "frontend/src/components/AnswerInspector.vue",
        repo / "frontend/src/components/DocumentsPanel.vue": payload / "frontend/src/components/DocumentsPanel.vue",
        repo / "frontend/src/components/CommandPalette.vue": payload / "frontend/src/components/CommandPalette.vue",
    }
    for target, src in added.items():
        if not src.exists():
            raise SystemExit(f"Patch payload missing: {src}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup(target)
        target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    transforms = {
        paths["models"]: patch_models,
        paths["api"]: patch_api,
        paths["service"]: patch_service,
        paths["api_ts"]: patch_api_ts,
        paths["app"]: patch_app,
        paths["sidebar"]: patch_sidebar,
        paths["chat"]: patch_chat_panel,
        paths["styles"]: patch_global_style,
        paths["compose"]: patch_compose,
        paths["env_merge"]: patch_env_merge,
    }

    for path, transform in transforms.items():
        backup(path)
        original = path.read_text(encoding="utf-8")
        patched = transform(original)
        path.write_text(patched, encoding="utf-8", newline="\n")
        print(f"[patched] {path.relative_to(repo)}")

    # Syntax validation for all changed backend Python files.
    for path in (
        paths["models"],
        paths["api"],
        paths["service"],
        repo / "backend/app/rag/v5/synthesis_retrieval.py",
        repo / "backend/tests/test_v52_synthesis_policy.py",
    ):
        py_compile.compile(str(path), doraise=True)

    print()
    print("Applied IMS UI v2 + multi-document synthesis v5.2 patch.")
    print("Backend:")
    print(" - automatic direct_lookup vs multi_document_synthesis strategy")
    print(" - broad relevant-document discovery for synthesis questions")
    print(" - evidence-dimension retrieval across all meaningfully relevant PDFs")
    print(" - document-balanced reranking and section expansion")
    print(" - cross-document completeness/conflict/authority review")
    print(" - targeted retry queries for missing synthesis dimensions")
    print(" - final evidence filtered to materially contributing documents")
    print(" - answer policy rag-v5.2-synthesis with per-point citations")
    print("UI:")
    print(" - responsive desktop 3-column chat + answer inspector")
    print(" - Sources / Query Plan / Details tabs")
    print(" - Key References on assistant answers")
    print(" - separate Documents workspace")
    print(" - Ctrl/Cmd+K global search palette using direct chunk search")
    print(" - system readiness card")
    print(" - existing Search Chunks, Live Working, Admin and Account preserved")
    print("No database migration, PDF reprocessing, or embedding rebuild is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
