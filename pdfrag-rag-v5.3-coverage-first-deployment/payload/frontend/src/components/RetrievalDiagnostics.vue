<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  listDocuments,
  type AnswerResponse,
  type DocumentRecord,
  type RetrievalDocumentDiagnostic,
  type RetrievalDiagnosticSummary,
} from '../services/api'

const props = defineProps<{ response: AnswerResponse | null }>()

type FilterName = 'all' | 'routed' | 'rejected' | 'contributing' | 'cited' | 'no_signal'
const filter = ref<FilterName>('all')
const search = ref('')
const documents = ref<DocumentRecord[]>([])
const loading = ref(false)
const loadError = ref('')
const limit = ref(80)

const diagnostics = computed(() => props.response?.retrieval_diagnostics ?? [])
const summary = computed<RetrievalDiagnosticSummary>(() => props.response?.retrieval_diagnostic_summary ?? {})
const citedNames = computed(() => new Set((props.response?.sources ?? []).map((item) => item.filename.toLowerCase())))
const diagnosticNames = computed(() => new Set(diagnostics.value.map((item) => item.filename.toLowerCase())))
const traceTruncated = computed(() => Boolean(summary.value.diagnostics_truncated))

const noSignalRows = computed<RetrievalDocumentDiagnostic[]>(() => {
  if (traceTruncated.value) return []
  return documents.value
    .filter((item) => item.status === 'ready' && !diagnosticNames.value.has(item.filename.toLowerCase()))
    .map((item) => ({
      document_id: item.id,
      filename: item.filename,
      discovery_score: 0,
      vector_score: 0,
      keyword_score: 0,
      dimension_hits: 0,
      signals: [],
      routed: false,
      deep_searched: false,
      rerank_role: '',
      final_evidence: false,
      contributing: false,
      decision: 'NO_RETRIEVAL_SIGNAL',
      reason: 'No retrieval signal was produced during the initial corpus-discovery stage.',
      best_page: null,
      best_heading: '',
    }))
})

const allRows = computed(() => [...diagnostics.value, ...noSignalRows.value])
const filteredRows = computed(() => {
  const needle = search.value.trim().toLowerCase()
  const rows = allRows.value.filter((row) => {
    if (needle && !row.filename.toLowerCase().includes(needle)) return false
    if (filter.value === 'routed') return row.routed
    if (filter.value === 'rejected') return !row.routed && row.decision !== 'NO_RETRIEVAL_SIGNAL'
    if (filter.value === 'contributing') return row.contributing
    if (filter.value === 'cited') return citedNames.value.has(row.filename.toLowerCase())
    if (filter.value === 'no_signal') return row.decision === 'NO_RETRIEVAL_SIGNAL'
    return true
  })
  return rows.slice(0, limit.value)
})

function score(value?: number): string {
  return Number.isFinite(value) ? Number(value).toFixed(2) : '—'
}

function status(row: RetrievalDocumentDiagnostic): string {
  if (citedNames.value.has(row.filename.toLowerCase())) return 'Cited'
  if (row.contributing) return 'Contributing'
  if (row.final_evidence) return 'Final evidence'
  if (row.routed) return 'Routed'
  if (row.decision === 'NO_RETRIEVAL_SIGNAL') return 'No signal'
  return 'Rejected'
}

function statusClass(row: RetrievalDocumentDiagnostic): string {
  return status(row).toLowerCase().replaceAll(' ', '-')
}

async function loadDocuments(): Promise<void> {
  if (documents.value.length || loading.value) return
  loading.value = true
  loadError.value = ''
  try {
    documents.value = await listDocuments()
  } catch (cause) {
    loadError.value = cause instanceof Error ? cause.message : 'Could not load the document list.'
  } finally {
    loading.value = false
  }
}

onMounted(() => { void loadDocuments() })
</script>

<template>
  <div class="diagnostics-root">
    <section class="diag-card">
      <span class="eyebrow">Testing / analysis</span>
      <h3>Retrieval diagnostics</h3>
      <p class="diag-note">
        Deterministic retrieval decisions only. This does not expose private chain-of-thought or hidden prompts.
      </p>
      <div class="summary-grid">
        <div><strong>{{ summary.eligible_documents ?? '—' }}</strong><span>corpus eligible</span></div>
        <div><strong>{{ summary.documents_with_signal ?? diagnostics.length }}</strong><span>with signal</span></div>
        <div><strong>{{ summary.routed_documents ?? diagnostics.filter((item) => item.routed).length }}</strong><span>routed</span></div>
        <div><strong>{{ summary.scope_count_routed ?? '—' }}</strong><span>RS/Line scopes routed</span></div>
        <div><strong>{{ summary.scope_final_evidence_documents ?? '—' }}</strong><span>scope docs in evidence</span></div>
        <div><strong>{{ summary.contributing_documents ?? diagnostics.filter((item) => item.contributing).length }}</strong><span>contributing</span></div>
        <div><strong>{{ summary.no_signal_documents ?? '—' }}</strong><span>no signal</span></div>
        <div><strong>{{ summary.corpus_discovery_stages ?? '—' }}</strong><span>corpus stage</span></div>
      </div>
      <div class="pipeline-line">
        {{ summary.coverage_mode
          ? 'Coverage-aware discovery → RS/Line scope pinning → deep search → rerank → coverage review → cited answer'
          : summary.definition_enumeration
            ? 'Exact alias scan → source-definition inventory → all meanings/locations → cited answer'
            : 'Unified discovery → routed deep search → AI rerank → coverage review → cited answer' }}
      </div>

      <div v-if="summary.definition_enumeration && summary.definition_inventory?.length" class="definition-inventory">
        <strong>Definition inventory</strong>
        <article v-for="item in summary.definition_inventory" :key="`${item.alias}-${item.meaning}`">
          <b>{{ item.alias }} — {{ item.meaning }}</b>
          <span v-for="source in item.sources" :key="`${source.filename}-${source.page_start}-${source.page_end}`">
            {{ source.filename }} · p. {{ source.page_start }}<template v-if="source.page_end && source.page_end !== source.page_start">–{{ source.page_end }}</template><template v-if="source.heading"> · {{ source.heading }}</template>
          </span>
        </article>
      </div>
    </section>

    <section class="diag-card">
      <div class="toolbar">
        <input v-model="search" type="search" placeholder="Find PDF in trace…" />
        <select v-model="filter">
          <option value="all">All decisions</option>
          <option value="routed">Routed</option>
          <option value="rejected">Rejected</option>
          <option value="contributing">Contributing</option>
          <option value="cited">Cited</option>
          <option value="no_signal">No retrieval signal</option>
        </select>
      </div>

      <p v-if="loadError" class="warning">{{ loadError }}</p>
      <p v-if="traceTruncated" class="warning">
        Diagnostic signal rows were capped by the backend. No-signal filenames cannot be derived exactly for this answer; counts remain valid.
      </p>
      <p v-else-if="loading" class="muted">Loading document names for no-signal comparison…</p>

      <div class="diag-list">
        <article v-for="row in filteredRows" :key="`${row.document_id}-${row.filename}`" class="diag-row">
          <div class="diag-title">
            <strong :title="row.filename">{{ row.filename }}</strong>
            <span :class="`decision ${statusClass(row)}`">{{ status(row) }}</span>
          </div>
          <div class="score-grid">
            <span><b>{{ score(row.discovery_score) }}</b> discovery</span>
            <span><b>{{ score(row.vector_score) }}</b> vector</span>
            <span><b>{{ score(row.keyword_score) }}</b> lexical</span>
            <span><b>{{ row.dimension_hits ?? 0 }}</b> dimensions</span>
          </div>
          <div v-if="row.scope_label && row.scope_type !== 'common'" class="scope-line">
            <strong>{{ row.scope_label }}</strong>
            <span v-if="row.scope_pinned">pinned for scope coverage</span>
            <span v-if="row.route_rank">route rank {{ row.route_rank }}<template v-if="row.global_route_cap"> / ordinary cap {{ row.global_route_cap }}</template></span>
          </div>
          <div v-if="row.best_heading || row.best_page" class="best-match">
            <strong v-if="row.best_heading">{{ row.best_heading }}</strong>
            <span v-if="row.best_page">p. {{ row.best_page }}</span>
          </div>
          <p><b>{{ row.decision }}</b> · {{ row.reason }}</p>
          <small v-if="row.signals?.length">Signals: {{ row.signals.join(', ') }}</small>
          <small v-if="row.rerank_role">Reranker: {{ row.rerank_role.replaceAll('_', ' ') }}</small>
        </article>
      </div>

      <button v-if="filteredRows.length >= limit && limit < allRows.length" class="more" @click="limit += 80">
        Show more
      </button>
    </section>
  </div>
</template>

<style scoped>
.diagnostics-root { display: grid; gap: 11px; }
.diag-card { padding: 13px; border: 1px solid #dde6e2; border-radius: 13px; background: #fff; box-shadow: 0 6px 20px rgba(36, 67, 56, .035); }
.diag-card h3 { margin: 3px 0 6px; color: #2b4037; font-size: 12px; }
.eyebrow { color: #82908a; font-size: 7px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; }
.diag-note, .muted, .warning { margin: 0; color: #788680; font-size: 8px; line-height: 1.5; }
.warning { margin-top: 8px; padding: 7px; border-radius: 8px; background: #faf4e8; color: #765f39; }
.summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 10px; }
.summary-grid div { padding: 7px; border-radius: 9px; background: #f5f8f6; }
.summary-grid strong, .summary-grid span { display: block; }
.summary-grid strong { color: #29473b; font-size: 12px; }
.summary-grid span { margin-top: 1px; color: #89958f; font-size: 7px; }
.pipeline-line { margin-top: 8px; padding: 7px 8px; border-radius: 8px; background: #eef5f1; color: #4d675d; font-size: 7.5px; line-height: 1.45; }
.definition-inventory { display: grid; gap: 6px; margin-top: 8px; padding: 8px; border-radius: 9px; background: #f7faf8; color: #53675f; font-size: 7px; }
.definition-inventory > strong { color: #314b40; font-size: 8px; }
.definition-inventory article { display: grid; gap: 2px; padding-top: 5px; border-top: 1px solid #e4ebe8; }
.definition-inventory article:first-of-type { border-top: 0; padding-top: 0; }
.definition-inventory b { color: #355549; font-size: 7.5px; }
.definition-inventory span { color: #74837d; overflow-wrap: anywhere; }
.toolbar { display: grid; grid-template-columns: minmax(0, 1fr) 120px; gap: 7px; }
.toolbar input, .toolbar select { min-width: 0; height: 32px; border: 1px solid #d9e3df; border-radius: 8px; padding: 0 8px; background: #fff; color: #40534b; font: inherit; font-size: 8px; }
.diag-list { display: grid; gap: 7px; margin-top: 10px; }
.diag-row { padding: 9px; border: 1px solid #e4ebe8; border-radius: 10px; background: #fbfcfc; }
.diag-title { display: flex; align-items: start; gap: 7px; }
.diag-title > strong { min-width: 0; flex: 1; overflow: hidden; color: #334940; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
.decision { flex: 0 0 auto; padding: 2px 5px; border-radius: 999px; background: #eef2f0; color: #65736d; font-size: 6.5px; font-weight: 900; text-transform: uppercase; }
.decision.cited, .decision.contributing { background: #e6f5ed; color: #286b52; }
.decision.routed, .decision.final-evidence { background: #ebf1f8; color: #496582; }
.decision.rejected, .decision.no-signal { background: #f5f1eb; color: #79664a; }
.score-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin-top: 7px; }
.score-grid span { padding: 4px; border-radius: 6px; background: #f1f5f3; color: #82908a; font-size: 6.5px; }
.score-grid b { color: #43554d; font-size: 7.5px; }
.scope-line { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; color: #5d746a; font-size: 7px; }
.scope-line strong { color: #2e5545; }
.scope-line span { padding: 1px 4px; border-radius: 999px; background: #eef5f1; }
.best-match { display: flex; align-items: baseline; gap: 5px; margin-top: 6px; color: #718079; font-size: 7px; }
.best-match strong { color: #42544c; }
.diag-row p { margin: 6px 0 0; color: #67756f; font-size: 7.5px; line-height: 1.45; }
.diag-row small { display: block; margin-top: 3px; color: #8a9691; font-size: 6.5px; overflow-wrap: anywhere; }
.more { width: 100%; min-height: 30px; margin-top: 9px; border: 1px solid #d8e2de; border-radius: 8px; background: #f7faf8; color: #4e675d; font-size: 8px; font-weight: 800; }
@media (max-width: 1100px) { .summary-grid { grid-template-columns: repeat(2, 1fr); } .toolbar { grid-template-columns: 1fr; } }
</style>
