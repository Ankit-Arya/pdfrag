<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  downloadDocument,
  listDocuments,
  searchChunks,
  type ChunkSearchMode,
  type ChunkSearchResult,
  type DocumentRecord,
  type KnowledgeStatus,
} from '../services/api'

defineProps<{
  knowledge: KnowledgeStatus | null
}>()

const query = ref('')
const mode = ref<ChunkSearchMode>('hybrid')
const documentId = ref('')
const contentType = ref('')
const limit = ref(30)
const documents = ref<DocumentRecord[]>([])
const results = ref<ChunkSearchResult[]>([])
const searching = ref(false)
const loadingDocuments = ref(false)
const error = ref('')
const lastQuery = ref('')
const copiedChunkId = ref<string | null>(null)
let copyTimer: number | null = null

const readyDocuments = computed(() =>
  documents.value
    .filter((document) => document.status === 'ready')
    .sort((left, right) => left.filename.localeCompare(right.filename, undefined, { numeric: true })),
)

function preview(value: string): string {
  const clean = value.trim()
  return clean.length <= 520 ? clean : `${clean.slice(0, 520).trimEnd()}...`
}

function scoreLabel(score: number): string {
  if (!Number.isFinite(score)) return '-'
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`
}

async function loadDocuments(): Promise<void> {
  if (loadingDocuments.value) return
  loadingDocuments.value = true
  try {
    documents.value = await listDocuments()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not load document filters.'
  } finally {
    loadingDocuments.value = false
  }
}

async function submit(): Promise<void> {
  const value = query.value.trim()
  if (value.length < 2 || searching.value) return
  searching.value = true
  error.value = ''
  try {
    const response = await searchChunks({
      query: value,
      mode: mode.value,
      documentId: documentId.value || null,
      contentType: contentType.value || null,
      limit: limit.value,
    })
    results.value = response.results
    lastQuery.value = response.query
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Chunk search failed.'
  } finally {
    searching.value = false
  }
}

function clearSearch(): void {
  query.value = ''
  results.value = []
  lastQuery.value = ''
  error.value = ''
}

async function copyChunk(result: ChunkSearchResult): Promise<void> {
  try {
    await navigator.clipboard.writeText(result.text)
    copiedChunkId.value = result.chunk_id
    if (copyTimer !== null) window.clearTimeout(copyTimer)
    copyTimer = window.setTimeout(() => {
      copiedChunkId.value = null
      copyTimer = null
    }, 1600)
  } catch {
    error.value = 'Could not copy this chunk.'
  }
}

async function downloadSource(result: ChunkSearchResult): Promise<void> {
  try {
    await downloadDocument(result.document_id, result.filename)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not download the source PDF.'
  }
}

onMounted(() => {
  void loadDocuments()
})
</script>

<template>
  <main class="content-shell chunk-search-shell">
    <header class="topbar chunk-search-topbar">
      <div>
        <span class="eyebrow">Direct corpus search</span>
        <h1>Search PDF chunks</h1>
        <p>
          Search the processed PDF evidence directly. This does not create a chat,
          call the answer model, or save anything to chat history.
        </p>
      </div>
      <div class="chunk-search-ready">
        <strong>{{ knowledge?.ready_documents ?? 0 }}</strong>
        <span>ready PDFs</span>
      </div>
    </header>

    <section class="chunk-search-card">
      <form class="chunk-search-form" @submit.prevent="submit">
        <label class="chunk-search-query">
          <span>Search chunks</span>
          <div class="chunk-search-input-row">
            <input
              v-model="query"
              type="search"
              minlength="2"
              maxlength="500"
              autocomplete="off"
              placeholder="Search a phrase, rule, responsibility, equipment, condition..."
              autofocus
            />
            <button type="submit" :disabled="query.trim().length < 2 || searching">
              <span v-if="searching" class="chunk-search-spinner" aria-hidden="true" />
              {{ searching ? 'Searching...' : 'Search' }}
            </button>
          </div>
        </label>

        <div class="chunk-search-filters">
          <label>
            <span>Search mode</span>
            <select v-model="mode">
              <option value="hybrid">Hybrid - recommended</option>
              <option value="keyword">Keyword / phrase</option>
              <option value="semantic">Semantic meaning</option>
            </select>
          </label>

          <label>
            <span>Document</span>
            <select v-model="documentId">
              <option value="">All ready documents</option>
              <option
                v-for="document in readyDocuments"
                :key="document.id"
                :value="document.id"
              >
                {{ document.filename }}
              </option>
            </select>
          </label>

          <label>
            <span>Chunk type</span>
            <select v-model="contentType">
              <option value="">All types</option>
              <option value="text">Text</option>
              <option value="list">List</option>
              <option value="table_row">Table row</option>
              <option value="figure">Figure / caption</option>
            </select>
          </label>

          <label>
            <span>Results</span>
            <select v-model.number="limit">
              <option :value="20">20</option>
              <option :value="30">30</option>
              <option :value="50">50</option>
              <option :value="100">100</option>
            </select>
          </label>
        </div>

        <div class="chunk-search-help">
          <span><strong>Hybrid</strong> combines local semantic embeddings with full-text search.</span>
          <span><strong>Keyword</strong> is best for exact rule names, codes and quoted phrases.</span>
          <span><strong>Semantic</strong> is best when your wording differs from the PDF.</span>
        </div>
      </form>
    </section>

    <section class="chunk-results-section" aria-live="polite">
      <div class="chunk-results-heading">
        <div>
          <span class="eyebrow">Search results</span>
          <h2 v-if="lastQuery">
            {{ results.length }} matching chunk{{ results.length === 1 ? '' : 's' }}
          </h2>
          <h2 v-else>Search the evidence index directly</h2>
        </div>
        <button
          v-if="lastQuery"
          type="button"
          class="chunk-clear-button"
          @click="clearSearch"
        >
          Clear
        </button>
      </div>

      <div v-if="error" class="chunk-search-error" role="alert">{{ error }}</div>

      <div v-if="searching" class="chunk-search-empty">
        <span class="chunk-search-spinner large" aria-hidden="true" />
        <strong>Searching active v5 chunks...</strong>
        <p>No answer generation is being performed.</p>
      </div>

      <div v-else-if="lastQuery && !results.length" class="chunk-search-empty">
        <strong>No matching chunks found.</strong>
        <p>Try Hybrid or Semantic mode, remove a filter, or use a shorter search phrase.</p>
      </div>

      <div v-else-if="!lastQuery" class="chunk-search-empty">
        <strong>Search without starting a conversation.</strong>
        <p>
          Results show the source PDF, page, section, chunk type, retrieval mode,
          relevance score and the actual stored chunk text.
        </p>
      </div>

      <div v-else class="chunk-result-list">
        <article
          v-for="(result, index) in results"
          :key="result.chunk_id"
          class="chunk-result-card"
        >
          <header class="chunk-result-header">
            <div class="chunk-result-rank">{{ index + 1 }}</div>
            <div class="chunk-result-source">
              <strong :title="result.filename">{{ result.filename }}</strong>
              <span>
                p. {{ result.pages }}
                <template v-if="result.section"> - {{ result.section }}</template>
              </span>
            </div>
            <div class="chunk-result-score">
              <strong>{{ scoreLabel(result.score) }}</strong>
              <span>{{ result.retrieval_method }}</span>
            </div>
          </header>

          <div class="chunk-result-meta">
            <span>{{ result.content_type.replace('_', ' ') }}</span>
            <span>chunk {{ result.chunk_index }}</span>
            <span v-if="result.authority_status && result.authority_status !== 'unknown'">
              {{ result.authority_status.replaceAll('_', ' ') }}
            </span>
          </div>

          <p class="chunk-result-preview">{{ preview(result.text) }}</p>

          <details class="chunk-result-details">
            <summary>View full chunk</summary>
            <pre>{{ result.text }}</pre>
          </details>

          <footer class="chunk-result-actions">
            <button type="button" @click="copyChunk(result)">
              {{ copiedChunkId === result.chunk_id ? 'Copied' : 'Copy chunk' }}
            </button>
            <button type="button" @click="downloadSource(result)">
              Download source PDF
            </button>
          </footer>
        </article>
      </div>
    </section>
  </main>
</template>

<style scoped>
.chunk-search-shell { min-width: 0; }
.chunk-search-topbar { align-items: flex-start; }
.chunk-search-topbar p { max-width: 760px; }
.chunk-search-ready { min-width: 112px; padding: 12px 14px; border: 1px solid #dfe8e4; border-radius: 14px; background: #f8fbf9; text-align: right; }
.chunk-search-ready strong, .chunk-search-ready span { display: block; }
.chunk-search-ready strong { color: #263c34; font-size: 24px; }
.chunk-search-ready span { margin-top: 2px; color: #7a8983; font-size: 11px; }
.chunk-search-card { margin: 0 28px 18px; padding: 18px; border: 1px solid #dde7e2; border-radius: 18px; background: #fff; box-shadow: 0 10px 32px rgba(35, 67, 55, .05); }
.chunk-search-form { display: grid; gap: 14px; }
.chunk-search-query > span, .chunk-search-filters label > span { display: block; margin-bottom: 6px; color: #596a63; font-size: 11px; font-weight: 700; }
.chunk-search-input-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 9px; }
.chunk-search-input-row input, .chunk-search-filters select { width: 100%; border: 1px solid #d7e2dd; border-radius: 11px; background: #fbfdfc; color: #26362f; outline: none; }
.chunk-search-input-row input { min-height: 44px; padding: 0 13px; font-size: 13px; }
.chunk-search-input-row input:focus, .chunk-search-filters select:focus { border-color: #759b8b; box-shadow: 0 0 0 3px rgba(84, 129, 111, .09); }
.chunk-search-input-row button { min-width: 112px; border: 0; border-radius: 11px; padding: 0 17px; background: #315f4f; color: #fff; font-weight: 700; cursor: pointer; }
.chunk-search-input-row button:disabled { cursor: not-allowed; opacity: .55; }
.chunk-search-filters { display: grid; grid-template-columns: 1.1fr 2fr 1fr .7fr; gap: 10px; }
.chunk-search-filters select { height: 38px; padding: 0 9px; font-size: 11px; }
.chunk-search-help { display: flex; flex-wrap: wrap; gap: 6px 14px; color: #819089; font-size: 10px; line-height: 1.45; }
.chunk-results-section { padding: 0 28px 34px; }
.chunk-results-heading { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
.chunk-results-heading h2 { margin: 3px 0 0; color: #263a32; font-size: 16px; }
.chunk-clear-button { border: 0; background: transparent; color: #60766d; cursor: pointer; font-size: 11px; }
.chunk-search-error { margin-bottom: 12px; padding: 10px 12px; border: 1px solid #e8c7c7; border-radius: 10px; background: #fff7f7; color: #8a3e3e; font-size: 11px; }
.chunk-search-empty { display: grid; place-items: center; min-height: 220px; padding: 32px; border: 1px dashed #dbe5e0; border-radius: 16px; color: #73827c; text-align: center; }
.chunk-search-empty strong { color: #3c5249; font-size: 13px; }
.chunk-search-empty p { max-width: 560px; margin: 5px 0 0; font-size: 11px; line-height: 1.55; }
.chunk-result-list { display: grid; gap: 10px; }
.chunk-result-card { padding: 14px 15px; border: 1px solid #dfe7e3; border-radius: 14px; background: #fff; }
.chunk-result-header { display: grid; grid-template-columns: 28px minmax(0, 1fr) auto; align-items: start; gap: 9px; }
.chunk-result-rank { display: grid; place-items: center; width: 25px; height: 25px; border-radius: 8px; background: #edf5f1; color: #497160; font-size: 10px; font-weight: 800; }
.chunk-result-source { min-width: 0; }
.chunk-result-source strong, .chunk-result-source span { display: block; }
.chunk-result-source strong { overflow: hidden; color: #2b4037; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.chunk-result-source span { margin-top: 3px; color: #7d8b85; font-size: 10px; }
.chunk-result-score { text-align: right; }
.chunk-result-score strong, .chunk-result-score span { display: block; }
.chunk-result-score strong { color: #315e4e; font-size: 12px; }
.chunk-result-score span { margin-top: 2px; color: #8a9792; font-size: 9px; text-transform: capitalize; }
.chunk-result-meta { display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0 8px 37px; }
.chunk-result-meta span { padding: 3px 7px; border-radius: 999px; background: #f1f5f3; color: #66766f; font-size: 9px; text-transform: capitalize; }
.chunk-result-preview { margin: 0 0 0 37px; color: #485a53; font-size: 11px; line-height: 1.6; white-space: pre-wrap; }
.chunk-result-details { margin: 10px 0 0 37px; }
.chunk-result-details summary { color: #477260; cursor: pointer; font-size: 10px; font-weight: 700; }
.chunk-result-details pre { margin: 9px 0 0; padding: 12px; overflow: auto; border-radius: 10px; background: #f7faf8; color: #344941; font-family: inherit; font-size: 10.5px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; }
.chunk-result-actions { display: flex; gap: 8px; margin: 11px 0 0 37px; }
.chunk-result-actions button { border: 1px solid #d9e4df; border-radius: 8px; padding: 6px 9px; background: #fbfdfc; color: #526b60; cursor: pointer; font-size: 9px; font-weight: 700; }
.chunk-search-spinner { display: inline-block; width: 11px; height: 11px; margin-right: 5px; border: 2px solid rgba(255,255,255,.38); border-top-color: currentColor; border-radius: 999px; vertical-align: -2px; animation: chunk-spin .8s linear infinite; }
.chunk-search-spinner.large { width: 18px; height: 18px; margin: 0 0 8px; border-color: #d5e1dc; border-top-color: #4f7968; }
@keyframes chunk-spin { to { transform: rotate(360deg); } }
@media (max-width: 920px) { .chunk-search-filters { grid-template-columns: 1fr 1fr; } .chunk-search-ready { display: none; } }
@media (max-width: 620px) { .chunk-search-card, .chunk-results-section { margin-left: 14px; margin-right: 14px; padding-left: 0; padding-right: 0; } .chunk-search-card { padding: 14px; } .chunk-search-input-row, .chunk-search-filters { grid-template-columns: 1fr; } .chunk-search-input-row button { min-height: 40px; } .chunk-result-preview, .chunk-result-details, .chunk-result-actions, .chunk-result-meta { margin-left: 0; } }
</style>
