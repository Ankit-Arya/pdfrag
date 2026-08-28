<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  downloadDocument,
  listDocumentChunks,
  type DocumentChunkRecord,
  type DocumentChunkPage,
  type DocumentRecord,
} from '../services/api'

const props = defineProps<{ document: DocumentRecord }>()
const emit = defineEmits<{ close: [] }>()

const result = ref<DocumentChunkPage | null>(null)
const busy = ref(false)
const error = ref('')
const query = ref('')
const pageFilter = ref('')
const contentType = ref('')
const authorityStatus = ref('')
const pageSize = ref(100)
const offset = ref(0)
const rawText = ref(false)
const expandedChunkId = ref<string | null>(null)
const metadataChunkId = ref<string | null>(null)
const copied = ref('')
const downloading = ref(false)

const chunks = computed(() => result.value?.chunks ?? [])
const filteredCount = computed(() => result.value?.filtered_chunks ?? 0)
const startRow = computed(() => filteredCount.value ? offset.value + 1 : 0)
const endRow = computed(() => Math.min(offset.value + pageSize.value, filteredCount.value))
const canPrevious = computed(() => offset.value > 0)
const canNext = computed(() => offset.value + pageSize.value < filteredCount.value)

function parsedPage(): number | undefined {
  const value = Number.parseInt(pageFilter.value.trim(), 10)
  return Number.isFinite(value) && value > 0 ? value : undefined
}

function cleanChunkText(value: string): string {
  return value
    .replace(/\[PDF STRUCTURE\][\s\S]*?\[\/PDF STRUCTURE\]\s*/gi, '')
    .trim()
}

function shownText(chunk: DocumentChunkRecord): string {
  const value = rawText.value ? chunk.text : cleanChunkText(chunk.text)
  if (expandedChunkId.value === chunk.id || value.length <= 1800) return value
  return `${value.slice(0, 1800).trimEnd()}…`
}

function toggleExpanded(chunk: DocumentChunkRecord): void {
  expandedChunkId.value = expandedChunkId.value === chunk.id ? null : chunk.id
}

function toggleMetadata(chunk: DocumentChunkRecord): void {
  metadataChunkId.value = metadataChunkId.value === chunk.id ? null : chunk.id
}

function chunkJson(chunk: DocumentChunkRecord): string {
  return JSON.stringify(chunk, null, 2)
}

async function copy(value: string, key: string): Promise<void> {
  await navigator.clipboard.writeText(value)
  copied.value = key
  window.setTimeout(() => {
    if (copied.value === key) copied.value = ''
  }, 1500)
}

async function load(resetOffset = false): Promise<void> {
  if (resetOffset) offset.value = 0
  busy.value = true
  error.value = ''
  try {
    result.value = await listDocumentChunks(props.document.id, {
      offset: offset.value,
      limit: pageSize.value,
      q: query.value.trim() || undefined,
      page: parsedPage(),
      content_type: contentType.value || undefined,
      authority_status: authorityStatus.value || undefined,
    })
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not load v5 chunks.'
  } finally {
    busy.value = false
  }
}

async function previous(): Promise<void> {
  if (!canPrevious.value) return
  offset.value = Math.max(0, offset.value - pageSize.value)
  await load(false)
}

async function next(): Promise<void> {
  if (!canNext.value) return
  offset.value += pageSize.value
  await load(false)
}

async function download(): Promise<void> {
  downloading.value = true
  try {
    await downloadDocument(props.document.id, props.document.filename)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not download the PDF.'
  } finally {
    downloading.value = false
  }
}

function exportVisibleJson(): void {
  const payload = {
    document: props.document.filename,
    run_id: result.value?.run_id,
    processing_version: result.value?.processing_version,
    filters: {
      query: query.value,
      page: parsedPage(),
      content_type: contentType.value || null,
      authority_status: authorityStatus.value || null,
    },
    showing: { start: startRow.value, end: endRow.value, filtered: filteredCount.value },
    chunks: chunks.value,
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = window.document.createElement('a')
  anchor.href = url
  anchor.download = `${props.document.filename.replace(/\.pdf$/i, '')}-chunks-${startRow.value}-${endRow.value}.json`
  window.document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

onMounted(() => {
  void load(true)
})
</script>

<template>
  <main class="content-shell chunk-explorer-shell">
    <header class="topbar chunk-explorer-topbar">
      <div class="chunk-title-wrap">
        <button type="button" class="back-button" @click="emit('close')">← Documents</button>
        <div>
          <span class="eyebrow">PDF chunk diagnostics</span>
          <h1 :title="document.filename">{{ document.filename }}</h1>
          <p>Inspect the active RAG v5 chunks exactly as stored and searched.</p>
        </div>
      </div>
      <div class="header-actions">
        <button type="button" @click="exportVisibleJson" :disabled="!chunks.length">Export visible JSON</button>
        <button type="button" @click="download" :disabled="downloading">
          {{ downloading ? 'Downloading…' : 'Download PDF' }}
        </button>
      </div>
    </header>

    <section class="chunk-explorer-content">
      <div class="run-summary">
        <div><span>Document chunks</span><strong>{{ result?.total_chunks ?? document.chunk_count }}</strong></div>
        <div><span>Filtered chunks</span><strong>{{ result?.filtered_chunks ?? '—' }}</strong></div>
        <div><span>Processing version</span><strong>{{ result?.processing_version || '—' }}</strong></div>
        <div><span>Active run</span><strong class="mono">{{ result?.run_id ? result.run_id.slice(0, 12) : '—' }}</strong></div>
      </div>

      <div class="chunk-toolbar">
        <label class="chunk-query">
          <span>Text / heading / section</span>
          <input v-model="query" type="search" placeholder="BIC, Brake Isolation Cock, 25 km/h…" @keyup.enter="load(true)" />
        </label>
        <label>
          <span>PDF page</span>
          <input v-model="pageFilter" inputmode="numeric" placeholder="Any" @keyup.enter="load(true)" />
        </label>
        <label>
          <span>Content type</span>
          <select v-model="contentType">
            <option value="">All types</option>
            <option v-for="item in result?.content_types ?? []" :key="item" :value="item">{{ item }}</option>
          </select>
        </label>
        <label>
          <span>Authority</span>
          <select v-model="authorityStatus">
            <option value="">All</option>
            <option v-for="item in result?.authority_statuses ?? []" :key="item" :value="item">{{ item }}</option>
          </select>
        </label>
        <label>
          <span>Page size</span>
          <select v-model.number="pageSize">
            <option :value="50">50</option>
            <option :value="100">100</option>
            <option :value="200">200</option>
            <option :value="500">500</option>
          </select>
        </label>
        <button type="button" class="apply-button" :disabled="busy" @click="load(true)">
          {{ busy ? 'Loading…' : 'Apply filters' }}
        </button>
      </div>

      <div class="chunk-options">
        <label class="raw-toggle">
          <input v-model="rawText" type="checkbox" />
          <span>Show raw <code>[PDF STRUCTURE]</code> envelope</span>
        </label>
        <span>Showing {{ startRow }}–{{ endRow }} of {{ filteredCount }}</span>
      </div>

      <p v-if="error" class="chunk-error">{{ error }}</p>
      <div v-if="busy && !chunks.length" class="chunk-empty">Loading active v5 chunks…</div>
      <div v-else-if="!chunks.length" class="chunk-empty">No chunks match these filters.</div>

      <div v-else class="chunk-list">
        <article v-for="chunk in chunks" :key="chunk.id" class="chunk-card">
          <header class="chunk-card-header">
            <div class="chunk-index">#{{ chunk.chunk_index }}</div>
            <div class="chunk-heading-block">
              <strong>{{ chunk.heading || chunk.section_path[chunk.section_path.length - 1] || 'Unsectioned chunk' }}</strong>
              <span v-if="chunk.section_path.length">{{ chunk.section_path.join(' › ') }}</span>
            </div>
            <div class="chunk-badges">
              <span>p. {{ chunk.page_number }}<template v-if="chunk.page_end !== chunk.page_number">–{{ chunk.page_end }}</template></span>
              <span>{{ chunk.content_type }}</span>
              <span>{{ chunk.authority_status }}</span>
              <span>{{ chunk.char_count }} chars</span>
            </div>
          </header>

          <div class="chunk-technical-row">
            <span><strong>Parent</strong> {{ chunk.parent_key || '—' }}</span>
            <span><strong>Confidence</strong> {{ chunk.extraction_confidence.toFixed(3) }}</span>
            <span v-if="chunk.table_id"><strong>Table</strong> {{ chunk.table_id.slice(0, 12) }}<template v-if="chunk.table_row_index !== null"> · row {{ chunk.table_row_index }}</template></span>
            <span class="mono"><strong>ID</strong> {{ chunk.id }}</span>
          </div>

          <pre class="chunk-text">{{ shownText(chunk) }}</pre>

          <div class="chunk-actions">
            <button v-if="(rawText ? chunk.text : cleanChunkText(chunk.text)).length > 1800" type="button" @click="toggleExpanded(chunk)">
              {{ expandedChunkId === chunk.id ? 'Collapse text' : 'Show full text' }}
            </button>
            <button type="button" @click="copy(chunk.text, `text-${chunk.id}`)">
              {{ copied === `text-${chunk.id}` ? 'Copied' : 'Copy raw text' }}
            </button>
            <button type="button" @click="copy(chunkJson(chunk), `json-${chunk.id}`)">
              {{ copied === `json-${chunk.id}` ? 'Copied' : 'Copy chunk JSON' }}
            </button>
            <button type="button" @click="toggleMetadata(chunk)">
              {{ metadataChunkId === chunk.id ? 'Hide metadata' : 'Metadata' }}
            </button>
          </div>

          <pre v-if="metadataChunkId === chunk.id" class="chunk-metadata">{{ JSON.stringify(chunk.metadata, null, 2) }}</pre>
        </article>
      </div>

      <footer class="chunk-pagination">
        <button type="button" :disabled="!canPrevious || busy" @click="previous">← Previous</button>
        <span>{{ startRow }}–{{ endRow }} / {{ filteredCount }}</span>
        <button type="button" :disabled="!canNext || busy" @click="next">Next →</button>
      </footer>
    </section>
  </main>
</template>

<style scoped>
.chunk-explorer-shell { display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; }
.chunk-explorer-topbar { gap: 18px; }
.chunk-title-wrap { min-width: 0; display: flex; align-items: flex-start; gap: 12px; }
.chunk-title-wrap > div { min-width: 0; }
.chunk-title-wrap h1 { max-width: min(68vw, 900px); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.back-button { min-height: 34px; padding: 0 10px; border: 1px solid #d6e0dc; border-radius: 9px; background: #fff; color: #45695a; font-size: 9px; font-weight: 800; }
.header-actions { display: flex; gap: 7px; flex-wrap: wrap; justify-content: flex-end; }
.header-actions button { min-height: 34px; padding: 0 10px; border: 1px solid #d6e0dc; border-radius: 9px; background: #fff; color: #45695a; font-size: 8px; font-weight: 800; }
.chunk-explorer-content { min-height: 0; overflow-y: auto; padding: 18px clamp(18px, 3vw, 38px) 40px; }
.run-summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }
.run-summary > div { padding: 11px 12px; border: 1px solid #dde6e2; border-radius: 11px; background: #fff; }
.run-summary span, .run-summary strong { display: block; }
.run-summary span { color: #7c8984; font-size: 8px; }
.run-summary strong { margin-top: 4px; color: #30463d; font-size: 11px; overflow-wrap: anywhere; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.chunk-toolbar { display: grid; grid-template-columns: minmax(240px, 1fr) 90px 130px 120px 95px auto; align-items: end; gap: 8px; padding: 12px; border: 1px solid #dce5e1; border-radius: 13px; background: #fff; }
.chunk-toolbar label > span { display: block; margin-bottom: 5px; color: #66766f; font-size: 8px; font-weight: 800; }
.chunk-toolbar input, .chunk-toolbar select { width: 100%; min-height: 36px; padding: 0 9px; border: 1px solid #d7e1dd; border-radius: 8px; outline: none; background: #fbfdfc; color: #31453d; font-size: 9px; }
.apply-button { min-height: 36px; padding: 0 11px; border: 1px solid #bcd0c7; border-radius: 8px; background: #edf6f2; color: #2f6750; font-size: 8px; font-weight: 900; }
.chunk-options { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 9px 2px 12px; color: #7b8883; font-size: 8px; }
.raw-toggle { display: flex; align-items: center; gap: 6px; }
.raw-toggle code { font-size: 7px; }
.chunk-error { padding: 10px 12px; border: 1px solid #e8caca; border-radius: 10px; background: #fff7f7; color: #984c4c; font-size: 9px; }
.chunk-empty { min-height: 240px; display: grid; place-items: center; border: 1px dashed #d9e2de; border-radius: 13px; color: #7b8983; font-size: 10px; }
.chunk-list { display: grid; gap: 10px; }
.chunk-card { padding: 12px; border: 1px solid #dce5e1; border-radius: 12px; background: #fff; box-shadow: 0 4px 16px rgba(31, 65, 52, .025); }
.chunk-card-header { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: start; gap: 9px; }
.chunk-index { min-width: 48px; padding: 5px 7px; border-radius: 8px; background: #edf3f0; color: #416a59; font-size: 8px; font-weight: 900; text-align: center; }
.chunk-heading-block { min-width: 0; }
.chunk-heading-block strong { display: block; color: #30453c; font-size: 10px; overflow-wrap: anywhere; }
.chunk-heading-block span { display: block; margin-top: 3px; color: #81908a; font-size: 8px; line-height: 1.4; overflow-wrap: anywhere; }
.chunk-badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 4px; }
.chunk-badges span { padding: 3px 5px; border-radius: 999px; background: #f0f4f2; color: #64756e; font-size: 7px; white-space: nowrap; }
.chunk-technical-row { display: flex; flex-wrap: wrap; gap: 5px 12px; margin-top: 8px; padding: 7px 8px; border-radius: 8px; background: #f8faf9; color: #76847f; font-size: 7.5px; }
.chunk-technical-row strong { color: #50635b; }
.chunk-text { margin: 9px 0 0; padding: 11px; overflow-x: auto; white-space: pre-wrap; overflow-wrap: anywhere; border: 1px solid #e2e9e6; border-radius: 9px; background: #fcfdfd; color: #354840; font: 9px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.chunk-actions { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 7px; }
.chunk-actions button, .chunk-pagination button { min-height: 29px; padding: 0 8px; border: 1px solid #d5e0db; border-radius: 7px; background: #fbfdfc; color: #476c5d; font-size: 7.5px; font-weight: 800; }
.chunk-metadata { margin: 7px 0 0; padding: 9px; max-height: 360px; overflow: auto; border-radius: 8px; background: #18221e; color: #d9e5df; font: 8px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.chunk-pagination { display: flex; align-items: center; justify-content: center; gap: 12px; padding: 17px 0 0; color: #73827c; font-size: 8px; }
.chunk-pagination button:disabled, .header-actions button:disabled, .apply-button:disabled { opacity: .45; cursor: default; }
@media (max-width: 1050px) { .chunk-toolbar { grid-template-columns: 1fr 100px 130px; } .apply-button { align-self: stretch; } .run-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 700px) { .chunk-toolbar, .run-summary { grid-template-columns: 1fr; } .chunk-card-header { grid-template-columns: auto minmax(0, 1fr); } .chunk-badges { grid-column: 1 / -1; justify-content: flex-start; } .header-actions { display: none; } }
</style>
