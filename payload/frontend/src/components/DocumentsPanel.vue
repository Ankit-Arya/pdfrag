<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  downloadDocument,
  listDocuments,
  type DocumentRecord,
  type KnowledgeStatus,
} from '../services/api'

const props = defineProps<{
  knowledge: KnowledgeStatus | null
}>()

const documents = ref<DocumentRecord[]>([])
const query = ref('')
const status = ref('ready')
const busy = ref(false)
const error = ref('')
const downloadingId = ref<string | null>(null)

const filtered = computed(() => {
  const needle = query.value.trim().toLowerCase()
  return documents.value.filter((document) => {
    const statusMatch = status.value === 'all' || document.status === status.value
    const queryMatch = !needle || document.filename.toLowerCase().includes(needle)
    return statusMatch && queryMatch
  })
})

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(value))
}

async function refresh(): Promise<void> {
  busy.value = true
  error.value = ''
  try {
    documents.value = await listDocuments()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not load documents.'
  } finally {
    busy.value = false
  }
}

async function download(record: DocumentRecord): Promise<void> {
  downloadingId.value = record.id
  try {
    await downloadDocument(record.id, record.filename)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not download the PDF.'
  } finally {
    downloadingId.value = null
  }
}

onMounted(() => {
  void refresh()
})
</script>

<template>
  <main class="content-shell documents-shell">
    <header class="topbar documents-topbar">
      <div>
        <span class="eyebrow">Shared knowledge</span>
        <h1>Documents</h1>
        <p>Browse the PDFs available to IMS without starting a chat.</p>
      </div>
      <div class="document-header-stat">
        <strong>{{ knowledge?.ready_documents ?? 0 }}</strong>
        <span>ready PDFs</span>
      </div>
    </header>

    <section class="documents-content">
      <div class="documents-toolbar">
        <label class="documents-search">
          <span>Search documents</span>
          <input v-model="query" type="search" placeholder="Filename, code, manual, procedure…" />
        </label>
        <label>
          <span>Status</span>
          <select v-model="status">
            <option value="ready">Ready</option>
            <option value="all">All</option>
            <option value="processing">Processing</option>
            <option value="uploaded">Uploaded</option>
            <option value="failed">Failed</option>
          </select>
        </label>
        <button type="button" class="documents-refresh" :disabled="busy" @click="refresh">
          {{ busy ? 'Refreshing…' : 'Refresh' }}
        </button>
      </div>

      <div class="documents-summary">
        <span>{{ filtered.length }} matching document{{ filtered.length === 1 ? '' : 's' }}</span>
        <span>{{ knowledge?.total_chunks ?? 0 }} indexed chunks</span>
      </div>

      <p v-if="error" class="documents-error">{{ error }}</p>
      <div v-if="busy && !documents.length" class="documents-empty">Loading the document library…</div>
      <div v-else-if="!filtered.length" class="documents-empty">No documents match the current filters.</div>

      <div v-else class="document-grid">
        <article v-for="document in filtered" :key="document.id" class="document-card">
          <div class="document-icon">PDF</div>
          <div class="document-card-main">
            <strong :title="document.filename">{{ document.filename }}</strong>
            <div class="document-card-meta">
              <span>{{ document.page_count }} pages</span>
              <span>{{ document.chunk_count }} chunks</span>
              <span>{{ formatBytes(document.size_bytes) }}</span>
              <span>{{ formatDate(document.created_at) }}</span>
            </div>
            <small v-if="document.error">{{ document.error }}</small>
          </div>
          <span class="document-status" :class="document.status">{{ document.status }}</span>
          <button
            type="button"
            class="document-download"
            :disabled="downloadingId === document.id"
            @click="download(document)"
          >
            {{ downloadingId === document.id ? 'Downloading…' : 'Download' }}
          </button>
        </article>
      </div>
    </section>
  </main>
</template>

<style scoped>
.documents-shell {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  min-height: 0;
}

.documents-content {
  min-height: 0;
  overflow-y: auto;
  padding: 22px clamp(22px, 4vw, 54px) 40px;
}

.document-header-stat {
  min-width: 110px;
  padding: 11px 13px;
  border: 1px solid #dce5e1;
  border-radius: 13px;
  background: #fff;
  text-align: right;
}
.document-header-stat strong,
.document-header-stat span { display: block; }
.document-header-stat strong { color: #27463a; font-size: 21px; }
.document-header-stat span { margin-top: 2px; color: #7b8983; font-size: 9px; }

.documents-toolbar {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) 150px auto;
  align-items: end;
  gap: 10px;
  padding: 15px;
  border: 1px solid #dce5e1;
  border-radius: 15px;
  background: #fff;
}
.documents-toolbar label > span {
  display: block;
  margin-bottom: 6px;
  color: #66766f;
  font-size: 9px;
  font-weight: 800;
}
.documents-toolbar input,
.documents-toolbar select {
  width: 100%;
  min-height: 39px;
  padding: 0 10px;
  border: 1px solid #d7e1dd;
  border-radius: 9px;
  outline: none;
  background: #fbfdfc;
  color: #31453d;
  font-size: 10px;
}
.documents-refresh {
  min-height: 39px;
  padding: 0 14px;
  border: 1px solid #bfd2ca;
  border-radius: 9px;
  background: #eef6f2;
  color: #336651;
  font-size: 9px;
  font-weight: 800;
}

.documents-summary {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  margin: 12px 2px;
  color: #7c8a84;
  font-size: 9px;
}

.documents-error {
  padding: 10px 12px;
  border: 1px solid #e7caca;
  border-radius: 10px;
  background: #fff7f7;
  color: #974a4a;
  font-size: 9px;
}

.documents-empty {
  min-height: 260px;
  display: grid;
  place-items: center;
  border: 1px dashed #d9e2de;
  border-radius: 15px;
  color: #7b8983;
  font-size: 10px;
}

.document-grid { display: grid; gap: 8px; }
.document-card {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 11px;
  padding: 12px 13px;
  border: 1px solid #dde6e2;
  border-radius: 12px;
  background: #fff;
}
.document-icon {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: #f8eded;
  color: #a45151;
  font-size: 8px;
  font-weight: 900;
}
.document-card-main { min-width: 0; }
.document-card-main > strong {
  display: block;
  overflow: hidden;
  color: #30463d;
  font-size: 10.5px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.document-card-meta { display: flex; flex-wrap: wrap; gap: 5px 10px; margin-top: 4px; color: #84918c; font-size: 8px; }
.document-card-main small { display: block; margin-top: 5px; color: #a24e4e; font-size: 8px; }
.document-status {
  padding: 4px 7px;
  border-radius: 999px;
  background: #f0f4f2;
  color: #68766f;
  font-size: 8px;
  font-weight: 800;
  text-transform: capitalize;
}
.document-status.ready { background: #e8f6ef; color: #2d7158; }
.document-status.failed { background: #fff0ef; color: #a34e48; }
.document-download {
  min-height: 32px;
  padding: 0 10px;
  border: 1px solid #d5e0db;
  border-radius: 8px;
  background: #fbfdfc;
  color: #476c5d;
  font-size: 8px;
  font-weight: 800;
}

@media (max-width: 760px) {
  .documents-toolbar { grid-template-columns: 1fr; }
  .document-card { grid-template-columns: 34px minmax(0, 1fr); }
  .document-status,
  .document-download { grid-column: 2; justify-self: start; }
}
</style>
