<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  listDocuments,
  searchChunks,
  type ChunkSearchResult,
  type DocumentRecord,
} from '../services/api'

const open = defineModel<boolean>('open', { required: true })
const emit = defineEmits<{
  ask: [question: string]
  navigate: [view: 'chat' | 'search' | 'documents']
}>()

const query = ref('')
const documents = ref<DocumentRecord[]>([])
const chunks = ref<ChunkSearchResult[]>([])
const busy = ref(false)
const error = ref('')
const input = ref<HTMLInputElement | null>(null)
let searchTimer: number | null = null

const documentMatches = computed(() => {
  const needle = query.value.trim().toLowerCase()
  if (!needle) return documents.value.filter((item) => item.status === 'ready').slice(0, 6)
  return documents.value
    .filter((item) => item.status === 'ready' && item.filename.toLowerCase().includes(needle))
    .slice(0, 6)
})

function close(): void {
  open.value = false
}

function ask(): void {
  const value = query.value.trim()
  if (!value) return
  emit('ask', value)
  close()
}

function navigate(view: 'chat' | 'search' | 'documents'): void {
  emit('navigate', view)
  close()
}

async function loadDocuments(): Promise<void> {
  try {
    documents.value = await listDocuments()
  } catch {
    // Chunk search can still work even if the filename catalogue is unavailable.
  }
}

async function runChunkSearch(value: string): Promise<void> {
  if (value.trim().length < 2) {
    chunks.value = []
    return
  }
  busy.value = true
  error.value = ''
  try {
    const response = await searchChunks({ query: value.trim(), mode: 'hybrid', limit: 8 })
    if (query.value.trim() === value.trim()) chunks.value = response.results
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Search failed.'
  } finally {
    busy.value = false
  }
}

function handleKeydown(event: KeyboardEvent): void {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault()
    open.value = !open.value
    return
  }
  if (event.key === 'Escape' && open.value) close()
}

watch(query, (value) => {
  if (searchTimer !== null) window.clearTimeout(searchTimer)
  searchTimer = window.setTimeout(() => {
    void runChunkSearch(value)
  }, 250)
})

watch(open, async (value) => {
  window.document.body.style.overflow = value ? 'hidden' : ''
  if (!value) return
  if (!documents.value.length) void loadDocuments()
  await nextTick()
  input.value?.focus()
})

onMounted(() => window.addEventListener('keydown', handleKeydown))
onBeforeUnmount(() => {
  if (searchTimer !== null) window.clearTimeout(searchTimer)
  window.removeEventListener('keydown', handleKeydown)
  window.document.body.style.overflow = ''
})
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="command-backdrop" @mousedown.self="close">
      <section class="command-palette" role="dialog" aria-modal="true" aria-label="Search IMS">
        <header class="command-search-row">
          <span class="command-search-icon">⌕</span>
          <input
            ref="input"
            v-model="query"
            type="search"
            placeholder="Search documents, sections, rules, responsibilities…"
            autocomplete="off"
            @keydown.enter.prevent="ask"
          />
          <kbd>Esc</kbd>
        </header>

        <div class="command-actions">
          <button type="button" :disabled="!query.trim()" @click="ask">
            <strong>Ask IMS AI</strong>
            <span>Use this wording as a new question</span>
          </button>
          <button type="button" @click="navigate('search')">
            <strong>Search chunks</strong>
            <span>Open direct evidence search</span>
          </button>
          <button type="button" @click="navigate('documents')">
            <strong>Documents</strong>
            <span>Browse the shared PDF library</span>
          </button>
        </div>

        <div class="command-results">
          <section v-if="documentMatches.length">
            <span class="command-heading">Documents</span>
            <div class="command-list">
              <button
                v-for="document in documentMatches"
                :key="document.id"
                type="button"
                @click="navigate('documents')"
              >
                <span class="command-file-icon">PDF</span>
                <span class="command-result-copy">
                  <strong>{{ document.filename }}</strong>
                  <small>{{ document.page_count }} pages · {{ document.chunk_count }} chunks</small>
                </span>
              </button>
            </div>
          </section>

          <section v-if="query.trim().length >= 2">
            <div class="command-heading-row">
              <span class="command-heading">Chunk matches</span>
              <span>{{ busy ? 'Searching…' : `${chunks.length} shown` }}</span>
            </div>
            <p v-if="error" class="command-error">{{ error }}</p>
            <div v-else-if="chunks.length" class="command-list chunk-list">
              <button
                v-for="chunk in chunks"
                :key="chunk.chunk_id"
                type="button"
                @click="navigate('search')"
              >
                <span class="command-result-copy">
                  <strong>{{ chunk.filename }}</strong>
                  <small>
                    p. {{ chunk.pages }}
                    <template v-if="chunk.section"> · {{ chunk.section }}</template>
                  </small>
                  <p>{{ chunk.text.slice(0, 190) }}{{ chunk.text.length > 190 ? '…' : '' }}</p>
                </span>
                <span class="command-score">{{ Math.round(chunk.score * 100) }}%</span>
              </button>
            </div>
            <p v-else-if="!busy" class="command-empty">No chunk matches yet.</p>
          </section>
        </div>

        <footer class="command-footer">
          <span>Enter · Ask AI</span>
          <span>Ctrl/⌘ + K · Open/close</span>
          <span>Esc · Close</span>
        </footer>
      </section>
    </div>
  </Teleport>
</template>

<style scoped>
.command-backdrop {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: start center;
  padding: min(12vh, 100px) 18px 30px;
  background: rgba(13, 28, 23, .38);
  backdrop-filter: blur(7px);
}
.command-palette {
  width: min(760px, 100%);
  max-height: min(76vh, 720px);
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr) auto;
  overflow: hidden;
  border: 1px solid rgba(211, 225, 219, .95);
  border-radius: 18px;
  background: #fff;
  box-shadow: 0 30px 100px rgba(15, 41, 32, .24);
}
.command-search-row {
  min-height: 60px;
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  padding: 0 15px;
  border-bottom: 1px solid #e1e9e5;
}
.command-search-icon { color: #527565; font-size: 20px; }
.command-search-row input {
  width: 100%;
  border: 0;
  outline: none;
  color: #263b32;
  font-size: 13px;
}
.command-search-row kbd {
  padding: 4px 7px;
  border: 1px solid #dbe4e0;
  border-radius: 7px;
  background: #f7faf8;
  color: #7b8983;
  font-size: 8px;
}
.command-actions {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 7px;
  padding: 10px 12px;
  border-bottom: 1px solid #edf1ef;
  background: #fbfdfc;
}
.command-actions button {
  min-width: 0;
  padding: 9px 10px;
  border: 1px solid #dde6e2;
  border-radius: 10px;
  background: #fff;
  text-align: left;
}
.command-actions strong,
.command-actions span { display: block; }
.command-actions strong { color: #345046; font-size: 9px; }
.command-actions span { margin-top: 2px; color: #87938e; font-size: 7.5px; }
.command-results { min-height: 0; overflow-y: auto; padding: 12px; }
.command-results section + section { margin-top: 14px; }
.command-heading-row { display: flex; justify-content: space-between; gap: 10px; }
.command-heading,
.command-heading-row > span { color: #7d8b85; font-size: 8px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.command-list { display: grid; gap: 5px; margin-top: 7px; }
.command-list button {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) auto;
  gap: 8px;
  align-items: start;
  padding: 8px;
  border: 0;
  border-radius: 9px;
  background: transparent;
  text-align: left;
}
.command-list button:hover { background: #f2f7f4; }
.command-file-icon {
  width: 27px;
  height: 27px;
  display: grid;
  place-items: center;
  border-radius: 7px;
  background: #f9eeee;
  color: #a55454;
  font-size: 7px;
  font-weight: 900;
}
.command-result-copy { min-width: 0; }
.command-result-copy strong,
.command-result-copy small { display: block; }
.command-result-copy strong { overflow: hidden; color: #33483f; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
.command-result-copy small { margin-top: 2px; color: #87938e; font-size: 7.5px; }
.command-result-copy p { margin: 4px 0 0; color: #687871; font-size: 8px; line-height: 1.45; }
.command-score { color: #36715a; font-size: 8px; font-weight: 800; }
.command-error { color: #9a4c4c; font-size: 8px; }
.command-empty { color: #89958f; font-size: 8px; }
.command-footer {
  min-height: 34px;
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 8px 16px;
  align-items: center;
  padding: 6px 12px;
  border-top: 1px solid #edf1ef;
  background: #fbfdfc;
  color: #8b9792;
  font-size: 7px;
}
@media (max-width: 640px) {
  .command-actions { grid-template-columns: 1fr; }
  .command-palette { max-height: 86vh; }
}
</style>
