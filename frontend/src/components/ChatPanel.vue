<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import {
  downloadDocument,
  listDocuments,
  type AnswerResponse,
  type DocumentRecord,
  type KnowledgeStatus,
} from '../services/api'
import { renderMarkdown } from '../utils/markdown'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  createdAt: string
  response?: AnswerResponse
}

const props = defineProps<{
  knowledge: KnowledgeStatus | null
  busy: boolean
  activeChatTitle: string | null
}>()

const emit = defineEmits<{
  ask: [question: string]
  cancel: []
}>()

const question = defineModel<string>('question', { required: true })
const messages = defineModel<Message[]>('messages', { required: true })
const scrollArea = ref<HTMLElement | null>(null)
const documents = ref<DocumentRecord[]>([])
const libraryBusy = ref(false)
const libraryError = ref('')
const copiedMessageId = ref<string | null>(null)
const expandedEvidenceIds = ref<Set<string>>(new Set())
let copyResetTimer: number | null = null
const messageDateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
})

function formatMessageTime(value: string): string {
  const timestamp = new Date(value)
  if (Number.isNaN(timestamp.getTime())) return ''
  return messageDateTimeFormatter.format(timestamp)
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

async function loadDocumentLibrary(): Promise<void> {
  libraryBusy.value = true
  libraryError.value = ''
  try {
    documents.value = await listDocuments()
  } catch (cause) {
    libraryError.value = cause instanceof Error ? cause.message : 'Could not load PDFs.'
  } finally {
    libraryBusy.value = false
  }
}

async function downloadPdf(documentRecord: DocumentRecord): Promise<void> {
  libraryBusy.value = true
  libraryError.value = ''
  try {
    await downloadDocument(documentRecord.id, documentRecord.filename)
  } catch (cause) {
    libraryError.value = cause instanceof Error ? cause.message : 'Could not download this PDF.'
  } finally {
    libraryBusy.value = false
  }
}

async function submit(): Promise<void> {
  const value = question.value.trim()
  if (!value || !props.knowledge?.ready_documents || props.busy) return
  emit('ask', value)
  await nextTick()
  scrollArea.value?.scrollTo({
    top: scrollArea.value.scrollHeight,
    behavior: 'smooth',
  })
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void submit()
  }
}

async function copyAnswer(message: Message): Promise<void> {
  if (message.role !== 'assistant') return
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(message.text)
    } else {
      copyWithLegacyFallback(message.text)
    }
  } catch {
    copyWithLegacyFallback(message.text)
  }

  copiedMessageId.value = message.id
  if (copyResetTimer !== null) window.clearTimeout(copyResetTimer)
  copyResetTimer = window.setTimeout(() => {
    if (copiedMessageId.value === message.id) copiedMessageId.value = null
    copyResetTimer = null
  }, 1800)
}

function copyWithLegacyFallback(value: string): void {
  const textarea = window.document.createElement('textarea')
  textarea.value = value
  textarea.setAttribute('readonly', '')
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  window.document.body.appendChild(textarea)
  textarea.select()
  window.document.execCommand('copy')
  textarea.remove()
}

function handleEvidenceToggle(messageId: string, event: Event): void {
  const details = event.currentTarget as HTMLDetailsElement
  const updated = new Set(expandedEvidenceIds.value)
  if (details.open) updated.add(messageId)
  else updated.delete(messageId)
  expandedEvidenceIds.value = updated
}

function evidenceIsExpanded(messageId: string): boolean {
  return expandedEvidenceIds.value.has(messageId)
}

onMounted(() => {
  void loadDocumentLibrary()
})

onBeforeUnmount(() => {
  if (copyResetTimer !== null) window.clearTimeout(copyResetTimer)
})
</script>

<template>
  <main class="content-shell chat-shell">
    <header class="topbar">
      <div>
        <span class="eyebrow">Document-grounded assistant</span>
        <h1>{{ activeChatTitle || 'Ask the shared knowledge base' }}</h1>
        <p>
          Answers are generated only from PDFs that an administrator has processed.
        </p>
      </div>
      <div class="topbar-badges">
        <span class="memory-badge"><i /> History saved</span>
        <span class="knowledge-badge">
          {{ knowledge?.ready_documents ?? 0 }} docs · {{ knowledge?.total_chunks ?? 0 }} chunks
        </span>
      </div>
    </header>

    <section ref="scrollArea" class="conversation" aria-live="polite">
      <section class="document-library-card" aria-labelledby="document-library-title">
        <div class="document-library-heading">
          <div>
            <span class="eyebrow">Available PDFs</span>
            <h2 id="document-library-title">
              {{ documents.length }} uploaded PDF{{ documents.length === 1 ? '' : 's' }}
            </h2>
            <p>All signed-in users can view and download the shared PDF library.</p>
          </div>
          <button
            type="button"
            class="ghost-action compact"
            :disabled="libraryBusy"
            @click="loadDocumentLibrary"
          >
            {{ libraryBusy ? 'Loading…' : 'Refresh PDFs' }}
          </button>
        </div>

        <p v-if="libraryError" class="library-error">{{ libraryError }}</p>
        <p v-else-if="!documents.length" class="library-empty">
          No PDFs have been uploaded yet.
        </p>
        <div v-else class="library-doc-list">
          <article
            v-for="documentRecord in documents"
            :key="documentRecord.id"
            class="library-doc-row"
          >
            <div class="file-icon">PDF</div>
            <div class="library-doc-main">
              <strong>{{ documentRecord.filename }}</strong>
              <span>
                {{ formatBytes(documentRecord.size_bytes) }} · {{ documentRecord.page_count }} pages ·
                {{ documentRecord.chunk_count }} chunks
              </span>
              <small v-if="documentRecord.error" class="row-error">{{ documentRecord.error }}</small>
            </div>
            <span class="status-tag" :class="documentRecord.status">{{ documentRecord.status }}</span>
            <button
              type="button"
              class="library-download-button"
              :disabled="libraryBusy"
              @click="downloadPdf(documentRecord)"
            >
              Download
            </button>
          </article>
        </div>
      </section>

      <div v-if="!messages.length" class="empty-state">
        <div class="empty-orb" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path d="M4 5h16v12H8l-4 4z" />
            <path d="M8 9h8M8 13h5" />
          </svg>
        </div>
        <template v-if="knowledge?.ready_documents">
          <h2>Knowledge is ready</h2>
          <p>
            Ask a direct fact, a detailed procedure, or a broad document-reference search.
            The assistant uses chat context for intent while grounding facts in the PDFs.
          </p>
          <div class="suggestion-grid">
            <button @click="question = 'Summarize the main operating procedures in the documents.'">
              Summarize procedures
            </button>
            <button @click="question = 'What are the key requirements, responsibilities, and exceptions?'">
              Find requirements
            </button>
            <button @click="question = 'List important dates, identifiers, and named entities.'">
              Extract key facts
            </button>
          </div>
        </template>
        <template v-else>
          <h2>No processed documents yet</h2>
          <p>
            An administrator must upload and process at least one PDF before users can
            ask questions.
          </p>
        </template>
      </div>

      <article
        v-for="message in messages"
        :key="message.id"
        class="message"
        :class="message.role"
      >
        <div class="avatar" aria-hidden="true">
          {{ message.role === 'user' ? 'Y' : 'D' }}
        </div>
        <div class="message-content">
          <div class="message-heading">
            <span class="message-label">
              {{ message.role === 'user' ? 'You' : 'DMRC Q&A' }}
            </span>
            <time
              v-if="formatMessageTime(message.createdAt)"
              class="message-timestamp"
              :datetime="message.createdAt"
              :title="formatMessageTime(message.createdAt)"
            >
              {{ formatMessageTime(message.createdAt) }}
            </time>
          </div>
          <div
            v-if="message.role === 'assistant'"
            class="message-text markdown-body"
            v-html="renderMarkdown(message.text)"
          />
          <div v-else class="message-text">{{ message.text }}</div>

          <div v-if="message.role === 'assistant'" class="message-actions">
            <button
              type="button"
              class="copy-answer-button"
              :aria-label="copiedMessageId === message.id ? 'Answer copied' : 'Copy answer'"
              @click="copyAnswer(message)"
            >
              {{ copiedMessageId === message.id ? 'Copied' : 'Copy answer' }}
            </button>
          </div>

          <div
            v-if="
              message.response
              && !message.response.grounded
              && message.response.grounding_status === 'citation_validation_failed'
            "
            class="grounding-warning"
          >
            This answer is shown, but its citation format did not pass automatic validation.
          </div>

          <details
            v-if="
              message.response?.retrieval_mode !== 'references'
              && message.response?.evidence?.length
            "
            class="sources ai-evidence"
            @toggle="handleEvidenceToggle(message.id, $event)"
          >
            <summary class="sources-summary">
              <span>Evidence reviewed by AI</span>
              <span class="sources-count">
                {{ message.response.evidence.length }}
                chunk{{ message.response.evidence.length === 1 ? '' : 's' }}
              </span>
            </summary>
            <div v-if="evidenceIsExpanded(message.id)" class="source-list">
              <details
                v-for="source in message.response.evidence"
                :key="`ai-${message.id}-${source.id}`"
                class="source-card"
              >
                <summary>
                  <span class="source-id">{{ source.id }}</span>
                  <span class="source-title">{{ source.filename }}</span>
                  <span class="source-page">p. {{ source.page }}</span>
                </summary>
                <div
                  class="source-excerpt markdown-body"
                  v-html="renderMarkdown(source.excerpt)"
                />
                <span class="score">
                  {{ source.retrieval_method }} · score {{ source.score.toFixed(3) }}
                </span>
              </details>
            </div>
          </details>

          <details v-if="message.response?.sources.length" class="sources">
            <summary class="sources-summary">
              <span>Retrieved evidence</span>
              <span class="sources-count">
                {{ message.response.sources.length }} cited
                source{{ message.response.sources.length === 1 ? '' : 's' }}
              </span>
            </summary>
            <div class="source-list">
              <details
                v-for="source in message.response.sources"
                :key="source.id"
                class="source-card"
              >
                <summary>
                  <span class="source-id">{{ source.id }}</span>
                  <span class="source-title">{{ source.filename }}</span>
                  <span class="source-page">p. {{ source.page }}</span>
                </summary>
                <div
                  class="source-excerpt markdown-body"
                  v-html="renderMarkdown(source.excerpt)"
                />
                <span class="score">
                  {{ source.retrieval_method }} · score {{ source.score.toFixed(3) }}
                </span>
              </details>
            </div>
          </details>
        </div>
      </article>

      <article v-if="busy" class="message assistant">
        <div class="avatar" aria-hidden="true">D</div>
        <div class="message-content">
          <span class="message-label">DMRC Q&A</span>
          <div class="typing"><span /><span /><span /></div>
          <button class="cancel-link" @click="emit('cancel')">Cancel</button>
        </div>
      </article>
    </section>

    <footer class="composer-wrap">
      <form class="composer" @submit.prevent="submit">
        <textarea
          v-model="question"
          rows="1"
          :disabled="!knowledge?.ready_documents || busy"
          :placeholder="
            knowledge?.ready_documents
              ? 'Ask the PDFs…'
              : 'Waiting for an administrator to process documents'
          "
          aria-label="Question"
          @keydown="handleKeydown"
        />
        <button
          type="submit"
          :disabled="!knowledge?.ready_documents || !question.trim() || busy"
          aria-label="Send question"
        >
          <svg viewBox="0 0 24 24">
            <path d="M5 12h14M13 6l6 6-6 6" />
          </svg>
        </button>
      </form>
      <p>Verify important details against the cited PDF pages.</p>
    </footer>
  </main>
</template>

<style scoped>
.message-heading {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
}

.message-timestamp {
  color: #89958f;
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.4;
  white-space: nowrap;
}

.message.user .message-heading {
  justify-content: flex-end;
}

.message-actions {
  margin-top: 7px;
  display: flex;
  align-items: center;
  gap: 7px;
}

.copy-answer-button {
  min-height: 27px;
  padding: 0 8px;
  border: 1px solid #d4dfda;
  border-radius: 8px;
  color: #61726c;
  background: rgba(255, 255, 255, .76);
  font-size: 9px;
  font-weight: 800;
}

.copy-answer-button:hover {
  border-color: #99b8ac;
  color: #265b49;
  background: #fff;
}

.ai-evidence {
  margin-top: 12px;
  background: rgba(244, 250, 247, .78);
}

.source-excerpt {
  margin: 0;
  padding: 0 13px 9px 48px;
  color: #5e6966;
  font-size: 10px;
  line-height: 1.6;
  overflow-wrap: anywhere;
}

.source-excerpt :deep(p) {
  margin: 0 0 7px;
}

.source-excerpt :deep(p:last-child) {
  margin-bottom: 0;
}

.source-excerpt :deep(table) {
  font-size: 9px;
}

.document-library-card {
  max-width: 880px;
  margin: 0 auto 26px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: rgba(255, 255, 255, .9);
  box-shadow: 0 8px 24px rgba(36, 69, 58, .04);
}

.document-library-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.document-library-heading h2 {
  margin: 5px 0 0;
  font-size: 18px;
  letter-spacing: -.03em;
}

.document-library-heading p,
.library-empty,
.library-error {
  margin: 7px 0 0;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.6;
}

.library-error,
.row-error {
  color: #a0362d;
}

.library-doc-list {
  margin-top: 12px;
  display: grid;
}

.library-doc-row {
  min-width: 0;
  padding: 13px 0;
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 12px;
  border-top: 1px solid #e7ecea;
}

.library-doc-row:first-child {
  border-top: 0;
}

.library-doc-main {
  min-width: 0;
}

.library-doc-main strong {
  display: block;
  overflow: hidden;
  color: #283530;
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.library-doc-main span,
.library-doc-main small {
  display: block;
  margin-top: 4px;
  color: #7d8a85;
  font-size: 9px;
  line-height: 1.45;
}

.library-download-button {
  min-height: 30px;
  padding: 0 10px;
  border: 1px solid #d2ddd8;
  border-radius: 8px;
  color: #466158;
  background: #fff;
  font-size: 9px;
  font-weight: 800;
}

.library-download-button:hover:not(:disabled) {
  border-color: #93b2a6;
  background: #f7faf9;
}

@media (max-width: 820px) {
  .document-library-heading {
    display: grid;
  }

  .library-doc-row {
    grid-template-columns: 34px minmax(0, 1fr);
  }

  .library-doc-row .status-tag,
  .library-doc-row .library-download-button {
    grid-column: 2;
    justify-self: start;
  }
}
</style>
