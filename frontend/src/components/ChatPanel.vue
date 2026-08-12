<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  downloadDocument,
  listDocuments,
  type AnswerResponse,
  type ChatProgressEvent,
  type DocumentRecord,
  type KnowledgeStatus,
  type SourceResult,
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
  progress: ChatProgressEvent[]
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

const visibleProgress = computed(() => props.progress.slice(-6))
const currentProgress = computed(() => props.progress[props.progress.length - 1] ?? null)

function progressPercent(event: ChatProgressEvent): number | null {
  if (typeof event.current !== 'number' || typeof event.total !== 'number' || event.total <= 0) {
    return null
  }
  return Math.max(0, Math.min(100, Math.round((event.current / event.total) * 100)))
}

function isActiveProgress(event: ChatProgressEvent): boolean {
  return currentProgress.value?.stage === event.stage
}

interface EvidenceGroup {
  key: string
  filename: string
  pages: string
  section: string
  sources: SourceResult[]
}

function displayEvidenceExcerpt(value: string): string {
  const clean = value
    .replace(/\[PDF CHUNK CONTEXT\][\s\S]*?\[\/PDF CHUNK CONTEXT\]\s*/gi, '')
    .trim()
  if (!clean) return 'No readable excerpt was extracted. Verify the cited PDF page.'
  const lines = clean.split(/\r?\n/).filter((line) => line.trim())
  if (
    lines.length >= 2
    && lines[0].includes('|')
    && /-{3,}/.test(lines[1])
  ) {
    const cells = [lines[0], ...lines.slice(2)]
      .flatMap((line) => line.split('|'))
      .map((cell) => cell.replace(/[ *_`|]/g, '').trim())
    if (!cells.some((cell) => /[A-Za-z0-9]/.test(cell))) {
      return 'Table extraction contains no readable cells on this excerpt; verify the cited PDF page.'
    }
  }
  return clean
}

function groupEvidence(sources: SourceResult[] | undefined): EvidenceGroup[] {
  const groups = new Map<string, EvidenceGroup>()
  for (const source of sources ?? []) {
    const pages = source.pages?.trim() || String(source.page)
    const section = source.section?.trim() || ''
    const key = `${source.filename}::${pages}::${section}`
    const existing = groups.get(key)
    if (existing) existing.sources.push(source)
    else {
      groups.set(key, {
        key,
        filename: source.filename,
        pages,
        section,
        sources: [source],
      })
    }
  }
  return Array.from(groups.values())
}

watch(
  () => props.progress,
  async () => {
    if (!props.busy) return
    await nextTick()
    scrollArea.value?.scrollTo({
      top: scrollArea.value.scrollHeight,
      behavior: 'smooth',
    })
  },
  { deep: true },
)

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
            v-if="message.response?.evidence?.length"
            class="sources ai-evidence"
            @toggle="handleEvidenceToggle(message.id, $event)"
          >
            <summary class="sources-summary">
              <span>
                {{ message.response?.retrieval_mode === 'references'
                  ? 'Matching evidence'
                  : 'Evidence reviewed by AI' }}
              </span>
              <span class="sources-count">
                {{ message.response.evidence.length }}
                excerpt{{ message.response.evidence.length === 1 ? '' : 's' }}
              </span>
            </summary>
            <div v-if="evidenceIsExpanded(message.id)" class="evidence-groups">
              <section
                v-for="group in groupEvidence(message.response.evidence)"
                :key="`ai-${message.id}-${group.key}`"
                class="evidence-document-group"
              >
                <div class="evidence-document-heading">
                  <strong>{{ group.filename }}</strong>
                  <span>p. {{ group.pages }}</span>
                </div>
                <div v-if="group.section" class="evidence-section-title">{{ group.section }}</div>
                <div
                  v-for="source in group.sources"
                  :key="`ai-source-${message.id}-${source.id}`"
                  class="evidence-excerpt-row"
                >
                  <span class="source-id">{{ source.id }}</span>
                  <div
                    class="source-excerpt markdown-body"
                    v-html="renderMarkdown(displayEvidenceExcerpt(source.excerpt))"
                  />
                </div>
              </section>
            </div>
          </details>

          <details
            v-if="message.response?.retrieval_mode !== 'references' && message.response?.sources.length"
            class="sources"
          >
            <summary class="sources-summary">
              <span>Retrieved evidence</span>
              <span class="sources-count">
                {{ message.response.sources.length }} cited
                source{{ message.response.sources.length === 1 ? '' : 's' }}
              </span>
            </summary>
            <div class="evidence-groups cited-evidence-groups">
              <section
                v-for="group in groupEvidence(message.response.sources)"
                :key="`cited-${message.id}-${group.key}`"
                class="evidence-document-group"
              >
                <div class="evidence-document-heading">
                  <strong>{{ group.filename }}</strong>
                  <span>p. {{ group.pages }}</span>
                </div>
                <div v-if="group.section" class="evidence-section-title">{{ group.section }}</div>
                <div
                  v-for="source in group.sources"
                  :key="`cited-source-${message.id}-${source.id}`"
                  class="evidence-excerpt-row"
                >
                  <span class="source-id">{{ source.id }}</span>
                  <div
                    class="source-excerpt markdown-body"
                    v-html="renderMarkdown(displayEvidenceExcerpt(source.excerpt))"
                  />
                </div>
              </section>
            </div>
          </details>
        </div>
      </article>

      <article v-if="busy" class="message assistant work-message">
        <div class="avatar" aria-hidden="true">D</div>
        <div class="message-content">
          <span class="message-label">DMRC Q&A</span>
          <section class="work-progress" aria-live="polite" aria-label="Answer preparation progress">
            <div class="work-progress-current">
              <span class="work-spinner" aria-hidden="true" />
              <div>
                <strong>{{ currentProgress?.label || 'Starting document analysis' }}</strong>
                <p v-if="currentProgress?.detail">{{ currentProgress.detail }}</p>
                <p v-else>Connecting to the document retrieval pipeline…</p>
              </div>
            </div>

            <div
              v-if="currentProgress && progressPercent(currentProgress) !== null"
              class="work-progress-meter"
              role="progressbar"
              :aria-valuenow="progressPercent(currentProgress) ?? undefined"
              aria-valuemin="0"
              aria-valuemax="100"
            >
              <span :style="{ width: `${progressPercent(currentProgress) ?? 0}%` }" />
            </div>

            <ol v-if="visibleProgress.length" class="work-progress-history">
              <li
                v-for="event in visibleProgress"
                :key="event.stage"
                :class="{ active: isActiveProgress(event) }"
              >
                <span class="work-step-icon" aria-hidden="true">
                  <i v-if="isActiveProgress(event)" class="work-step-spinner" />
                  <svg v-else viewBox="0 0 16 16">
                    <path d="M3.2 8.1 6.5 11.2 12.8 4.9" />
                  </svg>
                </span>
                <span class="work-step-copy">
                  <strong>{{ event.label }}</strong>
                  <small v-if="event.detail">{{ event.detail }}</small>
                </span>
                <span
                  v-if="typeof event.current === 'number' && typeof event.total === 'number'"
                  class="work-step-count"
                >
                  {{ event.current }}/{{ event.total }}
                </span>
              </li>
            </ol>
          </section>
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
.work-message .message-content {
  width: min(720px, calc(100% - 52px));
}

.work-progress {
  margin-top: 8px;
  padding: 13px 14px;
  border: 1px solid #dce6e1;
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(249, 252, 250, .98), rgba(244, 249, 246, .92));
  box-shadow: 0 8px 24px rgba(34, 67, 55, .045);
}

.work-progress-current {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.work-progress-current strong {
  display: block;
  color: #293a34;
  font-size: 11px;
  line-height: 1.4;
}

.work-progress-current p {
  margin: 3px 0 0;
  color: #76857f;
  font-size: 9px;
  line-height: 1.45;
}

.work-spinner,
.work-step-spinner {
  display: inline-block;
  border-radius: 999px;
  border: 2px solid #d9e5df;
  border-top-color: #477b68;
  animation: work-spin .85s linear infinite;
}

.work-spinner {
  width: 15px;
  height: 15px;
  margin-top: 1px;
  flex: 0 0 auto;
}

.work-step-spinner {
  width: 9px;
  height: 9px;
}

.work-progress-meter {
  height: 3px;
  margin: 10px 0 4px 25px;
  overflow: hidden;
  border-radius: 999px;
  background: #e3ebe7;
}

.work-progress-meter > span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: #628f7d;
  transition: width .25s ease;
}

.work-progress-history {
  margin: 10px 0 0 2px;
  padding: 0;
  display: grid;
  gap: 6px;
  list-style: none;
}

.work-progress-history li {
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr) auto;
  align-items: start;
  gap: 7px;
  color: #82908b;
  opacity: .78;
}

.work-progress-history li.active {
  color: #355f50;
  opacity: 1;
}

.work-step-icon {
  width: 18px;
  height: 18px;
  display: inline-grid;
  place-items: center;
}

.work-step-icon svg {
  width: 11px;
  height: 11px;
  fill: none;
  stroke: currentColor;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 1.8;
}

.work-step-copy strong,
.work-step-copy small {
  display: block;
}

.work-step-copy strong {
  font-size: 9px;
  line-height: 1.4;
}

.work-step-copy small {
  margin-top: 1px;
  color: #8d9995;
  font-size: 8px;
  line-height: 1.35;
}

.work-step-count {
  padding-top: 1px;
  color: #8a9892;
  font-size: 8px;
  font-variant-numeric: tabular-nums;
}

@keyframes work-spin {
  to { transform: rotate(360deg); }
}

@media (prefers-reduced-motion: reduce) {
  .work-spinner,
  .work-step-spinner {
    animation-duration: 1.8s;
  }
}

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

.formatted-evidence-wrap {
  padding: 0 7px 7px;
}

.formatted-evidence {
  padding: 12px 14px;
  border-top: 1px solid #e7ecea;
  color: #46534f;
  font-size: 11px;
  line-height: 1.65;
}

.formatted-evidence :deep(h3) {
  margin: 14px 0 5px;
  color: #263b34;
  font-size: 12px;
}

.formatted-evidence :deep(h3:first-child) {
  margin-top: 0;
}

.formatted-evidence :deep(h4) {
  margin: 8px 0 4px;
  color: #53655f;
  font-size: 10px;
}

.formatted-evidence :deep(p),
.formatted-evidence :deep(ul),
.formatted-evidence :deep(ol),
.formatted-evidence :deep(.table-scroll) {
  margin-top: 6px;
  margin-bottom: 8px;
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

.evidence-groups {
  display: grid;
  gap: 10px;
  margin-top: 9px;
}

.evidence-document-group {
  border: 1px solid #dce5e1;
  border-radius: 10px;
  background: #fbfdfc;
  overflow: hidden;
}

.evidence-document-heading {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 9px 11px;
  background: #f3f7f5;
  border-bottom: 1px solid #e3ebe7;
  font-size: 11px;
}

.evidence-document-heading strong {
  min-width: 0;
  overflow-wrap: anywhere;
}

.evidence-document-heading span {
  flex: none;
  color: #6e7c76;
}

.evidence-section-title {
  padding: 8px 11px 0;
  font-size: 10px;
  font-weight: 700;
  color: #5e6d67;
}

.evidence-excerpt-row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 8px;
  padding: 9px 11px;
  border-top: 1px solid #edf2ef;
}

.evidence-section-title + .evidence-excerpt-row {
  border-top: 0;
}

.evidence-excerpt-row .source-id {
  align-self: start;
  margin-top: 2px;
}

.evidence-excerpt-row .source-excerpt {
  min-width: 0;
}

</style>
