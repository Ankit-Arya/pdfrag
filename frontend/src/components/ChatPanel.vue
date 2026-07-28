<script setup lang="ts">
import { nextTick, ref } from 'vue'
import type { AnswerResponse, KnowledgeStatus } from '../services/api'
import { renderMarkdown } from '../utils/markdown'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
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
            Start a new question or open a saved chat. Chat history is stored, but each
            question should still include enough context to stand on its own.
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
          <span class="message-label">
            {{ message.role === 'user' ? 'You' : 'DMRC Q&A' }}
          </span>
          <div
            v-if="message.role === 'assistant'"
            class="message-text markdown-body"
            v-html="renderMarkdown(message.text)"
          />
          <div v-else class="message-text">{{ message.text }}</div>

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

          <div v-if="message.response?.sources.length" class="sources">
            <span class="eyebrow">Retrieved evidence</span>
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
              <p>{{ source.excerpt }}</p>
              <span class="score">
                {{ source.retrieval_method }} · score {{ source.score.toFixed(3) }}
              </span>
            </details>
          </div>
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
              ? 'Ask a self-contained question about the shared PDFs…'
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
