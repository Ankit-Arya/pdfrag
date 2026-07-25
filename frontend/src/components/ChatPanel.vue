<script setup lang="ts">
import { nextTick, ref } from 'vue'
import type { AnswerResponse, CollectionResponse } from '../services/api'
import { renderMarkdown } from '../utils/markdown'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  response?: AnswerResponse
}

const props = defineProps<{
  collection: CollectionResponse | null
  busy: boolean
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
  if (!value || !props.collection || props.busy) return
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
  <main class="chat-shell">
    <header class="topbar">
      <div>
        <span class="mobile-title">DMRC QnA</span>
        <h1>Ask your documents</h1>
        <p>Each answer must be supported by the current PDF collection.</p>
      </div>
      <div
        class="memory-badge"
        title="Previous messages are never sent to the model"
      >
        <span aria-hidden="true">○</span> Memory off
      </div>
    </header>

    <section
      ref="scrollArea"
      class="conversation"
      aria-live="polite"
    >
      <div
        v-if="!messages.length"
        class="empty-state"
      >
        <div
          class="empty-orb"
          aria-hidden="true"
        >
          <svg viewBox="0 0 24 24">
            <path d="M4 5h16v12H8l-4 4z" />
            <path d="M8 9h8M8 13h5" />
          </svg>
        </div>
        <h2>
          {{ collection ? 'Your documents are ready' : 'Upload PDFs to begin' }}
        </h2>
        <p v-if="collection">
          Ask a specific question. Follow-up questions must be self-contained
          because chat memory is disabled.
        </p>
        <p v-else>
          Add one or more PDFs. The assistant will answer only from indexed
          document evidence.
        </p>
        <div
          v-if="collection"
          class="suggestion-grid"
        >
          <button @click="question = 'Summarize the key points in these documents.'">
            Summarize key points
          </button>
          <button @click="question = 'What are the main requirements or obligations?'">
            Find requirements
          </button>
          <button @click="question = 'List the important dates, amounts, and named entities.'">
            Extract key facts
          </button>
        </div>
      </div>

      <article
        v-for="message in messages"
        :key="message.id"
        class="message"
        :class="message.role"
      >
        <div
          class="avatar"
          aria-hidden="true"
        >
          {{ message.role === 'user' ? 'Y' : 'D' }}
        </div>

        <div class="message-content">
          <span class="message-label">
            {{ message.role === 'user' ? 'You' : 'DMRC QnA' }}
          </span>

          <div
            v-if="message.role === 'assistant'"
            class="message-text markdown-body"
            v-html="renderMarkdown(message.text)"
          />
          <div
            v-else
            class="message-text"
          >
            {{ message.text }}
          </div>

          <div
            v-if="
              message.response
              && !message.response.grounded
              && message.response.grounding_status === 'citation_validation_failed'
            "
            class="grounding-warning"
          >
            The response is shown, but its citation format did not pass automatic validation.
          </div>

          <div
            v-if="message.response?.sources.length"
            class="sources"
          >
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
                Similarity {{ source.score.toFixed(3) }}
              </span>
            </details>
          </div>
        </div>
      </article>

      <article
        v-if="busy"
        class="message assistant"
      >
        <div
          class="avatar"
          aria-hidden="true"
        >
          D
        </div>
        <div class="message-content">
          <span class="message-label">DMRC QnA</span>
          <div class="typing">
            <span />
            <span />
            <span />
          </div>
          <button
            class="cancel-link"
            @click="emit('cancel')"
          >
            Cancel
          </button>
        </div>
      </article>
    </section>

    <footer class="composer-wrap">
      <form
        class="composer"
        @submit.prevent="submit"
      >
        <textarea
          v-model="question"
          rows="1"
          :disabled="!collection || busy"
          :placeholder="
            collection
              ? 'Ask a self-contained question about your PDFs…'
              : 'Upload PDFs before asking a question'
          "
          aria-label="Question"
          @keydown="handleKeydown"
        />
        <button
          type="submit"
          :disabled="!collection || !question.trim() || busy"
          aria-label="Send question"
        >
          <svg viewBox="0 0 24 24">
            <path d="M5 12h14M13 6l6 6-6 6" />
          </svg>
        </button>
      </form>
      <p>
        Answers can be incomplete. Verify important details against the cited pages.
      </p>
    </footer>
  </main>
</template>
