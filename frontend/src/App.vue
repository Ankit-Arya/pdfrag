<script setup lang="ts">
import { ref } from 'vue'
import ChatPanel from './components/ChatPanel.vue'
import UploadPanel from './components/UploadPanel.vue'
import {
  askQuestion,
  createCollection,
  deleteCollection,
  type AnswerResponse,
  type CollectionResponse,
} from './services/api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  response?: AnswerResponse
}

const collection = ref<CollectionResponse | null>(null)
const uploading = ref(false)
const answering = ref(false)
const question = ref('')
const messages = ref<Message[]>([])
const error = ref('')
let controller: AbortController | null = null

function id(): string {
  return crypto.randomUUID()
}

async function upload(files: File[]): Promise<void> {
  uploading.value = true
  error.value = ''
  try {
    collection.value = await createCollection(files)
    messages.value = []
    question.value = ''
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Upload failed.'
  } finally {
    uploading.value = false
  }
}

async function ask(value: string): Promise<void> {
  if (!collection.value) return
  messages.value.push({ id: id(), role: 'user', text: value })
  question.value = ''
  answering.value = true
  error.value = ''
  controller = new AbortController()
  try {
    const response = await askQuestion(collection.value.collection_id, value, controller.signal)
    messages.value.push({
      id: id(),
      role: 'assistant',
      text: response.answer,
      response,
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') return
    error.value = cause instanceof Error ? cause.message : 'Question failed.'
  } finally {
    answering.value = false
    controller = null
  }
}

function cancel(): void {
  controller?.abort()
}

async function reset(): Promise<void> {
  cancel()
  const current = collection.value
  collection.value = null
  messages.value = []
  question.value = ''
  error.value = ''
  if (current) {
    try {
      await deleteCollection(current.collection_id)
    } catch {
      // The collection may already have expired; local reset still succeeds.
    }
  }
}
</script>

<template>
  <div class="app-layout">
    <UploadPanel
      :collection="collection"
      :busy="uploading || answering"
      @upload="upload"
      @reset="reset"
    />
    <ChatPanel
      v-model:question="question"
      v-model:messages="messages"
      :collection="collection"
      :busy="answering"
      @ask="ask"
      @cancel="cancel"
    />
    <div v-if="error" class="error-toast" role="alert">
      <span>!</span>
      <p>{{ error }}</p>
      <button aria-label="Dismiss error" @click="error = ''">×</button>
    </div>
  </div>
</template>
