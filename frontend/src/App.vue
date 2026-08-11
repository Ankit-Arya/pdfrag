<script setup lang="ts">
import { onMounted, ref } from 'vue'
import AccountPanel from './components/AccountPanel.vue'
import AdminPanel from './components/AdminPanel.vue'
import ChatPanel from './components/ChatPanel.vue'
import LoginPanel from './components/LoginPanel.vue'
import UploadPanel from './components/UploadPanel.vue'
import {
  askQuestion,
  clearStoredSession,
  createUser,
  currentSessionId as getCurrentSessionId,
  deleteChat as deleteStoredChat,
  deleteDocument,
  getChat,
  getCurrentUser,
  getKnowledgeStatus,
  hasStoredSession,
  listAuthSessions,
  listChats,
  listDocuments,
  listUsers,
  login,
  logout,
  processDocuments,
  revokeAuthSession,
  setUserActive,
  uploadDocument,
  type AdminUser,
  type AnswerResponse,
  type AuthSession,
  type ChatSession,
  type DocumentRecord,
  type KnowledgeStatus,
  type SourceResult,
  type User,
} from './services/api'

type ViewName = 'chat' | 'admin' | 'account'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  createdAt: string
  response?: AnswerResponse
}

const initializing = ref(true)
const authBusy = ref(false)
const operationBusy = ref(false)
const answering = ref(false)
const user = ref<User | null>(null)
const view = ref<ViewName>('chat')
const knowledge = ref<KnowledgeStatus | null>(null)
const documents = ref<DocumentRecord[]>([])
const users = ref<AdminUser[]>([])
const sessions = ref<AuthSession[]>([])
const chats = ref<ChatSession[]>([])
const activeChatId = ref<string | null>(null)
const activeChatTitle = ref<string | null>(null)
const messages = ref<Message[]>([])
const question = ref('')
const error = ref('')
let controller: AbortController | null = null

const DOCUMENT_POLL_INTERVAL_MS = 1500
const DOCUMENT_POLL_TIMEOUT_MS = 30 * 60 * 1000


function id(): string {
  const bytes = new Uint8Array(16)
  crypto.getRandomValues(bytes)

  // RFC 4122 version 4 UUID. getRandomValues() is available on HTTP LAN origins,
  // unlike crypto.randomUUID(), which requires a secure context.
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80

  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0'))
  return [
    hex.slice(0, 4).join(''),
    hex.slice(4, 6).join(''),
    hex.slice(6, 8).join(''),
    hex.slice(8, 10).join(''),
    hex.slice(10, 16).join(''),
  ].join('-')
}

function messageResponse(
  content: string,
  metadata: Record<string, unknown>,
): AnswerResponse {
  const sources = Array.isArray(metadata.sources)
    ? (metadata.sources as SourceResult[])
    : []
  const evidence = Array.isArray(metadata.evidence)
    ? (metadata.evidence as SourceResult[])
    : []
  return {
    answer: content,
    sources,
    evidence,
    formatted_sources:
      typeof metadata.formatted_sources === 'string' ? metadata.formatted_sources : '',
    formatted_evidence:
      typeof metadata.formatted_evidence === 'string' ? metadata.formatted_evidence : '',
    grounded: metadata.grounded === true,
    grounding_status:
      typeof metadata.grounding_status === 'string'
        ? metadata.grounding_status
        : metadata.grounded === true
          ? 'verified'
          : 'insufficient_evidence',
    interpreted_question:
      typeof metadata.interpreted_question === 'string'
        ? metadata.interpreted_question
        : null,
    contextual_question:
      typeof metadata.contextual_question === 'string'
        ? metadata.contextual_question
        : null,
    retrieval_mode:
      typeof metadata.retrieval_mode === 'string'
        ? metadata.retrieval_mode
        : 'answer',
    resolved_abbreviations: Array.isArray(metadata.resolved_abbreviations)
      ? (metadata.resolved_abbreviations as string[])
      : [],
    routing_hints: Array.isArray(metadata.routing_hints)
      ? (metadata.routing_hints as string[])
      : [],
    primary_documents: Array.isArray(metadata.primary_documents)
      ? (metadata.primary_documents as string[])
      : [],
    candidate_chunks:
      typeof metadata.candidate_chunks === 'number' ? metadata.candidate_chunks : 0,
    evidence_chunks:
      typeof metadata.evidence_chunks === 'number' ? metadata.evidence_chunks : evidence.length,
    search_queries: Array.isArray(metadata.search_queries)
      ? (metadata.search_queries as string[])
      : [],
    request_id:
      typeof metadata.request_id === 'string' ? metadata.request_id : null,
    chat_session_id: activeChatId.value,
  }
}

function showError(cause: unknown, fallback: string): void {
  error.value = cause instanceof Error ? cause.message : fallback
  if (!hasStoredSession()) resetLocalSession()
}

function resetLocalSession(): void {
  user.value = null
  knowledge.value = null
  documents.value = []
  users.value = []
  sessions.value = []
  chats.value = []
  activeChatId.value = null
  activeChatTitle.value = null
  messages.value = []
  question.value = ''
  view.value = 'chat'
}

function errorMessage(cause: unknown, fallback: string): string {
  return cause instanceof Error ? cause.message : fallback
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

async function waitForDocumentProcessing(documentIds: string[]): Promise<DocumentRecord[]> {
  const trackedIds = new Set(documentIds)
  const started = Date.now()

  while (true) {
    const latest = await listDocuments()
    documents.value = latest
    const tracked = latest.filter((document) => trackedIds.has(document.id))
    const stillProcessing = tracked.some((document) => document.status === 'processing')

    if (!stillProcessing) {
      knowledge.value = await getKnowledgeStatus()
      return tracked
    }

    if (Date.now() - started >= DOCUMENT_POLL_TIMEOUT_MS) {
      throw new Error(
        'Document processing is still running in the background. You can refresh this page later.',
      )
    }
    await sleep(DOCUMENT_POLL_INTERVAL_MS)
  }
}

async function queueDocumentBatch(documentIds: string[]): Promise<DocumentRecord[]> {
  const uniqueIds = [...new Set(documentIds)]
  if (!uniqueIds.length) return []
  await processDocuments(uniqueIds)
  return waitForDocumentProcessing(uniqueIds)
}

function failedBatchMessage(records: DocumentRecord[]): string | null {
  const failed = records.filter((document) => document.status === 'failed')
  if (!failed.length) return null
  const names = failed.slice(0, 4).map((document) => document.filename).join(', ')
  const suffix = failed.length > 4 ? ` and ${failed.length - 4} more` : ''
  return `${failed.length} document(s) failed processing: ${names}${suffix}`
}

async function loadCommonData(): Promise<void> {
  const [knowledgeResult, chatsResult, sessionsResult] = await Promise.all([
    getKnowledgeStatus(),
    listChats(),
    listAuthSessions(),
  ])
  knowledge.value = knowledgeResult
  chats.value = chatsResult
  sessions.value = sessionsResult
}

async function loadAdminData(): Promise<void> {
  if (user.value?.role !== 'admin') return
  const [documentResult, userResult] = await Promise.all([
    listDocuments(),
    listUsers(),
  ])
  documents.value = documentResult
  users.value = userResult
}

async function loadWorkspace(): Promise<void> {
  await Promise.all([loadCommonData(), loadAdminData()])
}

async function bootstrap(): Promise<void> {
  initializing.value = true
  try {
    if (!hasStoredSession()) return
    user.value = await getCurrentUser()
    await loadWorkspace()
  } catch (cause) {
    clearStoredSession()
    resetLocalSession()
    showError(cause, 'Your session could not be restored.')
  } finally {
    initializing.value = false
  }
}

async function signIn(email: string, password: string): Promise<void> {
  authBusy.value = true
  error.value = ''
  try {
    user.value = await login(email, password)
    await loadWorkspace()
  } catch (cause) {
    showError(cause, 'Sign-in failed.')
  } finally {
    authBusy.value = false
  }
}

async function signOut(): Promise<void> {
  cancel()
  operationBusy.value = true
  try {
    await logout()
  } finally {
    resetLocalSession()
    operationBusy.value = false
  }
}

function navigate(nextView: ViewName): void {
  view.value = nextView
}

function newChat(): void {
  cancel()
  activeChatId.value = null
  activeChatTitle.value = null
  messages.value = []
  question.value = ''
  view.value = 'chat'
}

async function openChat(chatId: string): Promise<void> {
  operationBusy.value = true
  error.value = ''
  try {
    const chat = await getChat(chatId)
    activeChatId.value = chat.id
    activeChatTitle.value = chat.title
    messages.value = chat.messages
      .filter((message) => message.role === 'user' || message.role === 'assistant')
      .map((message) => ({
        id: message.id,
        role: message.role as 'user' | 'assistant',
        text: message.content,
        createdAt: message.created_at,
        response:
          message.role === 'assistant'
            ? messageResponse(message.content, message.metadata)
            : undefined,
      }))
    question.value = ''
    view.value = 'chat'
  } catch (cause) {
    showError(cause, 'Could not open this chat.')
  } finally {
    operationBusy.value = false
  }
}

async function removeChat(chatId: string): Promise<void> {
  if (!window.confirm('Delete this saved chat?')) return
  operationBusy.value = true
  error.value = ''
  try {
    await deleteStoredChat(chatId)
    if (activeChatId.value === chatId) newChat()
    chats.value = await listChats()
  } catch (cause) {
    showError(cause, 'Could not delete this chat.')
  } finally {
    operationBusy.value = false
  }
}

async function ask(value: string): Promise<void> {
  const questionMessageId = id()
  messages.value.push({
    id: questionMessageId,
    role: 'user',
    text: value,
    createdAt: new Date().toISOString(),
  })
  question.value = ''
  answering.value = true
  error.value = ''
  controller = new AbortController()
  try {
    const response = await askQuestion(value, activeChatId.value, controller.signal)
    activeChatId.value = response.chat_session_id ?? activeChatId.value
    if (response.question_created_at) {
      const savedQuestion = messages.value.find(
        (message) => message.id === questionMessageId,
      )
      if (savedQuestion) savedQuestion.createdAt = response.question_created_at
    }
    messages.value.push({
      id: id(),
      role: 'assistant',
      text: response.answer,
      createdAt: response.response_created_at ?? new Date().toISOString(),
      response,
    })
    chats.value = await listChats()
    const active = chats.value.find((chat) => chat.id === activeChatId.value)
    activeChatTitle.value = active?.title ?? activeChatTitle.value
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') return
    showError(cause, 'Question failed.')
  } finally {
    answering.value = false
    controller = null
  }
}

function cancel(): void {
  controller?.abort()
}

async function uploadFiles(files: File[]): Promise<void> {
  operationBusy.value = true
  error.value = ''
  const uploaded: DocumentRecord[] = []
  const failures: string[] = []

  try {
    // Stage every file first. A processing failure can no longer prevent later
    // PDFs from being safely stored and queued.
    for (const file of files) {
      try {
        const document = await uploadDocument(file, false)
        uploaded.push(document)
      } catch (cause) {
        failures.push(`${file.name}: ${errorMessage(cause, 'upload failed')}`)
      }
    }

    if (uploaded.length) {
      try {
        const processed = await queueDocumentBatch(uploaded.map((document) => document.id))
        const batchFailure = failedBatchMessage(processed)
        if (batchFailure) failures.push(batchFailure)
      } catch (cause) {
        failures.push(errorMessage(cause, 'Document processing failed.'))
      }
    }

    await Promise.allSettled([loadAdminData(), loadCommonData()])
    if (failures.length) error.value = failures.join(' | ')
  } finally {
    operationBusy.value = false
  }
}

async function reprocess(documentId: string): Promise<void> {
  operationBusy.value = true
  error.value = ''
  try {
    const processed = await queueDocumentBatch([documentId])
    await Promise.allSettled([loadAdminData(), loadCommonData()])
    const batchFailure = failedBatchMessage(processed)
    if (batchFailure) error.value = batchFailure
  } catch (cause) {
    await Promise.allSettled([loadAdminData(), loadCommonData()])
    showError(cause, 'Document processing failed.')
  } finally {
    operationBusy.value = false
  }
}

async function reprocessBatch(documentIds: string[]): Promise<void> {
  const uniqueIds = [...new Set(documentIds)]
  if (!uniqueIds.length) return
  if (!window.confirm(`Reprocess ${uniqueIds.length} document(s)?`)) return

  operationBusy.value = true
  error.value = ''
  try {
    const processed = await queueDocumentBatch(uniqueIds)
    await Promise.allSettled([loadAdminData(), loadCommonData()])
    const batchFailure = failedBatchMessage(processed)
    if (batchFailure) error.value = batchFailure
  } catch (cause) {
    await Promise.allSettled([loadAdminData(), loadCommonData()])
    showError(cause, 'Document batch processing failed.')
  } finally {
    operationBusy.value = false
  }
}

async function removeDocument(documentId: string): Promise<void> {
  if (!window.confirm('Delete this PDF and all of its stored chunks?')) return
  operationBusy.value = true
  error.value = ''
  try {
    await deleteDocument(documentId)
    await Promise.all([loadAdminData(), loadCommonData()])
  } catch (cause) {
    showError(cause, 'Could not delete the document.')
  } finally {
    operationBusy.value = false
  }
}

async function addUser(
  email: string,
  password: string,
  role: 'admin' | 'user',
): Promise<void> {
  operationBusy.value = true
  error.value = ''
  try {
    await createUser(email, password, role)
    users.value = await listUsers()
  } catch (cause) {
    showError(cause, 'Could not create the user.')
  } finally {
    operationBusy.value = false
  }
}

async function toggleUser(userId: string, active: boolean): Promise<void> {
  operationBusy.value = true
  error.value = ''
  try {
    await setUserActive(userId, active)
    users.value = await listUsers()
  } catch (cause) {
    showError(cause, 'Could not update the user.')
  } finally {
    operationBusy.value = false
  }
}

async function refreshAdmin(): Promise<void> {
  operationBusy.value = true
  error.value = ''
  try {
    await Promise.all([loadAdminData(), loadCommonData()])
  } catch (cause) {
    showError(cause, 'Could not refresh admin data.')
  } finally {
    operationBusy.value = false
  }
}

async function refreshSessions(): Promise<void> {
  operationBusy.value = true
  error.value = ''
  try {
    sessions.value = await listAuthSessions()
  } catch (cause) {
    showError(cause, 'Could not refresh sessions.')
  } finally {
    operationBusy.value = false
  }
}

async function revokeSession(targetSessionId: string): Promise<void> {
  if (targetSessionId === getCurrentSessionId()) {
    await signOut()
    return
  }
  operationBusy.value = true
  error.value = ''
  try {
    await revokeAuthSession(targetSessionId)
    sessions.value = await listAuthSessions()
  } catch (cause) {
    showError(cause, 'Could not revoke the session.')
  } finally {
    operationBusy.value = false
  }
}

onMounted(() => {
  void bootstrap()
})
</script>

<template>
  <div v-if="initializing" class="startup-screen">
    <span class="spinner startup-spinner" />
    <p>Restoring your secure session…</p>
  </div>

  <LoginPanel v-else-if="!user" :busy="authBusy" @login="signIn" />

  <div v-else class="app-layout">
    <UploadPanel
      :user="user"
      :view="view"
      :knowledge="knowledge"
      :chats="chats"
      :active-chat-id="activeChatId"
      @navigate="navigate"
      @new-chat="newChat"
      @open-chat="openChat"
      @delete-chat="removeChat"
      @logout="signOut"
    />

    <ChatPanel
      v-if="view === 'chat'"
      v-model:question="question"
      v-model:messages="messages"
      :knowledge="knowledge"
      :busy="answering"
      :active-chat-title="activeChatTitle"
      @ask="ask"
      @cancel="cancel"
    />

    <AdminPanel
      v-else-if="view === 'admin' && user.role === 'admin'"
      :documents="documents"
      :users="users"
      :knowledge="knowledge"
      :busy="operationBusy"
      :current-user-id="user.id"
      @upload="uploadFiles"
      @process="reprocess"
      @process-batch="reprocessBatch"
      @delete-document="removeDocument"
      @create-user="addUser"
      @set-user-active="toggleUser"
      @refresh="refreshAdmin"
    />

    <AccountPanel
      v-else
      :user="user"
      :sessions="sessions"
      :current-session-id="getCurrentSessionId()"
      :busy="operationBusy"
      @refresh="refreshSessions"
      @revoke="revokeSession"
      @logout="signOut"
    />

    <div v-if="error" class="error-toast" role="alert">
      <span>!</span>
      <p>{{ error }}</p>
      <button aria-label="Dismiss error" @click="error = ''">×</button>
    </div>
  </div>
</template>
