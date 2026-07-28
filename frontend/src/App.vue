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
  processDocument,
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


function id(): string {
  return crypto.randomUUID()
}

function messageResponse(
  content: string,
  metadata: Record<string, unknown>,
): AnswerResponse {
  const sources = Array.isArray(metadata.sources)
    ? (metadata.sources as SourceResult[])
    : []
  return {
    answer: content,
    sources,
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
  messages.value.push({ id: id(), role: 'user', text: value })
  question.value = ''
  answering.value = true
  error.value = ''
  controller = new AbortController()
  try {
    const response = await askQuestion(value, activeChatId.value, controller.signal)
    activeChatId.value = response.chat_session_id ?? activeChatId.value
    messages.value.push({
      id: id(),
      role: 'assistant',
      text: response.answer,
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
  try {
    for (const file of files) await uploadDocument(file, true)
    await Promise.all([loadAdminData(), loadCommonData()])
  } catch (cause) {
    await Promise.allSettled([loadAdminData(), loadCommonData()])
    showError(cause, 'Document upload failed.')
  } finally {
    operationBusy.value = false
  }
}

async function reprocess(documentId: string): Promise<void> {
  operationBusy.value = true
  error.value = ''
  try {
    await processDocument(documentId)
    await Promise.all([loadAdminData(), loadCommonData()])
  } catch (cause) {
    showError(cause, 'Document processing failed.')
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
