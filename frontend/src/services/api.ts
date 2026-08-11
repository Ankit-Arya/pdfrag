export interface User {
  id: string
  email: string
  role: 'admin' | 'user'
}

export interface TokenPair {
  access_token: string
  refresh_token: string
  token_type: string
}

export interface AuthSession {
  id: string
  created_at: string
  expires_at: string
  revoked_at: string | null
  user_agent: string | null
  ip_address: string | null
}

export interface DocumentRecord {
  id: string
  filename: string
  status: 'uploaded' | 'processing' | 'ready' | 'failed' | string
  size_bytes: number
  page_count: number
  chunk_count: number
  warnings: unknown[]
  error: string | null
  created_at: string
}

export interface DocumentBatchResponse {
  queued_document_ids: string[]
  already_processing: number
  missing: number
}

export interface KnowledgeStatus {
  ready_documents: number
  total_chunks: number
}

export interface AdminUser {
  id: string
  email: string
  role: 'admin' | 'user'
  is_active: boolean
  created_at: string
}

export interface SourceResult {
  id: string
  filename: string
  page: number
  score: number
  excerpt: string
  content_type: string
  retrieval_method: string
}

export interface AnswerResponse {
  answer: string
  /** Only evidence actually cited by the final answer. */
  sources: SourceResult[]
  /** Every evidence chunk reviewed by the answer/summarization pipeline. */
  evidence: SourceResult[]
  grounded: boolean
  grounding_status:
    | 'verified'
    | 'verified_after_repair'
    | 'insufficient_evidence'
    | 'citation_validation_failed'
    | string
  interpreted_question: string | null
  contextual_question?: string | null
  retrieval_mode?: 'answer' | 'references' | string
  resolved_abbreviations?: string[]
  candidate_chunks?: number
  evidence_chunks?: number
  search_queries: string[]
  request_id?: string | null
  chat_session_id?: string | null
  question_created_at?: string | null
  response_created_at?: string | null
}

export interface ChatSession {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface StoredChatMessage {
  id: string
  role: 'user' | 'assistant' | string
  content: string
  metadata: Record<string, unknown>
  created_at: string
}

export interface ChatDetail extends ChatSession {
  messages: StoredChatMessage[]
}

interface ErrorPayload {
  detail?: string | Array<{ msg?: string }>
}

const ACCESS_KEY = 'pdfrag.access-token'
const REFRESH_KEY = 'pdfrag.refresh-token'
let refreshPromise: Promise<boolean> | null = null

export function hasStoredSession(): boolean {
  return Boolean(localStorage.getItem(REFRESH_KEY))
}

export function clearStoredSession(): void {
  sessionStorage.removeItem(ACCESS_KEY)
  localStorage.removeItem(REFRESH_KEY)
}

export function currentSessionId(): string | null {
  const token = sessionStorage.getItem(ACCESS_KEY)
  if (!token) return null
  try {
    const payload = JSON.parse(decodeBase64Url(token.split('.')[1])) as { sid?: string }
    return payload.sid ?? null
  } catch {
    return null
  }
}

function storeTokens(tokens: TokenPair): void {
  sessionStorage.setItem(ACCESS_KEY, tokens.access_token)
  localStorage.setItem(REFRESH_KEY, tokens.refresh_token)
}

function decodeBase64Url(value: string): string {
  const normalized = value.replace(/-/g, '+').replace(/_/g, '/')
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '=')
  return decodeURIComponent(
    Array.from(atob(padded))
      .map((character) => `%${character.charCodeAt(0).toString(16).padStart(2, '0')}`)
      .join(''),
  )
}

function errorDetail(payload: ErrorPayload, statusCode: number): string {
  if (typeof payload.detail === 'string') return payload.detail
  if (Array.isArray(payload.detail)) {
    const messages = payload.detail
      .map((item) => item.msg)
      .filter((message): message is string => Boolean(message))
    if (messages.length) return messages.join('; ')
  }
  return `Request failed with status ${statusCode}`
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`
    try {
      detail = errorDetail((await response.json()) as ErrorPayload, response.status)
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(detail)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

async function refreshAccessToken(): Promise<boolean> {
  if (refreshPromise) return refreshPromise
  refreshPromise = (async () => {
    const refreshToken = localStorage.getItem(REFRESH_KEY)
    if (!refreshToken) return false

    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })
      if (!response.ok) {
        clearStoredSession()
        return false
      }
      storeTokens((await response.json()) as TokenPair)
      return true
    } catch {
      return false
    }
  })()

  try {
    return await refreshPromise
  } finally {
    refreshPromise = null
  }
}

async function authenticatedFetch(
  path: string,
  init: RequestInit = {},
  retryAfterRefresh = true,
): Promise<Response> {
  const headers = new Headers(init.headers)
  const accessToken = sessionStorage.getItem(ACCESS_KEY)
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)

  const response = await fetch(path, { ...init, headers })
  if (response.status === 401 && retryAfterRefresh && hasStoredSession()) {
    const refreshed = await refreshAccessToken()
    if (refreshed) return authenticatedFetch(path, init, false)
  }

  return response
}

async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  return parseResponse<T>(await authenticatedFetch(path, init))
}

export async function login(email: string, password: string): Promise<User> {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  const tokens = await parseResponse<TokenPair>(response)
  storeTokens(tokens)
  return getCurrentUser()
}

export async function getCurrentUser(): Promise<User> {
  return apiRequest<User>('/api/auth/me')
}

export async function logout(): Promise<void> {
  const refreshToken = localStorage.getItem(REFRESH_KEY)
  try {
    if (refreshToken) {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })
    }
  } finally {
    clearStoredSession()
  }
}

export async function listAuthSessions(): Promise<AuthSession[]> {
  return apiRequest<AuthSession[]>('/api/auth/sessions')
}

export async function revokeAuthSession(sessionId: string): Promise<void> {
  return apiRequest<void>(`/api/auth/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
}

export async function getKnowledgeStatus(): Promise<KnowledgeStatus> {
  return apiRequest<KnowledgeStatus>('/api/knowledge/status')
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  return apiRequest<DocumentRecord[]>('/api/documents')
}

export async function downloadDocument(
  documentId: string,
  filename: string,
): Promise<void> {
  const response = await authenticatedFetch(
    `/api/documents/${encodeURIComponent(documentId)}/download`,
  )
  if (!response.ok) {
    await parseResponse<never>(response)
    return
  }

  const blob = await response.blob()
  const objectUrl = URL.createObjectURL(blob)
  const anchor = window.document.createElement('a')
  anchor.href = objectUrl
  anchor.download = filename || 'document.pdf'
  window.document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0)
}

export async function uploadDocument(
  file: File,
  process = true,
): Promise<DocumentRecord> {
  const formData = new FormData()
  formData.append('file', file)
  return apiRequest<DocumentRecord>(
    `/api/admin/documents?process=${process ? 'true' : 'false'}`,
    { method: 'POST', body: formData },
  )
}

export async function processDocument(documentId: string): Promise<DocumentRecord> {
  return apiRequest<DocumentRecord>(
    `/api/admin/documents/${encodeURIComponent(documentId)}/process`,
    { method: 'POST' },
  )
}

export async function processDocuments(
  documentIds: string[],
): Promise<DocumentBatchResponse> {
  return apiRequest<DocumentBatchResponse>('/api/admin/documents/process-batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_ids: documentIds }),
  })
}

export async function deleteDocument(documentId: string): Promise<void> {
  return apiRequest<void>(
    `/api/admin/documents/${encodeURIComponent(documentId)}`,
    { method: 'DELETE' },
  )
}

export async function listUsers(): Promise<AdminUser[]> {
  return apiRequest<AdminUser[]>('/api/admin/users')
}

export async function createUser(
  email: string,
  password: string,
  role: 'admin' | 'user',
): Promise<AdminUser> {
  return apiRequest<AdminUser>('/api/admin/users', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, role }),
  })
}

export async function setUserActive(
  userId: string,
  isActive: boolean,
): Promise<AdminUser> {
  return apiRequest<AdminUser>(`/api/admin/users/${encodeURIComponent(userId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ is_active: isActive }),
  })
}

export async function listChats(): Promise<ChatSession[]> {
  return apiRequest<ChatSession[]>('/api/chats')
}

export async function getChat(chatId: string): Promise<ChatDetail> {
  return apiRequest<ChatDetail>(`/api/chats/${encodeURIComponent(chatId)}`)
}

export async function deleteChat(chatId: string): Promise<void> {
  return apiRequest<void>(`/api/chats/${encodeURIComponent(chatId)}`, {
    method: 'DELETE',
  })
}

export async function askQuestion(
  question: string,
  chatSessionId: string | null,
  signal?: AbortSignal,
): Promise<AnswerResponse> {
  return apiRequest<AnswerResponse>('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      chat_session_id: chatSessionId || null,
    }),
    signal,
  })
}
