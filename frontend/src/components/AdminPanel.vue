<script setup lang="ts">
import { computed, ref } from 'vue'
import type { AdminUser, DocumentRecord, KnowledgeStatus } from '../services/api'

const props = defineProps<{
  documents: DocumentRecord[]
  users: AdminUser[]
  knowledge: KnowledgeStatus | null
  busy: boolean
  currentUserId: string
}>()

const emit = defineEmits<{
  upload: [files: File[]]
  process: [documentId: string]
  deleteDocument: [documentId: string]
  createUser: [email: string, password: string, role: 'admin' | 'user']
  setUserActive: [userId: string, active: boolean]
  refresh: []
}>()

const tab = ref<'documents' | 'users'>('documents')
const selectedFiles = ref<File[]>([])
const fileInput = ref<HTMLInputElement | null>(null)
const folderInput = ref<HTMLInputElement | null>(null)
const dragging = ref(false)
const email = ref('')
const password = ref('')
const role = ref<'admin' | 'user'>('user')

const totalUploadSize = computed(() =>
  selectedFiles.value.reduce((total, file) => total + file.size, 0),
)

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function fileKey(file: File): string {
  const path = file.webkitRelativePath || file.name
  return `${path}::${file.size}::${file.lastModified}`
}

function fileLabel(file: File): string {
  return file.webkitRelativePath || file.name
}

function acceptFiles(files: FileList | File[], source?: HTMLInputElement): void {
  const existing = new Set(selectedFiles.value.map(fileKey))
  const additions: File[] = []

  for (const file of Array.from(files)) {
    const isPdf =
      file.type === 'application/pdf'
      || file.name.toLowerCase().endsWith('.pdf')
    const key = fileKey(file)

    if (!isPdf || existing.has(key)) continue
    existing.add(key)
    additions.push(file)
  }

  if (additions.length) selectedFiles.value = [...selectedFiles.value, ...additions]

  // Reset the picker so selecting the same folder again still fires change.
  if (source) source.value = ''
}

function selectFromInput(event: Event): void {
  const source = event.target as HTMLInputElement
  acceptFiles(source.files ?? [], source)
}

function removeSelectedFile(target: File): void {
  const targetKey = fileKey(target)
  selectedFiles.value = selectedFiles.value.filter(
    (file) => fileKey(file) !== targetKey,
  )
}

function clearSelection(): void {
  selectedFiles.value = []
  if (fileInput.value) fileInput.value.value = ''
  if (folderInput.value) folderInput.value.value = ''
}

function drop(event: DragEvent): void {
  dragging.value = false
  if (event.dataTransfer?.files) acceptFiles(event.dataTransfer.files)
}

function upload(): void {
  if (!selectedFiles.value.length || props.busy) return
  emit('upload', [...selectedFiles.value])
  clearSelection()
}

function submitUser(): void {
  const normalizedEmail = email.value.trim().toLowerCase()
  if (!normalizedEmail || password.value.length < 8 || props.busy) return
  emit('createUser', normalizedEmail, password.value, role.value)
  email.value = ''
  password.value = ''
  role.value = 'user'
}
</script>

<template>
  <main class="content-shell admin-shell">
    <header class="topbar admin-topbar">
      <div>
        <span class="eyebrow">Administrator</span>
        <h1>Knowledge and user management</h1>
        <p>Upload PDFs once, process their chunks, and grant users access to Q&A.</p>
      </div>
      <button class="ghost-action" :disabled="busy" @click="emit('refresh')">Refresh</button>
    </header>

    <section class="admin-content">
      <div class="admin-summary-grid">
        <article>
          <span>Ready documents</span>
          <strong>{{ knowledge?.ready_documents ?? 0 }}</strong>
        </article>
        <article>
          <span>Stored chunks</span>
          <strong>{{ knowledge?.total_chunks ?? 0 }}</strong>
        </article>
        <article>
          <span>User accounts</span>
          <strong>{{ users.length }}</strong>
        </article>
      </div>

      <div class="tab-bar" role="tablist">
        <button :class="{ active: tab === 'documents' }" @click="tab = 'documents'">
          Documents
        </button>
        <button :class="{ active: tab === 'users' }" @click="tab = 'users'">
          Users
        </button>
      </div>

      <template v-if="tab === 'documents'">
        <section class="panel-card upload-card">
          <div>
            <span class="eyebrow">Add knowledge</span>
            <h2>Upload and process PDFs</h2>
            <p>Each PDF is stored in PostgreSQL and processed into persistent chunks.</p>
          </div>
          <div class="multi-source-picker">
            <div
              class="admin-drop-zone"
              :class="{ dragging }"
              role="button"
              tabindex="0"
              @click="fileInput?.click()"
              @keydown.enter="fileInput?.click()"
              @keydown.space.prevent="fileInput?.click()"
              @dragover.prevent="dragging = true"
              @dragleave.prevent="dragging = false"
              @drop.prevent="drop"
            >
              <input
                ref="fileInput"
                hidden
                multiple
                type="file"
                accept="application/pdf,.pdf"
                @change="selectFromInput"
              />
              <input
                ref="folderInput"
                hidden
                multiple
                webkitdirectory
                type="file"
                accept="application/pdf,.pdf"
                @change="selectFromInput"
              />
              <strong>
                {{ selectedFiles.length
                  ? `${selectedFiles.length} PDF(s) queued from one or more folders`
                  : 'Drop PDFs or click to browse' }}
              </strong>
              <span v-if="selectedFiles.length">{{ formatBytes(totalUploadSize) }} total</span>
              <span v-else>Selections accumulate until you upload or clear the queue.</span>
            </div>
            <div class="picker-actions">
              <button type="button" class="ghost-action compact" :disabled="busy" @click="fileInput?.click()">
                Add PDFs
              </button>
              <button type="button" class="ghost-action compact" :disabled="busy" @click="folderInput?.click()">
                Add folder
              </button>
              <button
                v-if="selectedFiles.length"
                type="button"
                class="clear-selection"
                :disabled="busy"
                @click="clearSelection"
              >
                Clear all
              </button>
            </div>
          </div>
          <div v-if="selectedFiles.length" class="selected-chip-list">
            <span
              v-for="file in selectedFiles"
              :key="fileKey(file)"
              class="selected-file-chip"
              :title="fileLabel(file)"
            >
              <span>{{ fileLabel(file) }}</span>
              <button
                type="button"
                :aria-label="`Remove ${fileLabel(file)}`"
                :disabled="busy"
                @click="removeSelectedFile(file)"
              >
                ×
              </button>
            </span>
          </div>
          <button
            class="primary-action fit"
            :disabled="!selectedFiles.length || busy"
            @click="upload"
          >
            <span v-if="busy" class="spinner small" />
            {{ busy ? 'Processing…' : 'Upload and process' }}
          </button>
        </section>

        <section class="panel-card">
          <div class="panel-heading">
            <div>
              <span class="eyebrow">Persistent documents</span>
              <h2>{{ documents.length }} uploaded PDF{{ documents.length === 1 ? '' : 's' }}</h2>
            </div>
          </div>
          <p v-if="!documents.length" class="empty-table">No documents have been uploaded.</p>
          <div v-else class="data-list">
            <article v-for="document in documents" :key="document.id" class="data-row document-row">
              <div class="file-icon">PDF</div>
              <div class="data-main">
                <strong>{{ document.filename }}</strong>
                <span>
                  {{ formatBytes(document.size_bytes) }} · {{ document.page_count }} pages ·
                  {{ document.chunk_count }} chunks
                </span>
                <small v-if="document.error" class="row-error">{{ document.error }}</small>
              </div>
              <span class="status-tag" :class="document.status">{{ document.status }}</span>
              <div class="row-actions">
                <button
                  v-if="document.status !== 'ready' || document.error"
                  :disabled="busy || document.status === 'processing'"
                  @click="emit('process', document.id)"
                >
                  Process
                </button>
                <button class="danger-link" :disabled="busy" @click="emit('deleteDocument', document.id)">
                  Delete
                </button>
              </div>
            </article>
          </div>
        </section>
      </template>

      <template v-else>
        <section class="panel-card user-create-card">
          <div>
            <span class="eyebrow">New account</span>
            <h2>Create a user</h2>
            <p>Users sign in with JWT-backed sessions and can immediately ask the shared knowledge base.</p>
          </div>
          <form class="inline-user-form" @submit.prevent="submitUser">
            <label>
              <span>Email</span>
              <input v-model="email" type="email" required placeholder="user@example.com" />
            </label>
            <label>
              <span>Temporary password</span>
              <input v-model="password" type="password" minlength="8" required placeholder="8+ characters" />
            </label>
            <label>
              <span>Role</span>
              <select v-model="role">
                <option value="user">User</option>
                <option value="admin">Admin</option>
              </select>
            </label>
            <button class="primary-action" :disabled="busy || !email.trim() || password.length < 8">
              Create account
            </button>
          </form>
        </section>

        <section class="panel-card">
          <div class="panel-heading">
            <div>
              <span class="eyebrow">Access control</span>
              <h2>{{ users.length }} account{{ users.length === 1 ? '' : 's' }}</h2>
            </div>
          </div>
          <div class="data-list">
            <article v-for="account in users" :key="account.id" class="data-row user-row">
              <div class="user-avatar light">{{ account.email.charAt(0).toUpperCase() }}</div>
              <div class="data-main">
                <strong>{{ account.email }}</strong>
                <span>{{ account.role }} · created {{ formatDate(account.created_at) }}</span>
              </div>
              <span class="status-tag" :class="account.is_active ? 'ready' : 'failed'">
                {{ account.is_active ? 'active' : 'disabled' }}
              </span>
              <button
                class="ghost-action compact"
                :disabled="busy || account.id === currentUserId"
                @click="emit('setUserActive', account.id, !account.is_active)"
              >
                {{ account.is_active ? 'Disable' : 'Enable' }}
              </button>
            </article>
          </div>
        </section>
      </template>
    </section>
  </main>
</template>
