<script setup lang="ts">
import { computed, ref } from 'vue'
import type { CollectionResponse } from '../services/api'

defineProps<{
  collection: CollectionResponse | null
  busy: boolean
}>()

const emit = defineEmits<{
  upload: [files: File[]]
  reset: []
}>()

const selectedFiles = ref<File[]>([])
const dragging = ref(false)
const input = ref<HTMLInputElement | null>(null)

const totalSize = computed(() =>
  selectedFiles.value.reduce((sum, file) => sum + file.size, 0),
)

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function acceptFiles(files: FileList | File[]): void {
  const pdfs = Array.from(files).filter(
    (file) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf'),
  )
  selectedFiles.value = pdfs
}

function drop(event: DragEvent): void {
  dragging.value = false
  if (event.dataTransfer?.files) acceptFiles(event.dataTransfer.files)
}

function upload(): void {
  if (selectedFiles.value.length) emit('upload', selectedFiles.value)
}

function reset(): void {
  selectedFiles.value = []
  if (input.value) input.value.value = ''
  emit('reset')
}
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 24 24"><path d="M7 2h7l5 5v15H7z"/><path d="M14 2v6h6M10 13h6M10 17h6"/></svg>
      </div>
      <div>
        <strong>DMRC IMS PROTOTYPE</strong>
        <span>PDF-only answers</span>
      </div>
    </div>

    <section v-if="!collection" class="upload-section">
      <div
        class="drop-zone"
        :class="{ dragging }"
        role="button"
        tabindex="0"
        @click="input?.click()"
        @keydown.enter="input?.click()"
        @keydown.space.prevent="input?.click()"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="drop"
      >
        <input
          ref="input"
          hidden
          multiple
          type="file"
          accept="application/pdf,.pdf"
          @change="acceptFiles(($event.target as HTMLInputElement).files ?? [])"
        />
        <div class="upload-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path d="M12 16V4m0 0L7 9m5-5 5 5M5 14v5h14v-5"/></svg>
        </div>
        <strong>Drop PDFs here</strong>
        <span>or click to browse</span>
      </div>

      <div v-if="selectedFiles.length" class="selected-files">
        <div v-for="file in selectedFiles" :key="`${file.name}-${file.size}`" class="file-row">
          <div class="file-dot">PDF</div>
          <div class="file-copy">
            <strong :title="file.name">{{ file.name }}</strong>
            <span>{{ formatBytes(file.size) }}</span>
          </div>
        </div>
        <div class="selection-summary">
          <span>{{ selectedFiles.length }} file{{ selectedFiles.length === 1 ? '' : 's' }}</span>
          <span>{{ formatBytes(totalSize) }}</span>
        </div>
      </div>

      <button class="primary-button" :disabled="!selectedFiles.length || busy" @click="upload">
        <span v-if="busy" class="spinner small" />
        {{ busy ? 'Indexing documents…' : 'Create knowledge set' }}
      </button>
      <p class="privacy-note">Files are processed in memory and are not stored in a database.</p>
    </section>

    <section v-else class="collection-section">
      <div class="status-pill"><span /> Ready for questions</div>
      <div class="collection-stats">
        <div><strong>{{ collection.files.length }}</strong><span>PDFs</span></div>
        <div><strong>{{ collection.total_pages }}</strong><span>Pages</span></div>
        <div><strong>{{ collection.total_chunks }}</strong><span>Chunks</span></div>
      </div>
      <div class="indexed-files">
        <span class="eyebrow">Indexed files</span>
        <div v-for="file in collection.files" :key="file.name" class="file-row compact">
          <div class="file-dot">PDF</div>
          <div class="file-copy">
            <strong :title="file.name">{{ file.name }}</strong>
            <span>{{ file.pages }} pages · {{ file.chunks }} chunks</span>
          </div>
        </div>
      </div>
      <button class="secondary-button" :disabled="busy" @click="reset">Replace documents</button>
      <p class="privacy-note">This collection expires after {{ collection.expires_in_minutes }} minutes of inactivity.</p>
    </section>

    <div class="sidebar-footer">
      <span class="shield" aria-hidden="true">✓</span>
      <p><strong>Strictly grounded</strong><br />No web search or conversation memory</p>
    </div>
  </aside>
</template>
