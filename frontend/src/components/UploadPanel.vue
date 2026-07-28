<script setup lang="ts">
import type { ChatSession, KnowledgeStatus, User } from '../services/api'

type ViewName = 'chat' | 'admin' | 'account'

defineProps<{
  user: User
  view: ViewName
  knowledge: KnowledgeStatus | null
  chats: ChatSession[]
  activeChatId: string | null
}>()

const emit = defineEmits<{
  navigate: [view: ViewName]
  newChat: []
  openChat: [chatId: string]
  deleteChat: [chatId: string]
  logout: []
}>()

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
  }).format(new Date(value))
}
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 24 24">
          <path d="M7 2h7l5 5v15H7z" />
          <path d="M14 2v6h6M10 13h6M10 17h6" />
        </svg>
      </div>
      <div>
        <strong>DMRC IMS</strong>
        <span>Persistent PDF knowledge</span>
      </div>
    </div>

    <button class="new-chat-button" @click="emit('newChat')">
      <span aria-hidden="true">＋</span>
      New chat
    </button>

    <nav class="sidebar-nav" aria-label="Main navigation">
      <button :class="{ active: view === 'chat' }" @click="emit('navigate', 'chat')">
        <span aria-hidden="true">◌</span>
        Ask knowledge
      </button>
      <button
        v-if="user.role === 'admin'"
        :class="{ active: view === 'admin' }"
        @click="emit('navigate', 'admin')"
      >
        <span aria-hidden="true">▤</span>
        Admin console
      </button>
      <button :class="{ active: view === 'account' }" @click="emit('navigate', 'account')">
        <span aria-hidden="true">◎</span>
        Account & sessions
      </button>
    </nav>

    <section class="knowledge-card">
      <span class="eyebrow">Shared knowledge</span>
      <div class="knowledge-number">
        {{ knowledge?.ready_documents ?? '—' }}
        <small>ready PDFs</small>
      </div>
      <p>{{ knowledge?.total_chunks ?? 0 }} searchable chunks</p>
      <span
        class="knowledge-status"
        :class="{ empty: !knowledge?.ready_documents }"
      >
        <i />
        {{ knowledge?.ready_documents ? 'Ready for Q&A' : 'Waiting for admin documents' }}
      </span>
    </section>

    <section class="chat-history">
      <div class="section-heading">
        <span class="eyebrow">Recent chats</span>
        <span>{{ chats.length }}</span>
      </div>
      <p v-if="!chats.length" class="sidebar-empty">No saved chats yet.</p>
      <div v-else class="chat-list">
        <div
          v-for="chat in chats"
          :key="chat.id"
          class="chat-history-row"
          :class="{ active: activeChatId === chat.id && view === 'chat' }"
        >
          <button class="chat-open" @click="emit('openChat', chat.id)">
            <strong :title="chat.title">{{ chat.title }}</strong>
            <span>{{ formatDate(chat.updated_at) }}</span>
          </button>
          <button
            class="chat-delete"
            :aria-label="`Delete ${chat.title}`"
            @click="emit('deleteChat', chat.id)"
          >
            ×
          </button>
        </div>
      </div>
    </section>

    <div class="sidebar-user">
      <div class="user-avatar">{{ user.email.charAt(0).toUpperCase() }}</div>
      <div class="user-copy">
        <strong :title="user.email">{{ user.email }}</strong>
        <span>{{ user.role }}</span>
      </div>
      <button aria-label="Sign out" title="Sign out" @click="emit('logout')">↗</button>
    </div>
  </aside>
</template>
