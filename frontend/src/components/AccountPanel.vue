<script setup lang="ts">
import type { AuthSession, User } from '../services/api'

defineProps<{
  user: User
  sessions: AuthSession[]
  currentSessionId: string | null
  busy: boolean
}>()

const emit = defineEmits<{
  refresh: []
  revoke: [sessionId: string]
  logout: []
}>()

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function deviceLabel(userAgent: string | null): string {
  if (!userAgent) return 'Unknown device'
  if (/edg/i.test(userAgent)) return 'Microsoft Edge'
  if (/chrome/i.test(userAgent)) return 'Google Chrome'
  if (/firefox/i.test(userAgent)) return 'Mozilla Firefox'
  if (/safari/i.test(userAgent)) return 'Safari'
  return userAgent.slice(0, 72)
}
</script>

<template>
  <main class="content-shell account-shell">
    <header class="topbar admin-topbar">
      <div>
        <span class="eyebrow">Security</span>
        <h1>Account and active sessions</h1>
        <p>JWT access tokens are short-lived; refresh sessions can be revoked here.</p>
      </div>
      <button class="ghost-action" :disabled="busy" @click="emit('refresh')">Refresh</button>
    </header>

    <section class="admin-content account-content">
      <section class="profile-card panel-card">
        <div class="profile-avatar">{{ user.email.charAt(0).toUpperCase() }}</div>
        <div>
          <span class="eyebrow">Signed in as</span>
          <h2>{{ user.email }}</h2>
          <p>Role: <strong>{{ user.role }}</strong></p>
        </div>
        <button class="danger-action" @click="emit('logout')">Sign out</button>
      </section>

      <section class="panel-card">
        <div class="panel-heading">
          <div>
            <span class="eyebrow">Refresh-token sessions</span>
            <h2>{{ sessions.length }} session{{ sessions.length === 1 ? '' : 's' }}</h2>
          </div>
        </div>
        <p v-if="!sessions.length" class="empty-table">No sessions were returned.</p>
        <div v-else class="data-list">
          <article v-for="session in sessions" :key="session.id" class="data-row session-row">
            <div class="session-icon" aria-hidden="true">◉</div>
            <div class="data-main">
              <strong>
                {{ deviceLabel(session.user_agent) }}
                <span v-if="session.id === currentSessionId" class="current-label">Current</span>
              </strong>
              <span>
                Created {{ formatDate(session.created_at) }} · expires {{ formatDate(session.expires_at) }}
              </span>
              <small>{{ session.ip_address || 'IP unavailable' }}</small>
            </div>
            <span class="status-tag" :class="session.revoked_at ? 'failed' : 'ready'">
              {{ session.revoked_at ? 'revoked' : 'active' }}
            </span>
            <button
              v-if="!session.revoked_at"
              class="ghost-action compact"
              :disabled="busy"
              @click="emit('revoke', session.id)"
            >
              {{ session.id === currentSessionId ? 'Sign out here' : 'Revoke' }}
            </button>
          </article>
        </div>
      </section>
    </section>
  </main>
</template>
