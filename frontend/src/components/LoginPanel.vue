<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  busy: boolean
}>()

const emit = defineEmits<{
  login: [email: string, password: string]
}>()

const email = ref('')
const password = ref('')

function submit(): void {
  const normalizedEmail = email.value.trim().toLowerCase()
  if (!normalizedEmail || password.value.length < 8 || props.busy) return
  emit('login', normalizedEmail, password.value)
}
</script>

<template>
  <main class="auth-page">
    <section class="auth-card">
      <div class="auth-brand">
        <div class="brand-mark large" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path d="M7 2h7l5 5v15H7z" />
            <path d="M14 2v6h6M10 13h6M10 17h6" />
          </svg>
        </div>
        <div>
          <span class="eyebrow">Shared document intelligence</span>
          <h1>DMRC IMS</h1>
        </div>
      </div>

      <div class="auth-copy">
        <h2>Sign in to ask the knowledge base</h2>
        <p>
          Administrators manage the shared PDFs. Signed-in users can ask questions
          against the documents that are already processed and ready.
        </p>
      </div>

      <form class="auth-form" @submit.prevent="submit">
        <label>
          <span>Email</span>
          <input
            v-model="email"
            type="email"
            autocomplete="username"
            placeholder="you@example.com"
            required
          />
        </label>
        <label>
          <span>Password</span>
          <input
            v-model="password"
            type="password"
            autocomplete="current-password"
            placeholder="At least 8 characters"
            minlength="8"
            required
          />
        </label>
        <button
          class="primary-action"
          type="submit"
          :disabled="busy || !email.trim() || password.length < 8"
        >
          <span v-if="busy" class="spinner small" />
          {{ busy ? 'Signing in…' : 'Sign in' }}
        </button>
      </form>

      <p class="auth-note">
        The initial administrator account comes from
        <code>BOOTSTRAP_ADMIN_EMAIL</code> and
        <code>BOOTSTRAP_ADMIN_PASSWORD</code> in your environment.
      </p>
    </section>
  </main>
</template>
