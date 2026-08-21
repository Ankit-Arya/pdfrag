<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { ChatProgressEvent } from '../services/api'

const props = withDefaults(defineProps<{
  events: ChatProgressEvent[]
  live?: boolean
}>(), {
  live: false,
})

const now = ref(Date.now())
let timer: number | null = null

onMounted(() => {
  if (props.live) timer = window.setInterval(() => { now.value = Date.now() }, 250)
})

onBeforeUnmount(() => {
  if (timer !== null) window.clearInterval(timer)
})

const compactEvents = computed(() => {
  const order: string[] = []
  const latest = new Map<string, ChatProgressEvent>()
  for (const event of props.events ?? []) {
    const key = event.operation_id || event.stage
    if (!latest.has(key)) order.push(key)
    latest.set(key, event)
  }
  return order.map((key) => latest.get(key)).filter((event): event is ChatProgressEvent => Boolean(event))
})

const firstTimestampMs = computed(() => {
  const value = props.events?.[0]?.timestamp
  return typeof value === 'number' ? value * 1000 : null
})

const totalElapsedMs = computed(() => {
  const reported = Math.max(0, ...((props.events ?? []).map((event) => event.total_elapsed_ms ?? 0)))
  if (!props.live || !firstTimestampMs.value) return reported
  return Math.max(reported, now.value - firstTimestampMs.value)
})

const currentEvent = computed(() => compactEvents.value.at(-1) ?? null)

function actorLabel(actor?: string): string {
  switch ((actor || '').toLowerCase()) {
    case 'ai': return 'AI'
    case 'search': return 'Search'
    case 'verification': return 'Verification'
    default: return 'Backend'
  }
}

function formatDuration(milliseconds?: number): string {
  const value = Math.max(0, milliseconds ?? 0)
  if (value < 1000) return `${value} ms`
  if (value < 60_000) return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)}s`
  const minutes = Math.floor(value / 60_000)
  const seconds = Math.round((value % 60_000) / 1000)
  return `${minutes}m ${seconds}s`
}

function statusClass(event: ChatProgressEvent): string {
  const status = (event.status || '').toLowerCase()
  if (status === 'warning' || status === 'error' || status === 'failed') return 'warning'
  if (status === 'complete' || status === 'completed') return 'complete'
  const key = event.operation_id || event.stage
  const currentKey = currentEvent.value ? (currentEvent.value.operation_id || currentEvent.value.stage) : ''
  if (!props.live || key !== currentKey) return 'complete'
  return 'running'
}

function metricEntries(event: ChatProgressEvent): Array<[string, unknown]> {
  return Object.entries(event.metrics ?? {})
}
</script>

<template>
  <details class="activity-trace" :open="live || undefined">
    <summary class="activity-summary">
      <span class="activity-summary-icon" :class="{ live }" aria-hidden="true" />
      <span>
        <strong>{{ live ? 'Working' : 'Worked' }}</strong>
        <small>
          {{ formatDuration(totalElapsedMs) }}
          <template v-if="compactEvents.length"> · {{ compactEvents.length }} step{{ compactEvents.length === 1 ? '' : 's' }}</template>
        </small>
      </span>
      <span v-if="live && currentEvent" class="activity-current-label">{{ currentEvent.label }}</span>
    </summary>

    <div class="activity-body">
      <ol class="activity-list">
        <li
          v-for="event in compactEvents"
          :key="event.operation_id || `${event.stage}-${event.sequence ?? 0}`"
          class="activity-row"
          :class="statusClass(event)"
        >
          <span class="activity-status-icon" aria-hidden="true">
            <i v-if="statusClass(event) === 'running'" />
            <svg v-else-if="statusClass(event) === 'complete'" viewBox="0 0 16 16"><path d="M3 8.2 6.4 11.3 13 4.7" /></svg>
            <span v-else>!</span>
          </span>

          <div class="activity-copy">
            <div class="activity-title-line">
              <span class="activity-actor" :class="`actor-${event.actor || 'backend'}`">{{ actorLabel(event.actor) }}</span>
              <strong>{{ event.label }}</strong>
              <span v-if="event.duration_ms !== undefined" class="activity-duration">{{ formatDuration(event.duration_ms) }}</span>
            </div>
            <p v-if="event.detail">{{ event.detail }}</p>

            <div v-if="event.document || event.heading" class="activity-match">
              <strong v-if="event.heading">{{ event.heading }}</strong>
              <span>
                <template v-if="event.document">{{ event.document }}</template>
                <template v-if="event.page"> · p. {{ event.page }}</template>
              </span>
            </div>

            <div v-if="event.reasoning_summary" class="activity-insight">
              <span>AI summary</span>
              <p>{{ event.reasoning_summary }}</p>
            </div>

            <div v-if="event.prompt_summary" class="activity-insight prompt-summary">
              <span>AI task</span>
              <p>{{ event.prompt_summary }}</p>
            </div>

            <div v-if="metricEntries(event).length" class="activity-metrics">
              <span v-for="entry in metricEntries(event)" :key="entry[0]">
                <strong>{{ entry[1] }}</strong> {{ entry[0].replace(/_/g, ' ') }}
              </span>
            </div>

            <div
              v-if="typeof event.current === 'number' && typeof event.total === 'number' && event.total > 0"
              class="activity-meter"
              aria-hidden="true"
            >
              <span :style="{ width: `${Math.min(100, Math.max(0, (event.current / event.total) * 100))}%` }" />
            </div>
          </div>
        </li>
      </ol>

      <p class="activity-privacy-note">
        Shows operational activity and concise AI-task summaries, not private chain-of-thought or hidden prompts.
      </p>
    </div>
  </details>
</template>

<style scoped>
.activity-trace {
  margin: 9px 0 12px;
  border: 1px solid #dce5e1;
  border-radius: 13px;
  background: rgba(248, 251, 249, .92);
  overflow: hidden;
}

.activity-summary {
  min-height: 42px;
  padding: 9px 11px;
  display: grid;
  grid-template-columns: 18px auto minmax(0, 1fr);
  align-items: center;
  gap: 8px;
  cursor: pointer;
  list-style: none;
}

.activity-summary::-webkit-details-marker { display: none; }
.activity-summary > span:nth-child(2) { display: flex; align-items: baseline; gap: 6px; min-width: 0; }
.activity-summary strong { color: #35433e; font-size: 10px; }
.activity-summary small { color: #84918c; font-size: 8px; white-space: nowrap; }
.activity-current-label { color: #6d7d77; font-size: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: right; }

.activity-summary-icon {
  width: 12px;
  height: 12px;
  border: 2px solid #81918b;
  border-radius: 50%;
  box-sizing: border-box;
}
.activity-summary-icon.live {
  border-color: #d5e0db;
  border-top-color: #3f765f;
  animation: activity-spin .85s linear infinite;
}

.activity-body { border-top: 1px solid #e6ece9; padding: 9px 11px 8px; }
.activity-list { margin: 0; padding: 0; list-style: none; display: grid; gap: 8px; }
.activity-row { display: grid; grid-template-columns: 18px minmax(0, 1fr); gap: 7px; }
.activity-status-icon { width: 18px; height: 18px; display: grid; place-items: center; color: #6b7b75; }
.activity-status-icon i { width: 9px; height: 9px; border: 2px solid #d6e0dc; border-top-color: #477b68; border-radius: 50%; animation: activity-spin .8s linear infinite; }
.activity-status-icon svg { width: 12px; height: 12px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.activity-row.warning .activity-status-icon { color: #8b6441; font-size: 10px; font-weight: 900; }

.activity-copy { min-width: 0; }
.activity-title-line { display: flex; align-items: center; gap: 6px; min-width: 0; }
.activity-title-line > strong { color: #42514c; font-size: 9px; line-height: 1.4; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.activity-duration { margin-left: auto; color: #93a09b; font-size: 8px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.activity-copy > p { margin: 2px 0 0; color: #7b8883; font-size: 8px; line-height: 1.45; }

.activity-actor { flex: 0 0 auto; padding: 2px 5px; border-radius: 999px; font-size: 7px; font-weight: 900; letter-spacing: .02em; text-transform: uppercase; background: #edf2f0; color: #566761; }
.actor-ai { background: #eef0f8; color: #555f81; }
.actor-search { background: #edf5f0; color: #456d58; }
.actor-verification { background: #f5f1e9; color: #756344; }
.actor-backend { background: #eef2f3; color: #56696f; }

.activity-match { margin-top: 4px; padding: 6px 7px; border-radius: 8px; background: #fff; border: 1px solid #e2e9e6; }
.activity-match strong, .activity-match span { display: block; overflow-wrap: anywhere; }
.activity-match strong { color: #3f4f49; font-size: 8px; }
.activity-match span { margin-top: 2px; color: #7f8d88; font-size: 7px; }

.activity-insight { margin-top: 4px; padding-left: 7px; border-left: 2px solid #d9e3df; }
.activity-insight > span { color: #71817b; font-size: 7px; font-weight: 900; text-transform: uppercase; letter-spacing: .03em; }
.activity-insight p { margin: 2px 0 0; color: #687770; font-size: 8px; line-height: 1.45; }
.prompt-summary { border-left-color: #dfe1ef; }

.activity-metrics { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }
.activity-metrics span { padding: 3px 5px; border-radius: 6px; background: #edf2f0; color: #718079; font-size: 7px; }
.activity-metrics strong { color: #43534d; }
.activity-meter { height: 2px; margin-top: 5px; overflow: hidden; border-radius: 999px; background: #e2e9e6; }
.activity-meter span { display: block; height: 100%; background: #658d7c; border-radius: inherit; transition: width .2s ease; }
.activity-privacy-note { margin: 8px 0 0; padding-top: 7px; border-top: 1px solid #e7ecea; color: #95a09c; font-size: 7px; line-height: 1.45; }

@keyframes activity-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .activity-summary-icon.live, .activity-status-icon i { animation-duration: 1.8s; } }
</style>
