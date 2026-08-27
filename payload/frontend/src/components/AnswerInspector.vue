<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import {
  getHealth,
  type AnswerResponse,
  type HealthResponse,
  type KnowledgeStatus,
  type SourceResult,
} from '../services/api'

const props = defineProps<{
  response: AnswerResponse | null
  knowledge: KnowledgeStatus | null
}>()

const tab = ref<'sources' | 'plan' | 'details'>('sources')
const health = ref<HealthResponse | null>(null)
const healthError = ref('')

const citedSources = computed(() => props.response?.sources ?? [])
const reviewedEvidence = computed(() => props.response?.evidence ?? [])
const contributingDocuments = computed(() => props.response?.contributing_documents ?? [])
const relevantDocuments = computed(() => props.response?.relevant_documents ?? props.response?.primary_documents ?? [])
const conflicts = computed(() => props.response?.conflicts ?? [])
const citedDocumentCount = computed(() => new Set(citedSources.value.map((item) => item.filename)).size)

const contributionLabel = (source: SourceResult): string => {
  const match = String(source.retrieval_method || '').match(/v5\.2-synthesis:([a-z_]+)/i)
  return match?.[1]?.replaceAll('_', ' ') || ''
}

function scoreLabel(score: number): string {
  if (!Number.isFinite(score)) return '—'
  return score.toFixed(2)
}

function groundingLabel(value: string | undefined): string {
  if (!value) return 'Unknown'
  return value.replaceAll('_', ' ')
}

async function loadHealth(): Promise<void> {
  healthError.value = ''
  try {
    health.value = await getHealth()
  } catch (cause) {
    healthError.value = cause instanceof Error ? cause.message : 'Health status unavailable.'
  }
}

watch(
  () => props.response,
  () => {
    tab.value = 'sources'
  },
)

onMounted(() => {
  void loadHealth()
})
</script>

<template>
  <aside class="answer-inspector" aria-label="Answer details">
    <header class="inspector-header">
      <div>
        <span class="eyebrow">Answer inspector</span>
        <strong>{{ response ? 'Grounded response details' : 'Waiting for an answer' }}</strong>
      </div>
      <span
        v-if="response"
        class="inspector-grounding"
        :class="{ verified: response.grounded }"
      >
        {{ response.grounded ? 'Verified' : groundingLabel(response.grounding_status) }}
      </span>
    </header>

    <div class="inspector-tabs" role="tablist">
      <button :class="{ active: tab === 'sources' }" @click="tab = 'sources'">Sources</button>
      <button :class="{ active: tab === 'plan' }" @click="tab = 'plan'">Query Plan</button>
      <button :class="{ active: tab === 'details' }" @click="tab = 'details'">Details</button>
    </div>

    <div class="inspector-scroll">
      <div v-if="!response" class="inspector-empty">
        <strong>Ask the knowledge base</strong>
        <p>
          The latest answer's cited sources, retrieval plan and grounding diagnostics will appear here.
        </p>
      </div>

      <template v-else-if="tab === 'sources'">
        <section class="inspector-card">
          <div class="inspector-section-title">
            <div>
              <span class="eyebrow">Cited evidence</span>
              <h3>Top sources</h3>
            </div>
            <span>{{ citedSources.length }}</span>
          </div>

          <div v-if="citedSources.length" class="inspector-source-list">
            <article v-for="source in citedSources" :key="source.id" class="inspector-source-row">
              <div class="source-file-icon">PDF</div>
              <div class="source-main">
                <strong :title="source.filename">{{ source.filename }}</strong>
                <span>
                  p. {{ source.pages || source.page }}
                  <template v-if="source.section"> · {{ source.section }}</template>
                </span>
                <small v-if="contributionLabel(source)">{{ contributionLabel(source) }}</small>
              </div>
              <div class="source-score">
                <strong>{{ scoreLabel(source.score) }}</strong>
                <span>relevance</span>
              </div>
            </article>
          </div>
          <p v-else class="inspector-muted">No cited evidence is attached to this answer.</p>
        </section>

        <section class="inspector-card compact-card">
          <div class="inspector-metric-row">
            <span>Reviewed evidence</span>
            <strong>{{ reviewedEvidence.length }}</strong>
          </div>
          <div class="inspector-metric-row">
            <span>Contributing documents</span>
            <strong>{{ contributingDocuments.length || citedDocumentCount }}</strong>
          </div>
          <div class="inspector-metric-row">
            <span>Evidence coverage</span>
            <strong>{{ response.evidence_coverage_status?.replaceAll('_', ' ') || '—' }}</strong>
          </div>
        </section>
      </template>

      <template v-else-if="tab === 'plan'">
        <section class="inspector-card">
          <span class="eyebrow">Understanding</span>
          <h3>Query strategy</h3>
          <dl class="query-detail-list">
            <div>
              <dt>Answer strategy</dt>
              <dd>{{ response.answer_strategy?.replaceAll('_', ' ') || 'direct lookup' }}</dd>
            </div>
            <div>
              <dt>Search scope</dt>
              <dd>{{ response.search_scope?.replaceAll('_', ' ') || 'focused' }}</dd>
            </div>
            <div v-if="response.interpreted_question">
              <dt>Interpreted request</dt>
              <dd>{{ response.interpreted_question }}</dd>
            </div>
          </dl>
        </section>

        <section v-if="response.synthesis_dimensions?.length" class="inspector-card">
          <span class="eyebrow">Coverage targets</span>
          <h3>Synthesis dimensions</h3>
          <ul class="inspector-chip-list">
            <li v-for="item in response.synthesis_dimensions" :key="item">{{ item }}</li>
          </ul>
        </section>

        <section v-if="response.search_queries?.length" class="inspector-card">
          <div class="inspector-section-title">
            <div>
              <span class="eyebrow">Retrieval</span>
              <h3>Search formulations</h3>
            </div>
            <span>{{ response.search_queries.length }}</span>
          </div>
          <ol class="query-list">
            <li v-for="query in response.search_queries" :key="query">{{ query }}</li>
          </ol>
        </section>

        <section v-if="relevantDocuments.length" class="inspector-card">
          <div class="inspector-section-title">
            <div>
              <span class="eyebrow">Corpus routing</span>
              <h3>Relevant documents</h3>
            </div>
            <span>{{ relevantDocuments.length }}</span>
          </div>
          <ul class="document-name-list">
            <li v-for="name in relevantDocuments" :key="name">{{ name }}</li>
          </ul>
        </section>
      </template>

      <template v-else>
        <section class="inspector-card">
          <span class="eyebrow">Execution</span>
          <h3>Retrieval details</h3>
          <dl class="query-detail-list metrics-list">
            <div>
              <dt>Corpus available</dt>
              <dd>{{ knowledge?.ready_documents ?? '—' }} PDFs</dd>
            </div>
            <div>
              <dt>Candidate chunks</dt>
              <dd>{{ response.candidate_chunks ?? 0 }}</dd>
            </div>
            <div>
              <dt>Final evidence</dt>
              <dd>{{ response.evidence_chunks ?? reviewedEvidence.length }}</dd>
            </div>
            <div>
              <dt>Search rounds</dt>
              <dd>{{ response.search_rounds ?? 1 }}</dd>
            </div>
            <div>
              <dt>Grounding</dt>
              <dd>{{ groundingLabel(response.grounding_status) }}</dd>
            </div>
            <div>
              <dt>Policy</dt>
              <dd>{{ response.answer_policy_version || '—' }}</dd>
            </div>
          </dl>
        </section>

        <section v-if="conflicts.length" class="inspector-card conflict-card">
          <div class="inspector-section-title">
            <div>
              <span class="eyebrow">Cross-document review</span>
              <h3>Differences / conflicts</h3>
            </div>
            <span>{{ conflicts.length }}</span>
          </div>
          <article v-for="(conflict, index) in conflicts" :key="`${conflict.type}-${index}`" class="conflict-row">
            <strong>{{ conflict.type.replaceAll('_', ' ') }}</strong>
            <p v-if="conflict.summary">{{ conflict.summary }}</p>
            <small v-if="conflict.resolution">{{ conflict.resolution }}</small>
          </article>
        </section>

        <section class="inspector-card status-card">
          <div class="inspector-section-title">
            <div>
              <span class="eyebrow">AI status</span>
              <h3>System readiness</h3>
            </div>
            <button type="button" @click="loadHealth">Refresh</button>
          </div>
          <p v-if="healthError" class="status-error">{{ healthError }}</p>
          <template v-else-if="health">
            <div class="status-line"><i :class="{ ok: health.status === 'ok' }" /><span>API</span><strong>{{ health.status }}</strong></div>
            <div class="status-line"><i :class="{ ok: health.embedding_ready }" /><span>Embedding index</span><strong>{{ health.embedding_ready ? 'Ready' : 'Unavailable' }}</strong></div>
            <div class="status-line"><i :class="{ ok: health.ocr_available }" /><span>OCR</span><strong>{{ health.ocr_available ? 'Ready' : 'Unavailable' }}</strong></div>
            <div class="status-line"><i :class="{ ok: health.table_extraction_available }" /><span>Table extraction</span><strong>{{ health.table_extraction_available ? 'Ready' : 'Unavailable' }}</strong></div>
          </template>
          <p v-else class="inspector-muted">Checking system health…</p>
        </section>
      </template>
    </div>
  </aside>
</template>

<style scoped>
.answer-inspector {
  min-width: 0;
  height: 100vh;
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
  overflow: hidden;
  border-left: 1px solid #dde6e2;
  background: rgba(251, 253, 252, .96);
}

.inspector-header {
  min-height: 88px;
  padding: 20px 18px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid #e4ebe8;
}

.inspector-header strong {
  display: block;
  margin-top: 4px;
  color: #263a32;
  font-size: 12px;
}

.inspector-grounding {
  padding: 5px 8px;
  border-radius: 999px;
  background: #f1f4f3;
  color: #6d7b76;
  font-size: 9px;
  font-weight: 800;
  text-transform: capitalize;
}

.inspector-grounding.verified {
  background: #e5f6ee;
  color: #237055;
}

.inspector-tabs {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  padding: 0 12px;
  border-bottom: 1px solid #e4ebe8;
}

.inspector-tabs button {
  min-height: 45px;
  border: 0;
  border-bottom: 2px solid transparent;
  background: transparent;
  color: #74817c;
  font-size: 10px;
  font-weight: 800;
}

.inspector-tabs button.active {
  border-bottom-color: #236e58;
  color: #1f5948;
}

.inspector-scroll {
  min-height: 0;
  overflow-y: auto;
  padding: 14px 12px 26px;
}

.inspector-empty {
  margin: 18px 3px;
  padding: 22px;
  border: 1px dashed #d9e3df;
  border-radius: 14px;
  color: #71807a;
  text-align: center;
}

.inspector-empty strong { color: #344a41; font-size: 12px; }
.inspector-empty p { margin: 6px 0 0; font-size: 10px; line-height: 1.55; }

.inspector-card {
  margin-bottom: 11px;
  padding: 13px;
  border: 1px solid #dde6e2;
  border-radius: 13px;
  background: #fff;
  box-shadow: 0 6px 20px rgba(36, 67, 56, .035);
}

.inspector-card h3 {
  margin: 3px 0 10px;
  color: #2b4037;
  font-size: 12px;
}

.inspector-section-title {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: 10px;
}

.inspector-section-title > span {
  min-width: 24px;
  padding: 3px 6px;
  border-radius: 999px;
  background: #f0f4f2;
  color: #687871;
  font-size: 8px;
  font-weight: 800;
  text-align: center;
}

.inspector-source-list { display: grid; gap: 6px; }

.inspector-source-row {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) auto;
  align-items: start;
  gap: 8px;
  padding: 8px 0;
  border-bottom: 1px solid #edf1ef;
}
.inspector-source-row:last-child { border-bottom: 0; }

.source-file-icon {
  width: 27px;
  height: 27px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: #f8eeee;
  color: #a14d4d;
  font-size: 7px;
  font-weight: 900;
}

.source-main { min-width: 0; }
.source-main strong,
.source-main span,
.source-main small { display: block; }
.source-main strong {
  overflow: hidden;
  color: #334940;
  font-size: 9.5px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.source-main span { margin-top: 2px; color: #84908b; font-size: 8px; line-height: 1.35; }
.source-main small { margin-top: 3px; color: #39735e; font-size: 7.5px; text-transform: capitalize; }

.source-score { text-align: right; }
.source-score strong,
.source-score span { display: block; }
.source-score strong { color: #287056; font-size: 9px; }
.source-score span { margin-top: 1px; color: #99a39f; font-size: 7px; }

.compact-card { padding-top: 9px; padding-bottom: 9px; }
.inspector-metric-row,
.status-line {
  min-height: 28px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  border-bottom: 1px solid #edf1ef;
  color: #64746d;
  font-size: 8.5px;
}
.inspector-metric-row:last-child,
.status-line:last-child { border-bottom: 0; }
.inspector-metric-row strong { color: #354a41; text-transform: capitalize; }

.query-detail-list { margin: 0; display: grid; gap: 8px; }
.query-detail-list > div {
  display: grid;
  grid-template-columns: 105px minmax(0, 1fr);
  gap: 9px;
  align-items: start;
}
.query-detail-list dt { color: #88958f; font-size: 8px; }
.query-detail-list dd { margin: 0; color: #344a41; font-size: 8.5px; line-height: 1.45; text-transform: none; }
.metrics-list dd { text-align: right; text-transform: capitalize; }

.inspector-chip-list,
.document-name-list,
.query-list { margin: 0; padding: 0; list-style: none; }
.inspector-chip-list { display: flex; flex-wrap: wrap; gap: 5px; }
.inspector-chip-list li {
  padding: 4px 6px;
  border-radius: 8px;
  background: #eef6f2;
  color: #456b5d;
  font-size: 7.5px;
}

.query-list { counter-reset: query; display: grid; gap: 7px; }
.query-list li {
  position: relative;
  padding-left: 18px;
  color: #53655e;
  font-size: 8.5px;
  line-height: 1.45;
}
.query-list li::before {
  counter-increment: query;
  content: counter(query);
  position: absolute;
  left: 0;
  top: 0;
  width: 13px;
  height: 13px;
  display: grid;
  place-items: center;
  border-radius: 5px;
  background: #edf3f0;
  color: #658077;
  font-size: 7px;
  font-weight: 800;
}

.document-name-list { display: grid; gap: 5px; }
.document-name-list li {
  padding: 6px 7px;
  border-radius: 8px;
  background: #f7faf8;
  color: #53665e;
  font-size: 8px;
  overflow-wrap: anywhere;
}

.conflict-card { border-color: #eadfc8; background: #fffdf8; }
.conflict-row { padding: 7px 0; border-bottom: 1px solid #f0e7d5; }
.conflict-row:last-child { border-bottom: 0; }
.conflict-row strong { color: #805d25; font-size: 8px; text-transform: capitalize; }
.conflict-row p { margin: 3px 0 0; color: #6f624b; font-size: 8px; line-height: 1.45; }
.conflict-row small { display: block; margin-top: 3px; color: #8b7a5e; font-size: 7.5px; line-height: 1.4; }

.status-card .inspector-section-title button {
  border: 0;
  background: transparent;
  color: #477362;
  font-size: 8px;
  font-weight: 800;
}
.status-line { grid-template-columns: 10px minmax(0, 1fr) auto; }
.status-line i { width: 6px; height: 6px; border-radius: 50%; background: #c39b52; }
.status-line i.ok { background: #42a579; box-shadow: 0 0 0 3px rgba(66,165,121,.1); }
.status-line strong { color: #43574f; font-size: 8px; text-transform: capitalize; }
.status-error { margin: 0; color: #9a4b4b; font-size: 8px; line-height: 1.45; }
.inspector-muted { margin: 0; color: #85918c; font-size: 8px; line-height: 1.5; }
</style>
