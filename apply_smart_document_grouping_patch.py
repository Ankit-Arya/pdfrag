from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MARKER = "SMART_DOCUMENT_GROUPING_PATCH_V1"

HELPER = r'''// SMART_DOCUMENT_GROUPING_PATCH_V1
export interface GroupableDocument {
  filename: string
  status?: string
}

export interface SequenceSummary {
  observed: number[]
  missing: number[]
  min: number | null
  max: number | null
  confident: boolean
}

export interface DocumentFamilyGroup<T> {
  key: string
  label: string
  priority: number
  items: T[]
  statusCounts: Record<string, number>
  sequence: SequenceSummary
}

interface FamilyRule {
  key: string
  label: string
  priority: number
  patterns: RegExp[]
}

const FAMILY_RULES: FamilyRule[] = [
  { key: 'swo', label: 'SWOs', priority: 10, patterns: [/\\bSWO\\b/i] },
  { key: 'otm', label: 'OTMs', priority: 20, patterns: [/\\bOTM\\b/i] },
  {
    key: 'safety-circular',
    label: 'Safety Circulars',
    priority: 30,
    patterns: [/\\bSC[-_.\\s]*\\d+[A-Z]?\\b/i, /\\bSAFETY\\s+CIRCULARS?\\b/i],
  },
  {
    key: 'train-operation-ti',
    label: 'Train Operation Instructions',
    priority: 40,
    patterns: [/\\bTI[-_.\\s]*\\d+[A-Z]?\\b/i, /\\bTRAIN\\s+OPERATION\\s+TI\\b/i],
  },
  {
    key: 'sm-instruction',
    label: 'Station Management Instructions',
    priority: 50,
    patterns: [/\\bSM\\s*INST(?:RUCTION)?S?\\b/i, /\\bSTATION\\s+MANAGEMENT\\s+INSTRUCTION/i],
  },
  { key: 'atp', label: 'ATP Documents', priority: 60, patterns: [/\\bATP\\b/i] },
  { key: 'mrgr', label: 'MRGR', priority: 70, patterns: [/\\bMRGR\\b/i] },
  { key: 'handbook', label: 'Handbooks', priority: 80, patterns: [/\\bHANDBOOKS?\\b/i] },
  { key: 'addendum', label: 'Addenda', priority: 90, patterns: [/\\bADDEND(?:UM|A)\\b/i] },
  {
    key: 'procedure-order',
    label: 'Procedure Orders',
    priority: 100,
    patterns: [/\\bPROCEDURE\\s+ORDERS?\\b/i, /\\bPO\\s+(?:FOR|OF|TO)\\b/i],
  },
  { key: 'manual', label: 'Other Manuals', priority: 120, patterns: [/\\bMANUALS?\\b/i] },
]

const LEARNED_TOKEN_STOPWORDS = new Set([
  'PDF', 'REV', 'REVISION', 'VER', 'VERSION', 'FINAL', 'DMRC', 'DELHI', 'METRO',
  'RAIL', 'RAILWAY', 'LINE', 'LINES', 'THE', 'AND', 'FOR', 'FROM', 'WITH', 'OF',
  'TO', 'ON', 'IN', 'BY', 'NO', 'DOC', 'DOCUMENT', 'COMPRESSED', 'COPY', 'NEW',
])

function baseName(value: string): string {
  return value.replace(/\\\\/g, '/').split('/').pop() || value
}

function withoutExtension(value: string): string {
  return baseName(value).replace(/\\.pdf$/i, '')
}

export function normalizeFilenameIdentity(value: string): string {
  return baseName(value)
    .normalize('NFKC')
    .replace(/\\s+/g, ' ')
    .trim()
    .toLocaleLowerCase()
}

function knownFamily(filename: string): Omit<FamilyRule, 'patterns'> | null {
  const name = withoutExtension(filename)
  for (const rule of FAMILY_RULES) {
    if (rule.patterns.some((pattern) => pattern.test(name))) {
      return { key: rule.key, label: rule.label, priority: rule.priority }
    }
  }
  return null
}

function acronymCandidates(filename: string): string[] {
  const name = withoutExtension(filename)
  const tokens = name.split(/[^A-Za-z0-9]+/).filter(Boolean)
  return tokens.filter((token) => {
    if (!/^[A-Z][A-Z0-9]{1,7}$/.test(token)) return false
    if (/^\\d+$/.test(token)) return false
    return !LEARNED_TOKEN_STOPWORDS.has(token)
  })
}

function learnedTokenCounts(referenceNames: string[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const filename of referenceNames) {
    if (knownFamily(filename)) continue
    const unique = new Set(acronymCandidates(filename))
    for (const token of unique) counts.set(token, (counts.get(token) || 0) + 1)
  }
  return counts
}

function classifyFamily(filename: string, learnedCounts: Map<string, number>): {
  key: string
  label: string
  priority: number
} {
  const known = knownFamily(filename)
  if (known) return known

  const learned = acronymCandidates(filename)
    .filter((token) => (learnedCounts.get(token) || 0) >= 2)
    .sort((left, right) => {
      const frequency = (learnedCounts.get(right) || 0) - (learnedCounts.get(left) || 0)
      if (frequency) return frequency
      return right.length - left.length
    })[0]

  if (learned) {
    return {
      key: `learned:${learned.toLocaleLowerCase()}`,
      label: `${learned} documents`,
      priority: 500,
    }
  }

  return { key: 'other', label: 'Other documents', priority: 999 }
}

function extractSequenceNumber(filename: string, familyKey: string): number | null {
  const name = withoutExtension(filename)

  const familySpecific: Record<string, RegExp> = {
    'safety-circular': /\\bSC[-_.\\s]*0*(\\d{1,3})[A-Z]?\\b/i,
    'train-operation-ti': /\\bTI[-_.\\s]*0*(\\d{1,3})[A-Z]?\\b/i,
    swo: /\\bSWO[-_.\\s]*0*(\\d{1,3})\\b/i,
    otm: /\\bOTM[-_.\\s]*0*(\\d{1,3})\\b/i,
  }

  const embedded = familySpecific[familyKey]?.exec(name)
  if (embedded) return Number.parseInt(embedded[1], 10)

  // Many operational document families use a leading inventory number: "18. SHD SWO...".
  // Decimal prefixes such as "29.7 SC-29G" are treated as sequence 29 only when no better
  // family-specific code is available.
  const leading = /^\\s*0*(\\d{1,3})(?:\\.\\d+)?(?:[\\s._)-]|$)/.exec(name)
  return leading ? Number.parseInt(leading[1], 10) : null
}

function sequenceSummary<T extends GroupableDocument>(items: T[], familyKey: string): SequenceSummary {
  const observed = Array.from(new Set(
    items
      .map((item) => extractSequenceNumber(item.filename, familyKey))
      .filter((value): value is number => value !== null && Number.isFinite(value)),
  )).sort((a, b) => a - b)

  if (observed.length < 4) {
    return { observed, missing: [], min: observed[0] ?? null, max: observed.at(-1) ?? null, confident: false }
  }

  const min = observed[0]
  const max = observed[observed.length - 1]
  const span = max - min + 1
  const density = observed.length / Math.max(1, span)

  // Only call something a possible gap when the filenames behave like a real sequence.
  // This avoids claiming missing PDFs in sparse numbering systems such as rule/chapter codes.
  const confident = span <= 160 && density >= 0.5
  if (!confident) return { observed, missing: [], min, max, confident: false }

  const present = new Set(observed)
  const missing: number[] = []
  for (let value = min + 1; value < max; value += 1) {
    if (!present.has(value)) missing.push(value)
  }
  return { observed, missing, min, max, confident: true }
}

export function naturalFilenameCompare(left: string, right: string): number {
  return left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' })
}

export function groupDocuments<T extends GroupableDocument>(
  items: T[],
  referenceNames: string[] = items.map((item) => item.filename),
): DocumentFamilyGroup<T>[] {
  const learnedCounts = learnedTokenCounts(referenceNames)
  const grouped = new Map<string, DocumentFamilyGroup<T>>()

  for (const item of items) {
    const family = classifyFamily(item.filename, learnedCounts)
    let group = grouped.get(family.key)
    if (!group) {
      group = {
        key: family.key,
        label: family.label,
        priority: family.priority,
        items: [],
        statusCounts: {},
        sequence: { observed: [], missing: [], min: null, max: null, confident: false },
      }
      grouped.set(family.key, group)
    }
    group.items.push(item)
    const status = item.status || 'unknown'
    group.statusCounts[status] = (group.statusCounts[status] || 0) + 1
  }

  const result = Array.from(grouped.values())
  for (const group of result) {
    group.items.sort((left, right) => naturalFilenameCompare(left.filename, right.filename))
    group.sequence = sequenceSummary(group.items, group.key)
  }

  return result.sort((left, right) => {
    if (left.priority !== right.priority) return left.priority - right.priority
    return left.label.localeCompare(right.label, undefined, { sensitivity: 'base' })
  })
}
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Add smart grouped document inventory to the admin UI.")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    admin = repo / "frontend/src/components/AdminPanel.vue"
    styles = repo / "frontend/src/styles.css"
    helper = repo / "frontend/src/utils/documentGrouping.ts"

    for path in (admin, styles):
        if not path.exists():
            raise SystemExit(f"Not found: {path}")

    admin_text = admin.read_text(encoding="utf-8")
    styles_text = styles.read_text(encoding="utf-8")

    if MARKER in admin_text and MARKER in styles_text and helper.exists():
        print("Smart document grouping patch is already applied.")
        return 0

    for path in (admin, styles):
        backup = path.with_suffix(path.suffix + ".bak-smart-document-grouping")
        if not backup.exists():
            shutil.copy2(path, backup)

    helper.parent.mkdir(parents=True, exist_ok=True)
    if helper.exists() and MARKER not in helper.read_text(encoding="utf-8"):
        helper_backup = helper.with_suffix(".ts.bak-smart-document-grouping")
        if not helper_backup.exists():
            shutil.copy2(helper, helper_backup)
    helper.write_text(HELPER, encoding="utf-8", newline="\n")

    if MARKER not in admin_text:
        admin_text = replace_once(
            admin_text,
            "import type { AdminUser, DocumentRecord, KnowledgeStatus } from '../services/api'\n",
            "import type { AdminUser, DocumentRecord, KnowledgeStatus } from '../services/api'\n"
            "import { groupDocuments, normalizeFilenameIdentity } from '../utils/documentGrouping'\n"
            "\n// SMART_DOCUMENT_GROUPING_PATCH_V1\n",
            "grouping import",
        )

        admin_text = replace_once(
            admin_text,
            "const role = ref<'admin' | 'user'>('user')\n",
            "const role = ref<'admin' | 'user'>('user')\n"
            "const documentSearch = ref('')\n"
            "const documentStatus = ref('all')\n"
            "const familyFilter = ref('all')\n",
            "inventory refs",
        )

        anchor = '''const reprocessableDocumentIds = computed(() =>
  props.documents
    .filter((document) => document.status !== 'processing')
    .map((document) => document.id),
)
'''
        expanded = anchor + r'''

const uploadedFilenameSet = computed(() =>
  new Set(props.documents.map((document) => normalizeFilenameIdentity(document.filename))),
)

const allDocumentGroups = computed(() => groupDocuments(props.documents))

const filteredDocumentGroups = computed(() => {
  const query = documentSearch.value.trim().toLocaleLowerCase()
  return allDocumentGroups.value
    .filter((group) => familyFilter.value === 'all' || group.key === familyFilter.value)
    .map((group) => ({
      ...group,
      items: group.items.filter((document) => {
        if (documentStatus.value !== 'all' && document.status !== documentStatus.value) return false
        return !query || document.filename.toLocaleLowerCase().includes(query)
      }),
    }))
    .filter((group) => group.items.length > 0)
})

const queuedFileRows = computed(() => selectedFiles.value.map((file) => ({
  filename: file.name,
  displayName: fileLabel(file),
  file,
  status: uploadedFilenameSet.value.has(normalizeFilenameIdentity(file.name)) ? 'name-present' : 'new',
})))

const queuedFileGroups = computed(() => groupDocuments(
  queuedFileRows.value,
  [...props.documents.map((document) => document.filename), ...selectedFiles.value.map((file) => file.name)],
))

const queuedNewCount = computed(() =>
  queuedFileRows.value.filter((row) => row.status === 'new').length,
)

const queuedExistingNameCount = computed(() =>
  queuedFileRows.value.filter((row) => row.status === 'name-present').length,
)

function possibleGapLabel(group: { sequence: { confident: boolean; missing: number[]; min: number | null; max: number | null } }): string {
  if (!group.sequence.confident || !group.sequence.missing.length) return ''
  const visible = group.sequence.missing.slice(0, 12).join(', ')
  const suffix = group.sequence.missing.length > 12 ? ` +${group.sequence.missing.length - 12} more` : ''
  return `Possible gaps: ${visible}${suffix}`
}
'''
        admin_text = replace_once(admin_text, anchor, expanded, "inventory computed helpers")

        old_queue = r'''          <div v-if="selectedFiles.length" class="selected-chip-list">
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
'''
        new_queue = r'''          <div v-if="selectedFiles.length" class="queued-document-inventory">
            <div class="queued-inventory-summary">
              <strong>{{ queuedNewCount }} not currently listed</strong>
              <span v-if="queuedExistingNameCount">
                {{ queuedExistingNameCount }} filename{{ queuedExistingNameCount === 1 ? '' : 's' }} already present
              </span>
              <span>Select the master folder to compare its filenames with the uploaded inventory before uploading.</span>
            </div>
            <details
              v-for="group in queuedFileGroups"
              :key="`queued-${group.key}`"
              class="queued-family-group"
              open
            >
              <summary>
                <strong>{{ group.label }}</strong>
                <span>{{ group.items.length }} selected</span>
                <small v-if="possibleGapLabel(group)">{{ possibleGapLabel(group) }}</small>
              </summary>
              <div class="selected-chip-list grouped">
                <span
                  v-for="row in group.items"
                  :key="fileKey(row.file)"
                  class="selected-file-chip"
                  :class="{ existing: row.status === 'name-present' }"
                  :title="row.displayName"
                >
                  <span>{{ row.displayName }}</span>
                  <em>{{ row.status === 'name-present' ? 'name already listed' : 'new' }}</em>
                  <button
                    type="button"
                    :aria-label="`Remove ${row.displayName}`"
                    :disabled="busy"
                    @click="removeSelectedFile(row.file)"
                  >
                    ×
                  </button>
                </span>
              </div>
            </details>
          </div>
'''
        admin_text = replace_once(admin_text, old_queue, new_queue, "queued file grouping")

        old_docs = r'''        <section class="panel-card">
          <div class="panel-heading">
            <div>
              <span class="eyebrow">Persistent documents</span>
              <h2>{{ documents.length }} uploaded PDF{{ documents.length === 1 ? '' : 's' }}</h2>
            </div>
            <button
              class="ghost-action compact"
              :disabled="busy || !reprocessableDocumentIds.length"
              @click="emit('processBatch', reprocessableDocumentIds)"
            >
              Reprocess all
            </button>
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
'''
        new_docs = r'''        <section class="panel-card document-inventory-card">
          <div class="panel-heading">
            <div>
              <span class="eyebrow">Persistent documents</span>
              <h2>{{ documents.length }} uploaded PDF{{ documents.length === 1 ? '' : 's' }}</h2>
            </div>
            <button
              class="ghost-action compact"
              :disabled="busy || !reprocessableDocumentIds.length"
              @click="emit('processBatch', reprocessableDocumentIds)"
            >
              Reprocess all
            </button>
          </div>

          <p class="inventory-guidance">
            Documents are grouped from filename conventions. “Possible gaps” are inferred only inside a
            dense observed number range; the UI cannot know files beyond the highest uploaded number unless
            you select the source/master folder above for comparison.
          </p>

          <div v-if="documents.length" class="family-summary-strip">
            <button
              type="button"
              :class="{ active: familyFilter === 'all' }"
              @click="familyFilter = 'all'"
            >
              <strong>All</strong>
              <span>{{ documents.length }}</span>
            </button>
            <button
              v-for="group in allDocumentGroups"
              :key="`family-filter-${group.key}`"
              type="button"
              :class="{ active: familyFilter === group.key, attention: group.sequence.missing.length > 0 }"
              @click="familyFilter = familyFilter === group.key ? 'all' : group.key"
            >
              <strong>{{ group.label }}</strong>
              <span>{{ group.items.length }}</span>
              <small v-if="group.sequence.missing.length">{{ group.sequence.missing.length }} possible gap{{ group.sequence.missing.length === 1 ? '' : 's' }}</small>
            </button>
          </div>

          <div v-if="documents.length" class="inventory-toolbar">
            <input v-model="documentSearch" type="search" placeholder="Search uploaded filenames…" />
            <select v-model="documentStatus">
              <option value="all">All statuses</option>
              <option value="ready">Ready</option>
              <option value="processing">Processing</option>
              <option value="failed">Failed</option>
              <option value="uploaded">Uploaded</option>
            </select>
          </div>

          <p v-if="!documents.length" class="empty-table">No documents have been uploaded.</p>
          <p v-else-if="!filteredDocumentGroups.length" class="empty-table">No documents match the current filters.</p>
          <div v-else class="document-family-list">
            <details
              v-for="group in filteredDocumentGroups"
              :key="group.key"
              class="document-family-group"
              open
            >
              <summary>
                <div class="family-summary-main">
                  <strong>{{ group.label }}</strong>
                  <span>{{ group.items.length }} shown · {{ group.statusCounts.ready ?? 0 }} ready</span>
                </div>
                <div class="family-summary-meta">
                  <span v-if="group.statusCounts.processing" class="mini-status processing">{{ group.statusCounts.processing }} processing</span>
                  <span v-if="group.statusCounts.failed" class="mini-status failed">{{ group.statusCounts.failed }} failed</span>
                  <span v-if="possibleGapLabel(group)" class="sequence-gap">{{ possibleGapLabel(group) }}</span>
                </div>
              </summary>
              <div class="data-list grouped-document-list">
                <article v-for="document in group.items" :key="document.id" class="data-row document-row">
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
            </details>
          </div>
        </section>
'''
        admin_text = replace_once(admin_text, old_docs, new_docs, "document inventory section")

    if MARKER not in styles_text:
        styles_text += r'''

/* SMART_DOCUMENT_GROUPING_PATCH_V1 */
.document-inventory-card { overflow: visible; }
.inventory-guidance { max-width: 930px; margin: -2px 0 14px !important; }
.family-summary-strip {
  margin: 12px 0 14px;
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding: 2px 1px 6px;
}
.family-summary-strip button {
  min-width: 112px;
  padding: 9px 11px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 2px 8px;
  border: 1px solid #d5e0dc;
  border-radius: 11px;
  color: #465750;
  background: #f8fbfa;
  text-align: left;
  white-space: nowrap;
}
.family-summary-strip button:hover,
.family-summary-strip button.active { border-color: #84b09f; background: #edf7f3; }
.family-summary-strip button.attention:not(.active) { border-color: #ead6ad; background: #fffbf2; }
.family-summary-strip strong { overflow: hidden; font-size: 9px; text-overflow: ellipsis; }
.family-summary-strip span { color: #23664f; font-size: 10px; font-weight: 900; }
.family-summary-strip small { grid-column: 1 / -1; color: #8b671e; font-size: 8px; }
.inventory-toolbar { margin-bottom: 12px; display: grid; grid-template-columns: minmax(220px, 1fr) 170px; gap: 9px; }
.inventory-toolbar input,
.inventory-toolbar select {
  min-height: 38px;
  padding: 8px 11px;
  border: 1px solid #ccd9d4;
  border-radius: 10px;
  outline: none;
  color: var(--ink);
  background: #fff;
  font-size: 10px;
}
.inventory-toolbar input:focus,
.inventory-toolbar select:focus { border-color: #7eaa99; box-shadow: 0 0 0 3px rgba(55, 137, 106, .09); }
.document-family-list { display: grid; gap: 10px; }
.document-family-group { border: 1px solid #dfe7e4; border-radius: 13px; overflow: hidden; background: #fbfdfc; }
.document-family-group > summary {
  min-height: 54px;
  padding: 10px 13px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  list-style: none;
  cursor: pointer;
  background: #f5f9f7;
}
.document-family-group > summary::-webkit-details-marker,
.queued-family-group > summary::-webkit-details-marker { display: none; }
.document-family-group > summary::after,
.queued-family-group > summary::after { content: '+'; color: #638077; font-size: 15px; }
.document-family-group[open] > summary::after,
.queued-family-group[open] > summary::after { content: '−'; }
.family-summary-main { min-width: 0; flex: 1; }
.family-summary-main strong { display: block; color: #29483e; font-size: 11px; }
.family-summary-main span { display: block; margin-top: 3px; color: #7e8d87; font-size: 8px; }
.family-summary-meta { display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: 6px; }
.mini-status,
.sequence-gap { padding: 5px 7px; border-radius: 99px; font-size: 8px; font-weight: 800; white-space: nowrap; }
.mini-status.processing { color: #75520c; background: #fff2d3; }
.mini-status.failed { color: #992e25; background: #fdecea; }
.sequence-gap { color: #855f17; background: #fff5df; }
.grouped-document-list { padding: 0 13px 3px; background: #fff; }
.grouped-document-list .data-row:first-child { border-top: 0; }
.queued-document-inventory { grid-column: 2 / -1; display: grid; gap: 7px; min-width: 0; }
.queued-inventory-summary { display: flex; align-items: center; flex-wrap: wrap; gap: 7px 12px; color: #71817b; font-size: 8px; }
.queued-inventory-summary strong { color: #22654f; font-size: 9px; }
.queued-inventory-summary span:first-of-type { color: #8e6416; font-weight: 800; }
.queued-family-group { border: 1px solid #dce6e2; border-radius: 10px; overflow: hidden; background: #f9fcfb; }
.queued-family-group > summary { padding: 7px 10px; display: flex; align-items: center; gap: 8px; list-style: none; cursor: pointer; }
.queued-family-group > summary strong { color: #365b4e; font-size: 9px; }
.queued-family-group > summary span { color: #809089; font-size: 8px; }
.queued-family-group > summary small { margin-left: auto; color: #8c641b; font-size: 8px; }
.selected-chip-list.grouped { max-height: 130px; padding: 3px 8px 8px; }
.selected-file-chip { gap: 6px; }
.selected-file-chip em { flex: 0 0 auto; color: #2b735a; font-size: 7px; font-style: normal; font-weight: 900; text-transform: uppercase; }
.selected-file-chip.existing { color: #745c2e; background: #fff5df; }
.selected-file-chip.existing em { color: #946819; }

@media (max-width: 900px) {
  .inventory-toolbar { grid-template-columns: 1fr; }
  .document-family-group > summary { align-items: flex-start; flex-wrap: wrap; }
  .family-summary-meta { justify-content: flex-start; }
  .queued-document-inventory { grid-column: 1 / -1; }
}
'''

    admin.write_text(admin_text, encoding="utf-8", newline="\n")
    styles.write_text(styles_text, encoding="utf-8", newline="\n")

    print("Applied smart document grouping patch.")
    print("Changed:")
    print(" - frontend/src/components/AdminPanel.vue")
    print(" - frontend/src/styles.css")
    print("Added:")
    print(" - frontend/src/utils/documentGrouping.ts")
    print("No backend/database migration is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
