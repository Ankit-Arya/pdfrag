export interface FileSummary {
  name: string
  pages: number
  chunks: number
  ocr_pages: number
  tables: number
}

export interface CollectionResponse {
  collection_id: string
  files: FileSummary[]
  total_pages: number
  total_chunks: number
  expires_in_minutes: number
  warnings: string[]
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
  sources: SourceResult[]
  grounded: boolean
  grounding_status:
    | 'verified'
    | 'verified_after_repair'
    | 'insufficient_evidence'
    | 'citation_validation_failed'
    | string
  interpreted_question: string | null
  search_queries: string[]
  request_id?: string | null
}

interface ErrorPayload {
  detail?: string
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`

    try {
      const payload = (await response.json()) as ErrorPayload
      if (payload.detail) detail = payload.detail
    } catch {
      // Keep the status-based message when the response is not JSON.
    }

    throw new Error(detail)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export async function createCollection(
  files: File[],
): Promise<CollectionResponse> {
  const formData = new FormData()

  for (const file of files) {
    formData.append('files', file)
  }

  const response = await fetch('/api/collections', {
    method: 'POST',
    body: formData,
  })

  return parseResponse<CollectionResponse>(response)
}

export async function askQuestion(
  collectionId: string,
  question: string,
  signal?: AbortSignal,
): Promise<AnswerResponse> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      collection_id: collectionId,
      question,
    }),
    signal,
  })

  return parseResponse<AnswerResponse>(response)
}

export async function deleteCollection(
  collectionId: string,
): Promise<void> {
  const response = await fetch(
    `/api/collections/${encodeURIComponent(collectionId)}`,
    {
      method: 'DELETE',
    },
  )

  await parseResponse<void>(response)
}
