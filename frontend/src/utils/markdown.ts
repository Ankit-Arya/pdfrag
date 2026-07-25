const headingPattern = /^(#{1,6})\s+(.+)$/
const unorderedPattern = /^\s*[-*•]\s+(.+)$/
const orderedPattern = /^\s*\d+[.)]\s+(.+)$/
const tableSeparatorPattern =
  /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/

export function renderMarkdown(markdown: string): string {
  const normalized = markdown.replace(/\r\n?/g, '\n').trim()
  if (!normalized) return ''

  const lines = normalized.split('\n')
  const output: string[] = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index]

    if (!line.trim()) {
      index += 1
      continue
    }

    const heading = line.match(headingPattern)
    if (heading) {
      const level = Math.min(4, heading[1].length)
      output.push(
        `<h${level}>${renderInline(heading[2].trim())}</h${level}>`,
      )
      index += 1
      continue
    }

    if (isTableStart(lines, index)) {
      const rendered = renderTable(lines, index)
      output.push(rendered.html)
      index = rendered.nextIndex
      continue
    }

    if (orderedPattern.test(line)) {
      const rendered = renderList(lines, index, true)
      output.push(rendered.html)
      index = rendered.nextIndex
      continue
    }

    if (unorderedPattern.test(line)) {
      const rendered = renderList(lines, index, false)
      output.push(rendered.html)
      index = rendered.nextIndex
      continue
    }

    if (/^\s*---+\s*$/.test(line)) {
      output.push('<hr>')
      index += 1
      continue
    }

    const paragraph: string[] = [line.trim()]
    index += 1

    while (
      index < lines.length
      && lines[index].trim()
      && !isBlockStart(lines, index)
    ) {
      paragraph.push(lines[index].trim())
      index += 1
    }

    output.push(`<p>${renderInline(paragraph.join(' '))}</p>`)
  }

  return output.join('')
}

function isBlockStart(lines: string[], index: number): boolean {
  const line = lines[index]
  return (
    headingPattern.test(line)
    || orderedPattern.test(line)
    || unorderedPattern.test(line)
    || isTableStart(lines, index)
    || /^\s*---+\s*$/.test(line)
  )
}

function isTableStart(lines: string[], index: number): boolean {
  if (index + 1 >= lines.length) return false
  return (
    lines[index].includes('|')
    && tableSeparatorPattern.test(lines[index + 1])
  )
}

function renderTable(
  lines: string[],
  startIndex: number,
): { html: string; nextIndex: number } {
  const header = splitTableRow(lines[startIndex])
  const rows: string[][] = []
  let index = startIndex + 2

  while (
    index < lines.length
    && lines[index].trim()
    && lines[index].includes('|')
  ) {
    rows.push(splitTableRow(lines[index]))
    index += 1
  }

  const headerHtml = header
    .map((cell) => `<th>${renderInline(cell)}</th>`)
    .join('')

  const bodyHtml = rows
    .map((row) => {
      const normalized = [...row]
      while (normalized.length < header.length) normalized.push('')
      return `<tr>${normalized
        .slice(0, header.length)
        .map((cell) => `<td>${renderInline(cell)}</td>`)
        .join('')}</tr>`
    })
    .join('')

  return {
    html:
      `<div class="table-scroll"><table>` +
      `<thead><tr>${headerHtml}</tr></thead>` +
      `<tbody>${bodyHtml}</tbody></table></div>`,
    nextIndex: index,
  }
}

function splitTableRow(line: string): string[] {
  let normalized = line.trim()
  if (normalized.startsWith('|')) normalized = normalized.slice(1)
  if (normalized.endsWith('|')) normalized = normalized.slice(0, -1)
  return normalized.split('|').map((cell) => cell.trim())
}

function renderList(
  lines: string[],
  startIndex: number,
  ordered: boolean,
): { html: string; nextIndex: number } {
  const itemPattern = ordered ? orderedPattern : unorderedPattern
  const tag = ordered ? 'ol' : 'ul'
  const items: string[] = []
  let index = startIndex

  while (index < lines.length) {
    const itemMatch = lines[index].match(itemPattern)
    if (!itemMatch) break

    const body: string[] = [itemMatch[1].trim()]
    const nestedItems: string[] = []
    index += 1

    while (index < lines.length && lines[index].trim()) {
      if (itemPattern.test(lines[index])) break

      const nestedMatch = lines[index].match(unorderedPattern)
      if (ordered && nestedMatch) {
        nestedItems.push(nestedMatch[1].trim())
        index += 1
        continue
      }

      if (
        headingPattern.test(lines[index])
        || isTableStart(lines, index)
        || (ordered && orderedPattern.test(lines[index]))
        || (!ordered && unorderedPattern.test(lines[index]))
      ) {
        break
      }

      body.push(lines[index].trim())
      index += 1
    }

    const nestedHtml = nestedItems.length
      ? `<ul>${nestedItems
          .map((item) => `<li>${renderInline(item)}</li>`)
          .join('')}</ul>`
      : ''

    items.push(
      `<li>${renderInline(body.join(' '))}${nestedHtml}</li>`,
    )

    if (index < lines.length && !lines[index].trim()) {
      let lookahead = index
      while (lookahead < lines.length && !lines[lookahead].trim()) {
        lookahead += 1
      }
      if (lookahead < lines.length && itemPattern.test(lines[lookahead])) {
        index = lookahead
      } else {
        break
      }
    }
  }

  return {
    html: `<${tag}>${items.join('')}</${tag}>`,
    nextIndex: index,
  }
}

function renderInline(value: string): string {
  let rendered = escapeHtml(value)

  rendered = rendered.replace(/`([^`]+)`/g, '<code>$1</code>')
  rendered = rendered.replace(
    /\*\*([^*]+)\*\*/g,
    '<strong>$1</strong>',
  )
  rendered = rendered.replace(
    /__([^_]+)__/g,
    '<strong>$1</strong>',
  )
  rendered = rendered.replace(
    /\[S(\d+)\]/g,
    '<span class="citation">[S$1]</span>',
  )

  return rendered
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}
