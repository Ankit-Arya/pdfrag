# pdfrag per-message chat timestamp patch

This patch displays the time at which each question was asked and the time at
which each assistant response was saved.

## Behavior

- Every user question displays its own date and time.
- Every completed assistant response displays its own date and time.
- Saved chats use the existing PostgreSQL `chat_messages.created_at` value.
- Live chat responses return the server-persisted question and response times.
- Times are formatted in the browser's local timezone and locale.
- The semantic HTML `<time datetime="...">` element is used for accessibility.
- The temporary "typing" indicator does not show a response time because a
  response has not yet been produced.

## Compatibility

The installer makes targeted source edits rather than overwriting whole files,
so it preserves the previous audit-log patch and unrelated local changes. It can
also be applied directly to the current upstream files.

No database migration is needed. The existing `ChatMessage.created_at` column is
reused.

## Files updated

- `backend/app/models.py`
- `backend/app/api.py`
- `frontend/src/services/api.ts`
- `frontend/src/App.vue`
- `frontend/src/components/ChatPanel.vue`

Backups are created beside each changed file using the suffix:

```text
.before-chat-timestamps.bak
```

## Apply on Windows PowerShell

From the extracted patch directory:

```powershell
.\scripts\apply_patch.ps1 C:\path\to\pdfrag
```

## Apply on Linux/macOS

```bash
./scripts/apply_patch.sh /path/to/pdfrag
```

You may check applicability without changing files:

```bash
python3 scripts/apply_patch.py /path/to/pdfrag --dry-run
```

## Rebuild

```bash
docker compose down
docker compose build --no-cache backend frontend
docker compose up -d --force-recreate backend frontend nginx
```

## Verify

1. Sign in and ask a new question.
2. Confirm the question shows a timestamp next to **You**.
3. Confirm the completed response shows a timestamp next to **DMRC Q&A**.
4. Reload the page and reopen the saved chat.
5. Confirm both timestamps remain and match the stored messages.

## Revert

```bash
python3 scripts/revert_patch.py /path/to/pdfrag
```
